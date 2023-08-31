import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from multiprocessing import Pool, cpu_count
from string import punctuation
from nltk.stem import WordNetLemmatizer

pd.set_option('display.max_colwidth', None)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def tokenize_chunk(chunk):
    chunk['Tokenized_Plot'] = chunk['Plot'].apply(word_tokenize)
    return chunk

class MovieGenreClassifier:
    def __init__(self, data_path):
        dataset = pd.read_csv(data_path)
        columns_to_keep = ['Genre', 'Plot']
        self.data = dataset[columns_to_keep]
        self.lemmatizer = WordNetLemmatizer()
        
    def processDataset(self):
        print("Processing dataset...")
        print("Removing duplicates and empty values...")
        self.data.drop_duplicates(inplace=True) #remove duplicates
        self.data.dropna(inplace=True)  #remove empty values
        
        print("Preprocessing plot text...")
        self.data['Plot'] = self.data['Plot'].apply(self.preprocess_text)
        
        print("Tokenizing dataset...")
        num_cores = cpu_count()
        chunk_size = len(self.data) // num_cores
        
        with Pool(num_cores) as pool:
            self.data = pd.concat(pool.map(tokenize_chunk, [self.data[i:i+chunk_size] for i in range(0, len(self.data), chunk_size)]))
        print("Done...\n ")
    
    def preprocess_text(self, text):
        # Remove punctuation
        text = ''.join([char for char in text if char not in punctuation])
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
        
        return ' '.join(words)
    
    def printFirstFew(self):
        print(self.data.head(1))

if __name__ == "__main__":
    data_path = 'C:/Users/aevin/Desktop/wiki_movie_plots_deduped.csv'
    movie_classifier = MovieGenreClassifier(data_path)
    movie_classifier.processDataset()
    movie_classifier.printFirstFew()
    