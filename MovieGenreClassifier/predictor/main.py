#import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from multiprocessing import Pool, cpu_count
from string import punctuation
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
import joblib
import os

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
    def __init__(self):    
        #dataset = pd.read_csv('wiki_movie_plots_deduped.csv')
        columns_to_keep = ['Genre', 'Plot']
        #self.data = dataset[columns_to_keep]
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.label_encoder = LabelEncoder()
        # if not (os.path.exists(os.path.join('predictor', 'model', 'trained_model.pkl')) and os.path.exists(os.path.join('predictor', 'model', 'tfidf_vectorizer.pkl')) and os.path.exists(os.path.join('predictor', 'model', 'label_encoder.pkl'))):
        #     self.processDataset()
        # else:
        #     print("Existing model found...")
        #     self.load_trained_model()
        self.load_trained_model()
        
    def load_trained_model(self):
        self.trained_model = joblib.load(os.path.join('predictor', 'model', 'trained_model.pkl'))
        # Load the TF-IDF vectorizer as well
        self.vectorizer = joblib.load(os.path.join('predictor', 'model', 'tfidf_vectorizer.pkl'))
        # Finally load the label encoder
        self.label_encoder = joblib.load(os.path.join('predictor', 'model', 'label_encoder.pkl'))
    
    # def processDataset(self):
    #     print("Processing dataset...")
    #     self.data = self.data[self.data['Genre'] != 'unknown']
        
    #     print("Removing stop words and lemmatizing...")
    #     self.data['Plot'] = self.data['Plot'].apply(self.preprocess_text)
        
    #     print("Tokenizing dataset...")
    #     num_cores = cpu_count()
    #     chunk_size = len(self.data) // num_cores
        
    #     with Pool(num_cores) as pool:
    #         self.data = pd.concat(pool.map(tokenize_chunk, [self.data[i:i+chunk_size] for i in range(0, len(self.data), chunk_size)]))
        
    #     print("Feature extraction...")
    #     self.TFIDFMatrix = self.vectorizer.fit_transform(self.data['Tokenized_Plot'].apply(lambda x: ' '.join(x))).toarray()

    #     self.data['Encoded_Genre'] = self.label_encoder.fit_transform(self.data['Genre'])

    #     # Train-Test Split
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         self.TFIDFMatrix, self.data['Encoded_Genre'],
    #         test_size=0.2, random_state=42
    #     )

    #     trained_model = self.trainModel(X_train, y_train)
    #     self.evaluateModel(trained_model, X_test, y_test)
    
    def preprocess_text(self, text):
        # Remove punctuation
        text = ''.join([char for char in text if char not in punctuation])
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
        
        return ' '.join(words)
    
    #def trainModel(self, X_train, y_train):
    #    print("Training the model...")
    #    model = LogisticRegression(max_iter=100)  # You can adjust hyperparameters
    #    model.fit(X_train, y_train)
    #    self.trained_model = model
    #    joblib.dump(model, os.path.join('predictor', 'model', 'trained_model.pkl'))
    #    joblib.dump(self.vectorizer, os.path.join('predictor', 'model', 'tfidf_vectorizer.#pkl'))
    #    joblib.dump(self.label_encoder, os.path.join('predictor', 'model', 'label_encoder.#pkl'))
    #    return model
#
    #def evaluateModel(self, model, X_test, y_test):
    #    print("Evaluating the model...")
    #    y_pred = model.predict(X_test)
    #    accuracy = accuracy_score(y_test, y_pred)
    #    print("Accuracy:", accuracy)
    
    def predict_genre(self, input_plot = "In this uproarious film a bachelor party in Las Vegas spirals into a wild and unforgettable adventure. When three friends wake up with no memory of the previous night, they must retrace their steps to find their missing groom-to-be. Hilarity ensues as they encounter eccentric characters, unexpected challenges, and a trail of chaos that threatens to derail the upcoming wedding. With the clock ticking, the trio races against time to piece together the puzzle of their night of debauchery. Filled with outrageous antics and laugh-out-loud moments, this film showcases the unpredictable nature of friendship and the joys of embracing the unexpected."):
        preprocessed_plot = self.preprocess_text(input_plot)
        tfidf_matrix = self.vectorizer.transform([preprocessed_plot]).toarray()
        encoded_genre = self.trained_model.predict(tfidf_matrix)[0]
        predicted_genre = self.label_encoder.inverse_transform([encoded_genre])[0]
        return predicted_genre

if __name__ == "__main__":
    while True:
        choice = input("Would you like to use your custom plot to test the model? (default is a comedy) Y/N \n")
        if choice.upper() == 'Y' or choice.upper() == 'N':
            if choice.upper() == 'N':
                plot = "In this uproarious film a bachelor party in Las Vegas spirals into a wild and unforgettable adventure. When three friends wake up with no memory of the previous night, they must retrace their steps to find their missing groom-to-be. Hilarity ensues as they encounter eccentric characters, unexpected challenges, and a trail of chaos that threatens to derail the upcoming wedding. With the clock ticking, the trio races against time to piece together the puzzle of their night of debauchery. Filled with outrageous antics and laugh-out-loud moments, this film showcases the unpredictable nature of friendship and the joys of embracing the unexpected."
            else:
                plot = input("Enter movie plot:\n")
            break
            
    movie_classifier = MovieGenreClassifier()
    predicted_genre = movie_classifier.predict_genre(plot)
    
    print("Predicted Genre:", predicted_genre)
    