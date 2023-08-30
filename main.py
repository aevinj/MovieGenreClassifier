import pandas as pd

class MovieGenreClassifier:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        
    def printFirstFew(self):
        print(self.data.head(10))

if __name__ == "__main__":
    data_path = 'C:/Users/aevin/Desktop/wiki_movie_plots_deduped.csv'
    movie_classifier = MovieGenreClassifier(data_path)
    movie_classifier.printFirstFew()
    