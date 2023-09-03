# Movie Genre Classifier

Made solely by Aevin Jais

## Project Description

This is my first project in the realm of machine learning. I utilise vectorization and Term Frequency-Inverse Document Frequency to train a logistic regression model which then is used to determine the genre (movies) of a given inputted text.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#Usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Features

The user can choose whether to use a default movie plot or to use their own custom plot. In the event that the user deletes the model or does not clone the entire project, the code creates and trains the model using the dataset included in the project. This will generate 3 files: label_encoder, tfidf_vectorizer and trained_model. These files are saved locally to the user so that the model can be used again without the need for rebuilding the model.

The default plot is as follows: (generated by ChatGPT) - running the model on this input will result in comedy which is the correct answer.

"In this uproarious film a bachelor party in Las Vegas spirals into a wild and unforgettable adventure. When three friends wake up with no memory of the previous night, they must retrace their steps to find their missing groom-to-be. Hilarity ensues as they encounter eccentric characters, unexpected challenges, and a trail of chaos that threatens to derail the upcoming wedding. With the clock ticking, the trio races against time to piece together the puzzle of their night of debauchery. Filled with outrageous antics and laugh-out-loud moments, this film showcases the unpredictable nature of friendship and the joys of embracing the unexpected."

## Installation

Firstly, the user must have Python and pip installed on their machine. From there they will need to install the following libraries:

- nltk
- scikit-learn
- pandas
- django

These can be installed by running:

***pip install -r requirements.txt***

or alternatively,

Each can be installed by running the following line in the terminal: 

***pip install [enter library] .***

## Usage

![image](https://github.com/aevinj/MovieGenreClassifier/assets/64698098/5702a3e8-5c87-4f3a-82dd-52049cb76c68)

![image](https://github.com/aevinj/MovieGenreClassifier/assets/64698098/aa05fb8e-693b-4c4f-8b40-511cd6afa4b1)

## Technologies Used

Language:
 - Python

Libraries:
 - pandas
 - nltk
 - sci-kit
 - django

## Contributing

If you're open to contributing to this project please contact me via email: ajjaevinjais@gmail.com.

## License

Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)

## Contact Information

Email: ajjaevinjais@gmail.com
IG: aevin.j

## Acknowledgments

Credits to kaggle user JustinR for the dataset used to train the models.

Can be access via the following link:

https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots
