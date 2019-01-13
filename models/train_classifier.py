import sys
# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import re

def load_data(database_filepath):
	"""
	Loads data from the defined database_filepath
	"""
    # load data from database
    link='sqlite:///'+database_filepath
    engine = create_engine(link)
    conn=engine.connect()
    df = pd.read_sql_table('msg_ctg', conn)
    X = df.message.values
    Y = df.drop(['id','message','genre','original'], axis=1).values.astype(int)
    category_names=list(df.drop(['id','message','genre','original'], axis=1).columns)
    return X, Y, category_names

def tokenize(text):
	"""
	This function converts the text to lower text and removes puctuation and then subsequently tokenizes and lemmatizes them.
	"""
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens=word_tokenize(text)    
    # lemmatize 
    tokens = [lemmatizer.lemmatize(word) for word in tokens]    
    return tokens

def build_model():
	"""
	It builds a pipeline to implement vectorization, tfidf transformation and classification.
	It subsequently implements a grid search on this pipeline and returns the model.
	"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # specify parameters for grid search
    parameters = {
        "clf__estimator__min_samples_split":[2,3],
        "clf__estimator__min_samples_leaf": [1,2]
    }
    # create grid search object
    cv = GridSearchCV(estimator=pipeline,param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
	"""
	This fn. predicts on the test data and prints the calculated score
	"""
    Y_pred = model.predict(X_test)
    labels = np.unique(Y_pred)
    i=-1
    for column in category_names:
        i=i+1
        print ('The scores for {} are \n {}'.format((column),(classification_report(Y_test[:,i], Y_pred[:,i], labels= labels))))
        
def save_model(model, model_filepath):
	"""
	This fn. saves the trained model to the specified filepath
	"""
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
	"""
	Runs all the functions in sequence to load the data and then build, train, test and save the model using the loaded data.
	"""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()