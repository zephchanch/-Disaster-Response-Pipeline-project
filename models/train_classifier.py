import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk import word_tokenize, sent_tokenize, WordNetLemmatizer
import re
import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.ensemble import ExtraTreesClassifier

def load_data(database_filepath):
    """
    Load data from sqlite database

    Params
    ---------------
    database_filepath: String
        path to database file

    Return
    -----------
    X: Array
        Predictor
    Y: Array
        Target
    category_names:List
        List of category names
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('clean_table', engine)
    category_names =df.iloc[:,4:].columns.tolist()
    X = df.message.values
    Y = df.iloc[:,4:].values
    return X, Y, category_names


def tokenize(text):
    """
    tokenize the text data

    Params
    ----------
    text: String
        text to tokenize

    Return
    --------
    clean_tokens: String
        Tokenized text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline and use gridSearch to find the best parameter combination 
    """

    # Build pipeline 
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(),n_jobs=1))
        ])
    # Set parameters for grid search
    param_grid = { 
        'clf__estimator__n_estimators': [200, 500],
        'clf__estimator__max_features': ['auto', 'sqrt', 'log2'],
        'clf__estimator__max_depth' : [4,5,6,7,8],
        'clf__estimator__criterion' :['gini', 'entropy'],
        'tfidf__use_idf': [True, False],
        'vect__ngram_range': [(1,1),(2,1)]
    }  
    # Fine best parameter with grid search. This took us several hours to complete  
    cv = GridSearchCV(estimator=pipeline, param_grid= param_grid, n_jobs = 1, verbose=1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model

    Params:
    -----------
    model: Object
        machine learning model to evaluate
    X_test: numpy.ndarray
        Testing data X
    Y_test: numpy.ndarray
        Testing data Y
    category_names: List
        list of category names
    """
    # predict on test data
    y_pred = model.predict(X_test)
    c_report = classification_report(Y_test, y_pred, target_names = category_names)

    # Accuracy score for each category
    acc = []
    y_tt = pd.DataFrame(Y_test)
    y_pp = pd.DataFrame(y_pred)
    for x in range(len(y_tt.columns)):
        acc.append(round(accuracy_score(y_tt[x],y_pp[x]),3))

    a_score = pd.DataFrame(acc, columns=['Accuracy_score'], index=category_names)

    print("**********************Classification Report****************************\n")
    print(c_report)
    print("\n *********************Accuracy Score **********************************\n")
    print(a_score)

def save_model(model, model_filepath):
    """
    Save the model to disk

    Params:
    -----------
    model: Object
        Machine learning model to save
    model_filepath: String
        file path to save the model
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
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