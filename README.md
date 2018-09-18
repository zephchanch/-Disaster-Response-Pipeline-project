# Disaster Response Pipeline Project

### Synopsis:
This project consist of the follwoing parts:
- **ETL Pipeline**: Here, we build load the different datasets, and write a cleaning pipeline to clean, merge and store the data in a SQLite database
- **ML Pipeline**: Here, we write a machine learning pipeline that load data from SQLite database, split the dataset into training and test sets, build a text processing and machine learning pipeline, train and tune the model using GridSearchCV, output results on the test set and export the final model as a pickle file
- **Flask Web App**: Here, we modify an already prepared web app by adding data visualizations using Plotly. Also, we modify file paths for database and model as needed.

### Files Structure:
- **data/**: This folder has the datasets, the database and the file `process_data.py`
    - **data/process_data.py**: This file contain ETL Pipeline
    - **data/DisasterResponse.db**: This file is the sqlite dataset which contain the clean dataset
    - **data/disaster_categories.csv**: This file contain the disaster categories
    - **data/disaster_messages.csv**: This file contain the disaster messages
- **models/**: This folder has the pickle database and the file `train_classifier.py`
    - **models/train_classifier.py**: This file contain the ML Pipeline
    - **models/finalized_model.pkl**: This is a pickle database containing the ML model
- **app/**: This folder has the template folder and the file `run.py`
    - **app/run.py**: This file starts the web app
    - **app/templates/**: This folder contain template files.
- **Images/**: This folder contain images
 
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/finalized_model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Summary:

1. ETL Pipeline

This section consisted of just cleaning and preparing the dataset for machine learning. The dataset consisted of 2 seperate dataset which was cleaned, then merged in to a single dataset and stored in a SQLite database.

1. ML Pipeline

Our model pipeline has the follwoing structure:
```
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(),n_jobs=1))
        ])
```
We used the TF-IDF feature extraction technique with multi-output classification to enable multi-target classification. The model trained with **75%** accuracy.

To improve our model, we used **GridSearchCV** to find the best parameters combination for the model. However, this process took us several hours to complete
 
1. Data Visualization
1.1 Distribution of Message Genres related to natural disasters

Data visualization was added to the web app using Plotify. And the visualization was on message genres which are related to natural disasters like(**Weather Related, Floods, Storm, Earthquake**). The aim of this visualization was to show the distribution of these message genres. 

![title](Images/messageGenres.png)
**Figure 1**

From the graph above, we see clearly that messages related to earthquake record more counts in every genre type. Messages related to floods are almost not reported in social genre.

### Note:
- The GridSearchCV is too expensive. For this project, it took us several hours to complete

