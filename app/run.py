import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
	tokens = word_tokenize(text)
	lemmatizer = WordNetLemmatizer()

	clean_tokens = []
	for tok in tokens:
		clean_tok = lemmatizer.lemmatize(tok).lower().strip()
		clean_tokens.append(clean_tok)

	return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('clean_table', engine)

# load model
model = joblib.load("../models/finalized_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

	# extract data needed for visuals
	# TODO: Below is an example - modify to extract data for your own visuals
	#genre_counts = df.groupby('genre').count()['message']
	genre_counts = df.groupby('genre').count()["message"]
	genre_names = list(genre_counts.index)

	# Get categories which are related to natural disasters
	category1 =[ df.loc[df['weather_related']==1].loc[df['genre']=='direct'].count()['message'],
	                        df.loc[df['weather_related']==1].loc[df['genre']=='news'].count()['message'],
	                        df.loc[df['weather_related']==1].loc[df['genre']=='social'].count()['message']]

	category2 =[ df.loc[df['floods']==1].loc[df['genre']=='direct'].count()['message'],
	                        df.loc[df['floods']==1].loc[df['genre']=='news'].count()['message'],
	                        df.loc[df['floods']==1].loc[df['genre']=='social'].count()['message']]

	category3 =[ df.loc[df['storm']==1].loc[df['genre']=='direct'].count()['message'],
	                        df.loc[df['storm']==1].loc[df['genre']=='news'].count()['message'],
	                        df.loc[df['storm']==1].loc[df['genre']=='social'].count()['message']]

	category4 =[ df.loc[df['earthquake']==0].loc[df['genre']=='direct'].count()['message'],
	                        df.loc[df['earthquake']==0].loc[df['genre']=='news'].count()['message'],
	                        df.loc[df['earthquake']==0].loc[df['genre']=='social'].count()['message']]

	# create visuals
	# TODO: Below is an example - modify to create your own visuals
	graphs = [
	{
	    'data': [
	        Bar(
	            x=genre_names,
	            y=genre_counts
	        ),

	    ],

	    'layout': {
	        'title': 'Distribution of Message Genres with a Bar Chart',
	        'yaxis': {
	            'title': "Count"
	        },
	        'xaxis': {
	            'title': "Genre"
	        }
	    }
	},
	{
	    'data': [
	        Bar(
	            x=genre_names,
	            y=category1,
	            name='Weather Related'
	        ),
	        Bar(
	            x=genre_names,
	            y=category2,
	            name="Floods"
	        ),
	        Bar(
	            x=genre_names,
	            y=category3,
	            name="Storm"
	        ),
	        Bar(
	            x=genre_names,
	            y=category4,
	            name="Earthquake"
	        ),

	    ],

	    'layout': {
	        'title': 'Distribution of Message Genres related to natural disasters',
	        'yaxis': {
	            'title': "Count"
	        },
	        'xaxis': {
	            'title': "Genre"
	        }
	    }
	},
	{
		'data':[
			{
				"values": genre_counts,
				"labels": genre_names,
				 "domain": {"x": [0, .5]},
      			"name": "Genre",
      			"hoverinfo":"label+percent+name",
      			"hole": .3,
      			"type":"pie"
				}


		],
	    'layout': {
	        'title': 'Distribution of Message Genres with a Pie Chart',
	        'yaxis': {
	            'title': "Count"
	        },
	        'xaxis': {
	            'title': "Genre"
	        }
	    }
	}	
	]

	# encode plotly graphs in JSON
	ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
	graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

	# render web page with plotly graphs
	return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
	# save user input in query
	query = request.args.get('query', '') 

	# use model to predict classification for query
	classification_labels = model.predict([query])[0]
	classification_results = dict(zip(df.columns[4:], classification_labels))

	# This will render the go.html Please see that file. 
	return render_template(
		'go.html',
		query=query,
		classification_result=classification_results
	)


def main():
	app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
	main()