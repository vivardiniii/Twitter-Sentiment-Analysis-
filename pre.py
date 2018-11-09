import numpy as np
import pandas as pd
import re, string
import warnings

#visualization
import matplotlib.pyplot as plt
import matplotlib
#import seaborn as sns
#from Ipython.display import display
#from mpl_toolkits.basemap import Basemap
from wordcloud import WordCloud, STOPWORDS


#nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from nltk.stem.porter import *
from wordcloud import WordCloud

matplotlib.style.use('ggplot')
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")


tweets = pd.read_csv('new.csv', encoding = "ISO-8859-1")

tweets['handles'] =  ''

#remove handles
#len(tweets['text'])
for i in range(len(tweets['SentimentText'])):
    try:
        tweets['handles'][i] = tweets['SentimentText'].str.split(' ')[i][0]
    except AttributeError:    
        tweets['handles'][i] = 'other'
#len(tweets['text'])

#Preprocessing handles. select handles contains 'RT @'
for i in range(len(tweets['SentimentText'])):
    if tweets['handles'].str.contains('@')[i]  == False:
        tweets['handles'][i] = 'other'
        
# remove URLs, RTs, and twitter handles
for i in range(len(tweets['SentimentText'])):
	
    tweets['SentimentText'][i] = " ".join([word for word in tweets['SentimentText'][i].split()
                                if 'http' not in word and '@' not in word and '<' not in word])
#remove special characters, and numbers
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: re.sub('[!@$:).;,?&]', '', x.lower()))
tweets['SentimentText'] = tweets['SentimentText'].str.replace("[^a-zA-Z#]", " ")

#removes hashtags
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: re.sub(r'\B(\#[a-zA-Z]+\b)', '', x.lower()))


#removes short words 
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

#tokenization
tokens = tweets['SentimentText'].apply(lambda x: x.split())
#tokens.head()


#stemming
#stemmer = PorterStemmer()

#tokens = tokens.apply(lambda x: [stemmer.stem(i) for i in x]) 
#tokens.head()

#putting tokens back together
#for i in range(2):
    #tokens[i] = ' '.join(tokens[i])

#tweets['text'] = tokens

"""hi = tweets['SentimentText']
print (hi)


with open('pre_tweets.csv', "w") as outfile:
	for entries in tweets['SentimentText']:
		outfile.write(entries)
		outfile.write("\n")

"""
with open('pre_tweets.csv', "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(zip(tweets['Sentiment'], tweets['SentimentText']))


import fileinput

for line in fileinput.input(files=['pre_tweets.csv'], inplace=True):
	if fileinput.isfirstline():
		print ('Sentiment,Text')
	print (line),

