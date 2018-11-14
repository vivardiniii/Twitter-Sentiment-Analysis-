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
from nltk.corpus import stopwords 

matplotlib.style.use('ggplot')
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")


import fileinput

for line in fileinput.input(files=['pretest.csv'], inplace=True):
    if fileinput.isfirstline():
        print ('SentimentText')
    print (line),

tweets = pd.read_csv('pretest.csv', encoding = "ISO-8859-1")

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
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: re.sub('[!@$:).;,?&#]', ' ', x.lower()))
tweets['SentimentText'] = tweets['SentimentText'].str.replace("[^a-zA-Z]", " ")

#tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: re.sub(r'\B(\#[a-zA-Z]+\b)', '', x.lower()))

#removes short words 
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

#tokenization
tokens = tweets['SentimentText'].apply(lambda x: x.split())
#tokens.head()

tweets['newsSentimentText'] = ''

stop = stopwords.words('english') 
tweets['newSentimentText'] = tweets['SentimentText'].apply(lambda x: " ".join(x for x in x.split() if x not in stop)) 

ps = PorterStemmer()
tweets['newSentimentText'] = tweets['newSentimentText'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() ]))

with open('pretest.csv', "w") as outfile:
	for entries in tweets['newSentimentText']:
		outfile.write(entries)
		outfile.write("\n")

"""


for line in fileinput.input(files=['pretest.csv'], inplace=True):
    if fileinput.isfirstline():
        print ('SentimentText')
    print (line),
"""