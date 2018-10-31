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



tweets = pd.read_csv('tweets_all.csv', encoding = "ISO-8859-1")


tweets['handles'] =  ''


#remove handles
for i in range(len(tweets['text'])):
    try:
        tweets['handles'][i] = tweets['text'].str.split(' ')[i][0]
    except AttributeError:    
        tweets['handles'][i] = 'other'
#len(tweets['text'])

#Preprocessing handles. select handles contains 'RT @'
for i in range(len(tweets['text'])):
    if tweets['handles'].str.contains('@')[i]  == False:
        tweets['handles'][i] = 'other'
        
# remove URLs, RTs, and twitter handles
for i in range(len(tweets['text'])):
    tweets['text'][i] = " ".join([word for word in tweets['text'][i].split()
                                if 'http' not in word and '@' not in word and '<' not in word])
#remove special characters, and numbers
tweets['text'] = tweets['text'].apply(lambda x: re.sub('[!@$:).;,?&]', '', x.lower()))
tweets['text'] = tweets['text'].str.replace("[^a-zA-Z#]", " ")

#removes hashtags
tweets['text'] = tweets['text'].apply(lambda x: re.sub(r'\B(\#[a-zA-Z]+\b)', '', x.lower()))



#removes short words 
tweets['text'] = tweets['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

#tokenization
tokens = tweets['text'].apply(lambda x: x.split())
#tokens.head()


#stemming
#stemmer = PorterStemmer()

#tokens = tokens.apply(lambda x: [stemmer.stem(i) for i in x]) 
#tokens.head()

#putting tokens back together
#for i in range(2):
    #tokens[i] = ' '.join(tokens[i])

#tweets['text'] = tokens




#hi = tweets['text']
#print (hi)


with open('pre_tweets.csv', "w") as outfile:
	for entries in tweets['text']:
		outfile.write(entries)
		outfile.write("\n")


