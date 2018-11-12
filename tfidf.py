from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.svm import SVC

df = pd.read_csv(r"pre_tweets.csv", encoding ="ISO-8859-1") 
comments= ' '
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in df.Text: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
          
    for words in tokens: 
        comments = comments + words + ' '
  
  
wordcloud = WordCloud(width = 800, height = 800,
                background_color = 'white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comments) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

import numpy as np

from glob import glob
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

tweets = pd.read_csv('pre_tweets.csv', encoding = "ISO-8859-1")

#splits data, x contains text and y contains labels
x_train, x_test, y_train, y_test = train_test_split(tweets["Text"],tweets["Sentiment"], test_size = 0.2, random_state = 3)

#x = v.fit_transform(df['Review'].values.astype('U')) 


count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)

## for transforming the 80% of the train data ##

X_train_counts = count_vect.fit_transform(x_train.values.astype('U'))
X_train_tfidf = transformer.fit_transform(X_train_counts)

## for transforming the 20% of the train data which is being used for testing ##

x_test_counts = count_vect.transform(x_test.values.astype('U'))
x_test_tfidf = transformer.transform(x_test_counts)

'''
#training vectorizer model
model = SVC(probability=True)
model.fit(X_train_tfidf,y_train)

#testing model and accuracy
predictions = model.predict_proba(x_test_tfidf)
print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:,1])))
'''
#creating a model based on the tfidf of the x text and the corresponding y labels
model = LinearSVC()
model.fit(X_train_tfidf,y_train)

#testing model and accuracy

#predict the x tfidf text's sentiments
predictions = model.predict(x_test_tfidf)
print (predictions)

#test it against the actual labels to get the accuracy 
print (accuracy_score(y_test, predictions))












