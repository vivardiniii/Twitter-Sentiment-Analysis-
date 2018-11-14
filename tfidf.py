from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.svm import SVC
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


test_data = pd.read_csv("pretest.csv")
## for transforming the whole train data ##
#train_counts = count_vect.fit_transform(x_train.values.astype('U'))
#train_tfidf = transformer.fit_transform(train_counts)

## for transforming the test data ##
test_counts = count_vect.transform(test_data['sentimenttext'].values.astype('U'))
test_tfidf = transformer.transform(test_counts)

## fitting the model on the transformed train data ##
#model.fit(train_tfidf,train_data['label'])

## predicting the results ##
predictions1 = model.predict(test_tfidf)
print(predictions1)

pos = 0
neg = 0

for i in predictions1:
    if i == 1:
        pos = pos+1
    elif i == 0:
        neg = neg+1

totpos = pos*100/(pos+neg)
totneg = neg*100/(pos+neg)

print ("Percentage Positive Tweets: ", totpos)
print ("Percentage Negative Tweets: ", totneg)



df = pd.read_csv("pretest.csv")

df ['pred'] = predictions1


df.to_csv('pretest.csv')