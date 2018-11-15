from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
'''
df = pd.read_csv(r"pretest.csv", encoding ="ISO-8859-1") 
comments= ' '
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in df.sentimenttext: 
      
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
'''
df = pd.read_csv('pretest.csv',encoding ="ISO-8859-1")
stopwords = set(STOPWORDS) 


# 1 for positive wc and 0 for negative wc
neg_tweets = df[df.pred == 1]
neg_string = []
for t in neg_tweets.sentimenttext:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')


wordcloud = WordCloud(width = 800, height = 800,
                background_color = 'white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(neg_string) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.tight_layout(pad = 0) 

plt.show() 







