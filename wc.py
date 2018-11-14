from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np

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










