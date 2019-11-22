#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('C:\\Users\\Admin\\Desktop\Anu_shri\\senti_donald.csv',encoding = 'ISO-8859-1')


# In[3]:


data.head()


# In[4]:


# Feature Extraction
# Number of words

data['word_count'] = data['tweets'].apply(lambda x: len(str(x).split(" ")))
data[['tweets','word_count']].head()


# In[5]:


# Number of characters

data['char_count'] = data['tweets'].str.len() ## this also includes spaces
data[['tweets','char_count']].head()


# In[6]:


# Average word length

def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

data['avg_word'] = data['tweets'].apply(lambda x: avg_word(x))
data[['tweets','avg_word']].head()


# In[7]:


# Number of stopwords

from nltk.corpus import stopwords
stop = stopwords.words('english')

data['stopwords'] = data['tweets'].apply(lambda x: len([x for x in x.split() if x in stop]))
data[['tweets','stopwords']].head()


# In[8]:


# number of special characters

data['hastags'] = data['tweets'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
data[['tweets','hastags']].head()


# In[9]:


# number of numerics

data['numerics'] = data['tweets'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
data[['tweets','numerics']].head()


# In[10]:


# number of uppercase words

data['upper'] = data['tweets'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
data[['tweets','upper']].head()


# In[11]:


# Pre processing
# Lower case

data['tweets'] = data['tweets'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['tweets'].head()


# In[12]:


# removing punctuations

data['tweets'] = data['tweets'].str.replace('[^\w\s]','')
#data['tweets'] = data['tweets'].str.replace('[http://t.co/]','')
#data['tweets'] = data['tweets'].str.replace('[a-zA-Z0-9]','')

data['tweets'].head()


# In[13]:


# Removal of stop words


from nltk.corpus import stopwords
stop = stopwords.words('english')
data['tweets'] = data['tweets'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['tweets'].head()


# In[14]:


# Common word removal

freq = pd.Series(' '.join(data['tweets']).split()).value_counts()[:10]
freq


# In[15]:


freq = list(freq.index)
data['tweets'] = data['tweets'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
data['tweets'].head()


# In[16]:


# Tokenization

from nltk import tokenize
from textblob import TextBlob


# In[17]:


TextBlob(data['tweets'][1]).words


# In[18]:


# Stemming

from nltk.stem import PorterStemmer
st = PorterStemmer()
data['tweets'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


# In[19]:


# Lemmatization


import nltk
from textblob import Word
nltk.download('wordnet')
data['tweets'] = data['tweets'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['tweets'].head()


# In[22]:


data['tweets'] = data['tweets'].str.replace('[\d+]','')
data['tweets'].head()
#data['tweets'] = data['tweets'].str.replace('[a-zA-Z0-9]','')


# In[31]:


# Advance text processing
# N-grams
TextBlob(data['tweets'][0]).ngrams(2)


# In[32]:


#Term frequency

tf1 = (data['tweets'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1


# In[33]:


# Inverse document frequency

import numpy as np
for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(data.shape[0]/(len(data[data['tweets'].str.contains(word)])))

tf1


# In[34]:


# TF-IDF
# multiplication of TF and IDF

tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1


# In[35]:


# Bag of words

from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
data_bow = bow.fit_transform(data['tweets'])
data_bow


# In[36]:


# sentiment analysis

data['tweets'][:5].apply(lambda x: TextBlob(x).sentiment)


# In[37]:


data['sentiment'] = data['tweets'].apply(lambda x: TextBlob(x).sentiment[0] )
data[['tweets','sentiment']].head()


# In[38]:


data['sentiment'].head()


# In[39]:


import matplotlib.pyplot as plt
plt.hist(data.sentiment,bins=3,align='mid')

plt.xlabel('sentiment of tweets')
plt.title('distribution of sentiment')
plt.show()


# In[40]:


# EDA

# normal function example
def my_normal_function(x):
    return x**2 + 10
# lambda function example
my_lambda_function = lambda x: x**2 + 10


# In[41]:


# make a new column to highlight retweets
data['is_retweet'] = data['tweets'].apply(lambda x: x[:2]=='RT')
data['is_retweet'].sum()  # number of retweets


# In[42]:


# number of unique retweets
data.loc[data['is_retweet']].tweets.unique().size


# In[43]:


# 10 most repeated tweets
data.groupby(['tweets']).size().reset_index(name='counts')  .sort_values('counts', ascending=False).head(10)


# In[44]:


# number of times each tweet appears
counts = data.groupby(['tweets']).size()           .reset_index(name='counts')           .counts

# define bins for histogram
my_bins = np.arange(0,counts.max()+2, 1)-0.5

# plot histogram of tweet counts
plt.figure()
plt.hist(counts, bins = my_bins)
plt.xlabels = np.arange(1,counts.max()+1, 1)
plt.xlabel('copies of each tweet')
plt.ylabel('frequency')
plt.yscale('log', nonposy='clip')
plt.show()


# In[45]:


def find_retweeted(tweets):
    '''This function will extract the twitter handles of retweed people'''
    return re.findall('(?<=RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweets)

def find_mentioned(tweets):
    '''This function will extract the twitter handles of people mentioned in the tweet'''
    return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweets)  

def find_hashtags(tweets):
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweets) 


# In[46]:


# two sample tweets
my_tweet = 'RT @our_codingclub: Can @you find #all the #hashtags?'
my_other_tweet = 'Not a retweet. All views @my own'


# In[47]:


# make new columns for retweeted usernames, mentioned usernames and hashtags


import re
data['retweeted'] = data.tweets.apply(find_retweeted)
data['mentioned'] = data.tweets.apply(find_mentioned)
data['hashtags'] = data.tweets.apply(find_hashtags)


# In[48]:


# take the rows from the hashtag columns where there are actually hashtags
hashtags_list_df = data.loc[
                       data.hashtags.apply(
                           lambda hashtags_list: hashtags_list !=[]
                       ),['hashtags']]


# In[49]:


# create dataframe where each use of hashtag gets its own row
flattened_hashtags_df = pd.DataFrame(
    [hashtag for hashtags_list in hashtags_list_df.hashtags
    for hashtag in hashtags_list],
    columns=['hashtag'])


# In[50]:


# number of unique hashtags
flattened_hashtags_df['hashtag'].unique().size


# In[51]:


# cleaning unsructured text data


import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


# In[53]:


def remove_links(tweet):
    '''Takes a string and removes web links from it'''
    tweet = re.sub(r'http\S+', '', tweet) # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet) # rempve bitly links
    tweet = tweet.strip('[link]') # remove [links]
    return tweet

def remove_users(tweet):
    '''Takes a string and removes retweet and @user information'''
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove retweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove tweeted at
    return tweet


# In[54]:


my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

# cleaning master function
def clean_tweet(tweet, bigrams=False):
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = tweet.lower() # lower case
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) # strip punctuation
    tweet = re.sub('\s+', ' ', tweet) #remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet) # remove numbers
    tweet_token_list = [word for word in tweet.split(' ')
                            if word not in my_stopwords] # remove stopwords

    tweet_token_list = [word_rooter(word) if '#' not in word else word
                        for word in tweet_token_list] # apply word rooter
    if bigrams:
        tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                            for i in range(len(tweet_token_list)-1)]
    tweet = ' '.join(tweet_token_list)
    return tweet


# In[1]:


data['clean_tweet'] = data.tweets.apply(clean_tweet)
data['clean_tweet'].head()


# In[57]:


import matplotlib.pyplot as plt

all_words = ' '.join([text for text in data['clean_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=1000, height=700, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[58]:


# Applying topic modelling


from sklearn.feature_extraction.text import CountVectorizer

# the vectorizer object will be used to transform text to vector form
vectorizer = CountVectorizer(max_df=1.0, min_df=1, token_pattern='\w+|\$[\d\.]+|\S+')

# apply transformation
tf = vectorizer.fit_transform(data['clean_tweet']).toarray()

# tf_feature_names tells us what word each column in the matric represents
tf_feature_names = vectorizer.get_feature_names()


# In[59]:


from sklearn.decomposition import LatentDirichletAllocation

number_of_topics = 10

model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)


# In[60]:


model.fit(tf)


# In[61]:


def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


# In[62]:


no_top_words = 10
display_topics(model, tf_feature_names, no_top_words)

