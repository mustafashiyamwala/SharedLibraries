import requests
import json
import pandas as pd
import datetime as DT
from pandas import json_normalize

today = DT.date.today()
Days_ago = today - DT.timedelta(days=30)
end = DT.date.today().strftime('%Y-%m-%d')
thirtyDays_ago=Days_ago.strftime('%Y-%m-%d')

share_name='Apple'
url = ('https://newsapi.org/v2/everything?'
       'q={0}&'
       'from={1}&'
       'sortBy=popularity&'
       'apiKey=0e24ed7870d04cc392d0a5804381faf7').format(share_name, thirtyDays_ago)

response = requests.get(url).json()
print(response)

news_df=json_normalize(response['articles'])
news_df['publishedAt']=pd.to_datetime(news_df['publishedAt'], infer_datetime_format = True)
news_df['publishedAtDate']=news_df['publishedAt'].dt.date
news_df['publishedAtTime']=news_df['publishedAt'].dt.time
news_df=news_df.drop_duplicates('description')
news_df=news_df.sort_values(['publishedAt'])
news_df=news_df[['publishedAtDate','publishedAtTime','description','title']]
				      
news_df.to_csv('C:/Users/Owner/Downloads/sentiment-main/NewsData.csv', index=False)


news_df['description']= news_df['description'].apply(lambda x: " ".join(x.lower() for x in x.split()))
news_df['description'].head()

news_df['word_count'] = news_df['description'].apply(lambda x: len(str(x).split(" ")))
news_df[['description','word_count']].head()

from nltk.corpus import stopwords
stop = stopwords.words('english')
news_df['description'] = news_df['description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
news_df[['description','stopwords']].head()

def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)
	
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = text.replace('(ap)', '')
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)
    text = re.sub('[^a-zA-Z ?!]+', '', text)
    text = _removeNonAscii(text)
    text = text.strip()
    return text	

news_df=news_df.reset_index()
Newsdf_description=news_df[['description','publishedAtDate']]

for i in range(0,len(news_df)):
    Newsdf_description['description'][i]=_removeNonAscii(Newsdf_description['description'][i])
	
for i in range(0,len(news_df)):
     Newsdf_description['description'][i]=clean_text(Newsdf_description['description'][i])
	 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
neg=[]
pos=[]
neu=[]
compound=[]

for i in range(0,len(Newsdf_description)):
    #print(i)
    sentence=Newsdf_description['description'][i]
    scores= analyser.polarity_scores(sentence)
    pos.append(scores['pos'])
    neg.append(scores['neg'])
    neu.append(scores['neu'])
    compound.append(scores['compound'])
	
Newsdf_description['positive']=pos 
Newsdf_description['negative']=neg
Newsdf_description['neutral']=neu
Newsdf_description['compound']=compound