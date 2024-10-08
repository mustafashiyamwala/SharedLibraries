import os
import requests
import json
from datetime import datetime

#List 1000 tickers trading in US Exchanges---> One time activity
n = 1000
path_to_file = "C:\\Users\\Owner\\Desktop\\Hackaton\\"
noOfStocksToAnalyse = 10

url = ('https://api.tickertick.com/tickers?'
       'n={0}').format(n)
	
response = requests.get(url).json()

# Dump the 1000 tickers in file
with open(path_to_file + 'list_of_ticker.json', 'w') as f:
  json.dump(response, f, ensure_ascii = False, indent = 4)

# Read the tickers from file
with open(path_to_file + 'list_of_ticker.json', 'r') as f:
  data = json.load(f)

removeDuplicate = { t['company_name'] : t for t in data['tickers'] }.values()
#print(len(removeDuplicate))
  
#Get 1-10 tickers to fetch all the historical data based on news 
firstTenTickers = list(removeDuplicate)[1 : noOfStocksToAnalyse + 1]  

tickers = [ t['ticker'] for t in firstTenTickers ]
 
#Make Landing Directory 
for d in tickers:
   if not os.path.exists(path_to_file + '/' + d):
      os.makedirs(path_to_file + '/' + d)
 
lastId = 0 
for t in tickers: 
   while (lastId != -1):
      query = '(and T:curated tt:' +t+ ')' 
      url = ('https://api.tickertick.com/feed?'
             'q={0}&'
             'last={1}&'
             'n={2}').format(query, lastId, 100)
      response1 = requests.get(url)
      response1.raise_for_status()  
	  # raises exception when not a 2xx response
      if (response1.status_code != 204):
         response1 = response1.json()	 
      if len(list(response1['stories'])) == 0 :
         lastId = 0
         tickers.remove(t)
         print('No more data to download...')
         break		 
      lastTimeStamp = list(response1['stories'])[-1]['time']/1000 
      lastId = response1['last_id']
      fileName = str(datetime.fromtimestamp(lastTimeStamp).strftime('%Y-%m-%d_%H%M%S')) +'.json'
      path = os.path.join(path_to_file + t, fileName)
      print(path)
      with open(path, "w+") as f1:
          json.dump(response1, f1, indent = 4)

# check file exist before reding if it is not available then read from the url and dump that file and perform firther operations
# need to sleep after every 30 request for 1 mins
	
####################################################
from pandas import json_normalize
import pandas as pd

tickers = [ t['ticker'] for t in firstTenTickers ]

#Make Staging Directory 
for d in tickers:
   if not os.path.exists(path_to_file + 'staging\\' + d):
      os.makedirs(path_to_file + 'staging\\' + d)

for t in tickers: 
   for file_name in [file for file in os.listdir(path_to_file + 'landing\\' + t + '\\') if file.endswith('.json')]:
      with open(path_to_file + 'landing\\' + t + '\\' + file_name) as json_file:
         response = json.load(json_file)
      df = json_normalize(response['stories']) 
      df['publishedAt'] = pd.to_datetime(df['time'], unit='ms')
      df['publishedAtDate'] = df['publishedAt'].dt.date
      df['publishedAtTime'] = df['publishedAt'].dt.time
      newsDf = newsDf._append(df, ignore_index = True)
   newsDf = newsDf.drop_duplicates('title').sort_values(['publishedAtDate', 'publishedAtTime'])
   newsDf = newsDf[['id', 'publishedAtDate', 'publishedAtTime', 'title', 'site', 'description']]
   newsDf.to_csv(path_to_file + 'staging\\' + t + '\\' + 'historical_records.csv', index = False)   
   newsDf = pd.DataFrame()      
   

#######################################
import glob
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.downloader.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()

for d in tickers:
   if not os.path.exists(path_to_file + 'processed\\' + d):
      os.makedirs(path_to_file + 'processed\\' + d)
	  
with open(path_to_file +"stopwords.txt", "r") as f:
    stopwords = f.read().split("\n")[:-1]

for t in tickers:
   combinedDf = pd.DataFrame()
   for file in glob.glob(path_to_file  + 'staging\\' + t + '\\*.csv'):
      df = pd.read_csv(file)
      combinedDf = pd.concat([combinedDf, df])
   combinedDf["text_clean"] = combinedDf["title"].apply(preprocess_text)
   lm_dict = pd.read_csv(path_to_file + "Loughran-McDonald_MasterDictionary_1993-2023.csv")
   pos_words = lm_dict[lm_dict["Positive"] != 0]["Word"].str.lower().to_list()
   neg_words = lm_dict[lm_dict["Negative"] != 0]["Word"].str.lower().to_list()
   combinedDf = dictinory_algo(combinedDf, pos_words, neg_words)  
   combinedDf['compound'] = combinedDf['title'].apply(lambda title: vader.polarity_scores(title)['compound'])
   combinedDf.to_csv(path_to_file + 'processed\\' + t + '\\' + 'final_records.csv', index = False) 

def preprocess_text(text):
   words = text.split()
   words = [w.lower() for w in words]
   words = [w for w in words if w not in stopwords and w.isalpha()]
   return " ".join(words)

def dictinory_algo(combinedDf, pos_words, neg_words):
   combinedDf["n"] = combinedDf["text_clean"].apply(lambda x: len(x.split()))
   combinedDf["n_pos"] = combinedDf["text_clean"].apply(lambda x: len([w for w in x.split() if w in pos_words]))
   combinedDf["n_neg"] = combinedDf["text_clean"].apply(lambda x: len([w for w in x.split() if w in neg_words]))
   combinedDf["lm_level"] = combinedDf["n_pos"] - combinedDf["n_neg"]
   combinedDf["lm_score1"] = (combinedDf["n_pos"] - combinedDf["n_neg"]) / combinedDf["n"]
   combinedDf["lm_score2"] = (combinedDf["n_pos"] - combinedDf["n_neg"]) / (combinedDf["n_pos"] + combinedDf["n_neg"])
   CUTOFF = 0.3
   combinedDf["lm_sentiment"] = combinedDf["lm_score2"].apply(lambda x: "positive" if x > CUTOFF else "negative" if x < -CUTOFF else "neutral")
   return combinedDf

###########################################
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import scipy
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def finbert_sentiment(text: str) -> tuple[float, float, float, str]: 
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True,max_length=512)
        outputs = model(**inputs)
        logits = outputs.logits
        scores = {
            k: v
                for k, v in zip(
                model.config.id2label.values(),
                scipy.special.softmax(logits.numpy().squeeze()),
            )
        }
        return (scores["positive"], scores["negative"], scores["neutral"], max(scores, key=scores.get),)

combinedDf[["finbert_pos", "finbert_neg", "finbert_neu", "finbert_sentiment"]] = (combinedDf["title"].apply(finbert_sentiment).apply(pd.Series))
combinedDf["finbert_score"] = combinedDf["finbert_pos"] - dcombinedDff1["finbert_neg"]

##################################
import re
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
stop = stopwords.words('english')

def analyser(combinedDf):
   combinedDf['description']= combinedDf['description'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
   combinedDf['word_count'] = combinedDf['description'].apply(lambda x: len(str(x).split(" ")))
   combinedDf['stopwords'] = combinedDf['description'].apply(lambda x: " ".join(x for x in str(x).split() if x not in stop))
   
   return combinedDf

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

combinedDf=combinedDf.reset_index()
Newsdf_description=combinedDf[['description','publishedAtDate']]

for i in range(0,len(combinedDf)):
    Newsdf_description['description'][i]=_removeNonAscii(Newsdf_description['description'][i])
	


for i in range(0,len(combinedDf)):
     Newsdf_description['descriptionss'][i]=clean_text(Newsdf_description['description'][i])
	 
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






#https://github.com/hczhu/TickerTick-API?tab=readme-ov-file
print(Newsdf_description)


