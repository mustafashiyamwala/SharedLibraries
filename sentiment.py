from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd

finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'GOOG']

news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker + "&p=d"
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

parsed_data = []
for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.text
        date_data = row.td.text.split(' ')
        if len(date_data) == 21:
            time = date_data[12].replace("\r\n","")
        else:
            date = date_data[12]
            time = date_data[13].replace("\r\n","")
        parsed_data.append([ticker, date, time, title])

df1 = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])


with open("C:/Users/Owner/Downloads/sentiment-main/stopwords.txt", "r") as f:
    stopwords = f.read().split("\n")[:-1]


def preprocess_text(text):
    words = text.split()
    words = [w.lower() for w in words]
    words = [w for w in words if w not in stopwords and w.isalpha()]
    return " ".join(words)
	
df1["text_clean"] = df1["title"].apply(preprocess_text)

lm_dict = pd.read_csv("C:/Users/Owner/Downloads/sentiment-main/Loughran-McDonald_MasterDictionary_1993-2023.csv")
pos_words = lm_dict[lm_dict["Positive"] != 0]["Word"].str.lower().to_list()
neg_words = lm_dict[lm_dict["Negative"] != 0]["Word"].str.lower().to_list()

df1["n"] = df1["text_clean"].apply(lambda x: len(x.split()))
df1["n_pos"] = df1["text_clean"].apply(lambda x: len([w for w in x.split() if w in pos_words]))
df1["n_neg"] = df1["text_clean"].apply(lambda x: len([w for w in x.split() if w in neg_words]))

df1["lm_level"] = df1["n_pos"] - df1["n_neg"]
df1["lm_score1"] = (df1["n_pos"] - df1["n_neg"]) / df1["n"]
df1["lm_score2"] = (df1["n_pos"] - df1["n_neg"]) / (df1["n_pos"] + df1["n_neg"])
CUTOFF = 0.3
df1["lm_sentiment"] = df1["lm_score2"].apply(lambda x: "positive" if x > CUTOFF else "negative" if x < -CUTOFF else "neutral")


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

df1[["finbert_pos", "finbert_neg", "finbert_neu", "finbert_sentiment"]] = (df1["title"].apply(finbert_sentiment).apply(pd.Series))
df1["finbert_score"] = df1["finbert_pos"] - df1["finbert_neg"]


from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from tenacity import retry, stop_after_attempt, RetryError
from langchain_community.chat_models import ChatOllama

class SentimentClassification(BaseModel):
    sentiment: str = Field(..., description="The sentiment of the text", enum=["positive", "negative", "neutral"],)
    score: float = Field(..., description="The score of the sentiment", ge=-1, le=1)
    justification: str = Field(..., description="The justification of the sentiment")
    main_entity: str = Field(..., description="The main entity discussed in the text")
	
@retry(stop=stop_after_attempt(5))
def run_chain(text: str, chain) -> dict:
    return chain.invoke({"news": text}).dict()	
	
def llm_sentiment(text: str, llm) -> tuple[str, float, str, str]:
    parser = PydanticOutputParser(pydantic_object=SentimentClassification)
    prompt = PromptTemplate(template="Describe the sentiment of a text of financial news.\n{format_instructions}\n{news}\n", input_variables=["news"],
    partial_variables={"format_instructions": parser.get_format_instructions()},)	
    chain = prompt | llm | parser
    try:
        result = run_chain(text, chain)
        return (result["sentiment"], result["score"], result["justification"], result["main_entity"],)
    except RetryError as e:
        print(f"Error: {e}")
    return "error", 0, "", ""
		
llama2 = ChatOllama(model="llama2", temperature=0.1)
df1[["llama2_sentiment", "llama2_score", "llama2_justification","llama2_main_entity"]] = (df1["title"].apply(lambda x: llm_sentiment(x, llama2)).apply(pd.Series))




from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.downloader.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()
df1['compound'] = df1['title'].apply(lambda title: vader.polarity_scores(title)['compound'])


df1[["ticker","date","time","title","finbert_sentiment","lm_sentiment","compound"]]
"llama2_sentiment","mixtral_sentiment",


