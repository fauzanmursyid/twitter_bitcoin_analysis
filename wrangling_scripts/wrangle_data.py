import pandas as pd
import plotly.graph_objs as go
import datetime
from datetime import date
import tweepy as tw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from collections import Counter
import statistics as stats

# TODO: Scroll down to line 157 and set up a fifth visualization for the data dashboard

def return_figures():
    """Creates four plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

  # first chart plots arable land from 1990 to 2015 in top 10 economies 
  # as a line chart
    consumer_key=''
    consumer_secret=''
    access_key=''
    access_secret=''
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
    # Define the search term and the date_since date as variables
    search_words = "#bitcoin"
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    tweets_today = tw.Cursor(api.search,
              q=search_words,
              lang="en",
              since=today).items(150)
    tweets_yesterday = tw.Cursor(api.search,
                   q=search_words,
                   lang="en",
                   since=yesterday,
                   until=today,
                   result_type="recent"
                   ).items(150)
    today_tweet_list = []
    for tweet in tweets_today:    
        today_tweet_list.append(tweet.text)
    df_today = pd.DataFrame(today_tweet_list,columns=["tweets"])
    yesterday_tweet_list = []
    for tweet in tweets_yesterday:    
        yesterday_tweet_list.append(tweet.text)
    df_yesterday = pd.DataFrame(yesterday_tweet_list,columns=["tweets"])
    X=df_today['tweets']
    stop_words=stopwords.words('english')
    stemmer=PorterStemmer()
    import re
    cleaned_data=[]
    for i in range(len(X)):
        tweet=re.sub('[^a-zA-Z]',' ',X.iloc[i])
        tweet=tweet.lower().split()
        tweet=[stemmer.stem(word) for word in tweet if (word not in stop_words)]
        tweet=' '.join(tweet)
        cleaned_data.append(tweet)
    df_today = pd.DataFrame(cleaned_data,columns=["tweets_cleaned"])
    X=df_yesterday['tweets']
    stop_words=stopwords.words('english')
    stemmer=PorterStemmer()
    import re
    cleaned_data=[]
    for i in range(len(X)):
        tweet=re.sub('[^a-zA-Z]',' ',X.iloc[i])
        tweet=tweet.lower().split()
        tweet=[stemmer.stem(word) for word in tweet if (word not in stop_words)]
        tweet=' '.join(tweet)
        cleaned_data.append(tweet)
    df_yesterday = pd.DataFrame(cleaned_data,columns=["tweets_cleaned"])
    sid = SentimentIntensityAnalyzer()
    new_df_today = df_today['tweets_cleaned'].apply(lambda x: sid.polarity_scores(x))
    compound_sentiment_today = []
    for i in range(0,len(new_df_today)):
        compound_sentiment_today.append(new_df_today.loc[i]['compound'])    
    new_df_yesterday = df_yesterday['tweets_cleaned'].apply(lambda x: sid.polarity_scores(x))
    compound_sentiment_yesterday = []
    for i in range(0,len(new_df_yesterday)):
        compound_sentiment_yesterday.append(new_df_yesterday.loc[i]['compound'])
    yesterday_distribution = Counter(compound_sentiment_yesterday)
    today_distribution = Counter(compound_sentiment_today)    
    
    negative_sentiment_today = []
    for i in range(0,len(new_df_today)):
        negative_sentiment_today.append(new_df_today.loc[i]['neg'])      
    hist_negative = stats.mean(negative_sentiment_today)
    
    positive_sentiment_today = []
    for i in range(0,len(new_df_today)):
        positive_sentiment_today.append(new_df_today.loc[i]['pos'])      
    hist_positive = stats.mean(positive_sentiment_today)
    
    hist_x = ['positive','negative']
    hist_y = [hist_positive, hist_negative]
    
    graph_one = []
    graph_one.append(
        go.Scatter(
            x = list(yesterday_distribution.keys()),
            y = list(yesterday_distribution.values()),
            mode = 'markers',
            name = yesterday            
        )
    )
    graph_one.append(
        go.Scatter(
            x = list(today_distribution.keys()),
            y = list(today_distribution.values()),
            mode = 'markers',
            name = today            
        )
    )    

    layout_one = dict(title = 'Bitcoint Sentiment Polarity',
                xaxis = dict(title = 'Positivity'),
                yaxis = dict(title = 'Polarity Score'),
                )
    graph_two = []
    graph_two.append(
        go.Bar(
            x = hist_x,
            y = hist_y,            
        )
    )
    
    layout_two = dict(title = 'Todays Positive vs Negative Sentiment',
                xaxis = dict(title = 'Sentiment'),
                yaxis = dict(title = 'Number of Tweets'),
                )
    
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))    
    figures.append(dict(data=graph_two, layout=layout_two))

    return figures
