import praw
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import time
import os
import csv

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# ==============================
# CONFIGURATION: Reddit credentials
# ==============================
REDDIT_CLIENT_ID = "QEPjjKtnGAbozFbZ5kjvbQ"
REDDIT_CLIENT_SECRET = "N859WU04MDwkKM8AaJtSD6UlET4cdg"
REDDIT_USER_AGENT = "sentiment analyzer by /u/lenin_trader"

# ==============================
# Function: Initialize Reddit API
# ==============================
def init_reddit():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

# ==============================
# Function: Fetch Reddit posts from subreddit
# ==============================
def fetch_reddit_posts(subreddit="stocks", limit=100, days=5):
    reddit = init_reddit()
    subreddit_obj = reddit.subreddit(subreddit)
    cutoff_time = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    posts = []

    print(f"Fetching posts from r/{subreddit}...\n")
    for post in subreddit_obj.new(limit=limit):
        if post.created_utc >= cutoff_time:
            post_date = datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d')
            posts.append({
                'date': post_date,
                'title': post.title,
                'selftext': post.selftext
            })
        time.sleep(0.5)

    return pd.DataFrame(posts)

# ==============================
# Function: Apply VADER Sentiment Analysis
# ==============================
def analyze_sentiment(df):
    def get_sentiment(text):
        if not text:
            return 0
        return sia.polarity_scores(text)["compound"]

    df['content'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
    df['sentiment'] = df['content'].apply(get_sentiment)
    return df

# ==============================
# Function: Aggregate Sentiment by Day
# ==============================
def aggregate_daily_sentiment(df):
    return df.groupby('date')['sentiment'].mean().rename('daily_sentiment').reset_index()

# ==============================
# Function: Merge with indicator dataframe
# ==============================
def merge_with_indicators(indicator_df, sentiment_df):
    indicator_df['date'] = pd.to_datetime(indicator_df.index.date)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    merged = pd.merge(indicator_df, sentiment_df, on='date', how='left')
    merged['daily_sentiment'] = merged['daily_sentiment'].fillna(0)
    return merged

# ==============================
# Example Usage
# ==============================
if __name__ == '__main__':
    try:
        print("Fetching Reddit posts...")
        reddit_df = fetch_reddit_posts(subreddit="stocks", limit=100, days=5)

        if reddit_df.empty:
            print("No posts fetched. Please check API credentials or subreddit.")
        else:
            print(f"\nTotal posts fetched: {len(reddit_df)}")
            print("\nSample titles:")
            print(reddit_df[['date', 'title']].head())

            reddit_df = analyze_sentiment(reddit_df)
            sentiment_df = aggregate_daily_sentiment(reddit_df)
            print("\nSentiment summary by date:")
            print(sentiment_df)

            reddit_df.to_csv("reddit_stock_posts.csv", index=False, encoding="utf-8")
            print("Saved posts to reddit_stock_posts.csv")

            # Generate current 5-day date range for testing
            today = datetime.today()
            start_date = (today - timedelta(days=4)).strftime('%Y-%m-%d')
            example_df = pd.DataFrame(index=pd.date_range(start=start_date, periods=5, freq='D'))
            example_df = merge_with_indicators(example_df, sentiment_df)

            print("\nFinal merged DataFrame:")
            print(example_df[['date', 'daily_sentiment']])

    except Exception as e:
        print("Error occurred:", e)
        print("\nMake sure your Reddit credentials are correct and app type is 'script'.")
