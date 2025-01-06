import tweepy
import yaml
import pandas as pd
from datetime import datetime
from typing import Optional, List

def load_api_keys(yaml_path='flagged/X_API_keys.yaml'):
    """Load Twitter API credentials from YAML file."""
    with open(yaml_path, 'r') as file:
        keys = yaml.safe_load(file)
    return keys

def create_api_client():
    """Create and return authenticated Twitter API client."""
    keys = load_api_keys()
    
    client = tweepy.Client(
        bearer_token=keys['bearer_token'],
        consumer_key=keys['api_key'],
        consumer_secret=keys['api_key_secret'],
        wait_on_rate_limit=True
    )
    return client

def get_politician_tweets(
    twitter_handle: str,
    topics: List[str],
    max_results: int = 100,
    save_to_csv: bool = True,
    output_dir: str = 'data'
) -> Optional[pd.DataFrame]:
    """
    Fetch tweets from a politician about specific topics.

    Parameters:
    -----------
    twitter_handle : str
        The politician's Twitter handle (without '@')
    topics : List[str]
        List of topics/keywords to search for
    max_results : int, optional
        Maximum number of tweets to retrieve (default: 100)
    save_to_csv : bool, optional
        Whether to save results to CSV (default: True)
    output_dir : str, optional
        Directory to save CSV file (default: 'data')

    Returns:
    --------
    pd.DataFrame or None
        DataFrame containing the tweets and their metadata, or None if error occurs
    """
    client = create_api_client()
    
    # Build query string from topics
    topics_query = ' OR '.join(f'"{topic}"' for topic in topics)
    query = f"({topics_query}) from:{twitter_handle} -is:retweet"
    
    tweets_data = []
    
    try:
        tweets = client.search_recent_tweets(
            query=query,
            tweet_fields=['created_at', 'public_metrics', 'lang'],
            max_results=max_results
        )
        
        if tweets.data:
            for tweet in tweets.data:
                tweets_data.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'retweets': tweet.public_metrics['retweet_count'],
                    'likes': tweet.public_metrics['like_count'],
                    'replies': tweet.public_metrics['reply_count'],
                    'lang': tweet.lang
                })
    
    except tweepy.TweepyException as e:
        print(f"Error fetching tweets: {e}")
        return None
    
    df = pd.DataFrame(tweets_data)
    
    if save_to_csv and not df.empty:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{output_dir}/{twitter_handle}_tweets_{timestamp}.csv'
        df.to_csv(filename, index=False)
        print(f"Saved tweets to {filename}")
    
    return df

def get_political_reform_tweets(twitter_handle: str) -> Optional[pd.DataFrame]:
    """
    Convenience function to get tweets about political reform from a specific politician.
    
    Parameters:
    -----------
    twitter_handle : str
        The politician's Twitter handle (without '@')
    
    Returns:
    --------
    pd.DataFrame or None
        DataFrame containing the tweets and their metadata
    """
    topics = [
        "reforma política",
        "reformas políticas",
        "reforma eleitoral",
        "PEC",
        "emenda constitucional"
    ]
    
    return get_politician_tweets(twitter_handle, topics)

if __name__ == "__main__":
    # Example usage
    politicians = ["Romario", "jdoriajr", "ArthurLira_"]
    
    for politician in politicians:
        print(f"\nFetching tweets from @{politician}...")
        df = get_political_reform_tweets(politician)
        
        if df is not None and not df.empty:
            print(f"Retrieved {len(df)} tweets about political reform")
            print("\nSample tweets:")
            print(df[['created_at', 'text']].head())
        else:
            print(f"No tweets found or error occurred for @{politician}") 