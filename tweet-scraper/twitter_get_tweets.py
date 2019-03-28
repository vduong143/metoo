import sys
import csv
import time
import os
import json
import math
import got
from got.Tweet import tweet_to_dict
import progressbar

def get_tweets(input_dir, output_dir, max_tweets, search_phrase, since, until):
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.endswith('.json'):
            start = time.time()
            print("Getting tweets from <{}>.".format(file_name))
            users = json.load(open(input_dir+'/'+file_name, 'r'))
            tweet_batch = []

            for user in users:
                config = got.TweetCriteria()
                config.setUsername(user["username"])
                config.setSince(since)
                config.setUntil(until)
                config.setQuerySearch(search_phrase)
                if max_tweets:
                    config.setMaxTweets(max_tweets)
                try:
                    tweets = got.TweetManager.getTweets(config)
                except:
                    pass
                if tweets:
                    tweets = [tweet_to_dict(t) for t in tweets]
                    for tweet in tweets:
                        tweet["college"] = user["college"]
                    tweet_batch.extend(tweets)
            stop = time.time()
            print("{} done".format(file_name))
            with open(output_dir+'/'+file_name, 'w') as tweet_file:
                json.dump(tweet_batch, tweet_file, indent=2)
            print("Writing {}> done.".format(file_name))
            print("Sleeping to avoid rate limit.")
            for i in progressbar.progressbar(range(60*15)):
                time.sleep(1)

if __name__ == '__main__':
    directory = os.getcwd()
    input_dir = directory + '/user-batches'
    output_dir = directory + '/metoo-tweets'
    get_tweets(input_dir, output_dir, None, "metoo", "2017-10-15", "2017-11-15")
    