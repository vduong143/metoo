import sys
import csv
import time
import os
import json
import math
import got
from got.Tweet import tweet_to_dict

def get_tweets(input_dir, output_dir, max_tweets, search_phrase, since, until):
    metoo_users = []
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.endswith('.txt'):
            college = file_name.split('_followers')[0]
            print("Getting tweets from <{}> followers".format(college))
            usernames = open(input_dir+'/'+file_name,'r').readlines()
            usernames = [u.strip() for u in usernames]
            for user in usernames:
                if user in metoo_users:
                    with open(output_dir+'/{}.json'.format(user), 'r') as tweet_file:
                        tweets = json.load(tweet_file)
                    with open(output_dir+'/{}.json'.format(user), 'w') as tweet_file:
                        for tweet in tweets:
                            tweet["college"].append(college)
                        json.dump(tweets, tweet_file, indent=2)
                else:
                    config = got.TweetCriteria()
                    config.setUsername(user)
                    config.setSince(since)
                    config.setUntil(until)
                    config.setQuerySearch(search_phrase)
                    config.setMaxTweets(max_tweets)
                    try:
                        tweets = got.TweetManager.getTweets(config, randsleep=20)
                    except:
                        pass
                    if tweets:
                        metoo_users.append(user)
                        with open(output_dir+'/{}.json'.format(user), 'w') as tweet_file:
                            tweets = [tweet_to_dict(t) for t in tweets]
                            for tweet in tweets:
                                tweet["college"] = [college]
                            json.dump(tweets, tweet_file, indent=2)

            print("Getting tweets from <{}> followers".format(college))

if __name__ == '__main__':
    directory = os.getcwd()
    input_dir = directory + '/college_followers_1'
    output_dir = directory + '/metoo_tweets'
    MAX_TWEETS = sys.maxsize

    get_tweets(input_dir, output_dir, MAX_TWEETS, "metoo", "2017-10-15", "2017-11-15")