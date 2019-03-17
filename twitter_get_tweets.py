import sys
import csv
import time
import os
import json
import math
import got

for files in os.listdir('/Users/vietduong/Desktop/240project/college_celeb_clean/'):
    if files.endswith('celeb.txt'):
        file_name = ('/Users/vietduong/Desktop/240project/college_celeb_clean/'+files)
        f = open(file_name,'r') # opens file with name of '...txt'
        college = files[:(len(files)-10)]
        directory = "/Users/vietduong/Desktop/240project/user_timeline/"+college+"/"
    	try:
    		os.stat(directory)
    	except:
    		os.mkdir(directory)

        users = []
        lines = f.readlines()
        for line in lines:
            #print line
            users.append(line.strip())

        for user in users:
            try:
                tweetCriteria = got.manager.TweetCriteria().setUsername(user).setSince("2017-10-15").setUntil("2017-11-15").setMaxTweets(sys.maxint)
                tweets = got.manager.TweetManager.getTweets(tweetCriteria)

                print('******')
                print "writing to {0}.txt".format(user)
                count = 0

                for tweet in tweets:
                    text = (tweet.text).encode("utf-8").replace("\n"," ").replace("\t"," ")
                    date = tweet.date.strftime("%Y-%m-%d %H:%M")
                    retweets = tweet.retweets
                    favorites = tweet.favorites
                    geo = tweet.geo
                    mentions = tweet.mentions
                    hashtags = tweet.hashtags
                    with open(directory+"{0}.txt".format(user) , 'a') as f:
                        f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(text,date,retweets,favorites,geo,mentions,hashtags))
                    count = count+1
                print(count)
                print('******')
            except:
                print('error')
        os.remove(file_name)
