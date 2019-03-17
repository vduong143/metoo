import os
import sys
import pandas as pd

directory = '/home/vietduong/Desktop/metoo_project/'
college_file = open(directory+'colleges.txt','r')
college_lines = college_file.readlines()
colleges = []
for line in college_lines:
    colleges.append(line.strip())

metoo_tweets = pd.DataFrame(columns=('user','sent'))

metoo_user_file = open(directory+'metoo_users.txt','r')
metoo_user_lines = metoo_user_file.readlines()
metoo_users = []
for line in metoo_user_lines:
    metoo_users.append(line.strip())

# metoo_users_2 = []

i = 0
for college in colleges:
    print(college)
    for file_name in os.listdir(directory+'user_timeline/{}/'.format(college)):
        tweet_data_path = directory+'user_timeline/{}/'.format(college)+file_name
        tweet_data = []
        tweet_file = open(tweet_data_path, 'r')
        # tweet_lines = tweet_file.readline()
        user = file_name[:(len(file_name)-4)]
        if user in metoo_users:
            # tweets = ''
            tweet_id = 0
            for line in tweet_file:
                column = line.split('\t')
                # print column[0]
                tweet = column[0].lower().replace(',',' ')
                if 'metoo' in tweet:
                    metoo_tweets.loc[i,'user'] = user
                    metoo_tweets.loc[i,'tweet'] = tweet
                    i = i + 1
                    print(i)

                    # tweets = tweets + ' ' + tweet
                    # metoo_tweets.loc[i,'date'] = column[1]
                    # metoo_tweets.loc[i,'college'] = college

                    # if user not in metoo_users_2:
                    #     metoo_users_2.append(user)
            metoo_users.remove(user)



metoo_tweets.to_csv(directory+'metoo_tweets.csv',index=False)

# count = 0
# for user in metoo_users_2:
#     with open(directory+'metoo_users_2.txt', 'a') as f:
#         f.write(user+'\n')
#         count = count + 1
#         print count
