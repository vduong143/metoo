import re
import sys
import os
import pandas as pd

directory='/home/vietduong/Desktop/metoo_project/'
tweets = pd.read_csv(directory+'metoo_tweets.csv',dtype='str')

df = tweets['tweet']
data = df.values.tolist()
data = [str(sent) for sent in data]

data = [sent.strip() for sent in data]

data = [re.sub("@ ", "@", sent) for sent in data]
data = [re.sub("# ", "#", sent) for sent in data]

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
data = [re.sub('\S*#\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove links
data = [re.sub(r"http\S+", "", sent) for sent in data]

data = [re.sub(r"# \S+", "", sent) for sent in data]

data = [re.sub(r"#\S+", "", sent) for sent in data]

data = [re.sub(r'\w*twitter.co\w*', '', sent) for sent in data]

data = [re.sub(r"@\S+", "", sent) for sent in data]

data = [re.sub(r'\w*twitter.com\w*', '', sent) for sent in data]

data = [re.sub(r"./\S+", "", sent) for sent in data]

data = [re.sub(r"@ \S+", "", sent) for sent in data]

for sent in data:
    with open(directory+'test_sents.txt', 'a') as f:
        if sent:
            f.write(sent+'\n')
