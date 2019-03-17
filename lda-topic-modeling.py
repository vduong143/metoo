from __future__ import unicode_literals
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# import sys
# reload(sys)
# sys.setdefaulencoding('utf8')

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use','metoo','not','do',"arent", "couldnt", "dont", "didnt", "doesnt", "hadnt", "hasnt", "havent", "isnt", "mightnt","mustnt", "neednt", "shant"])
stop_words.extend(["shouldnt", "wasnt", "werent", "wont", "wouldnt",'cannot','cant'])

directory='/home/vietduong/Desktop/metoo_project/'
df = pd.read_csv(directory+'metoo_tweets.csv')

data = df.tweet.values.tolist()

data = [str(sent) for sent in data]

data = [re.sub("@ ", "@", sent) for sent in data]
data = [re.sub("# ", "#", sent) for sent in data]

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# data = [re.sub('\S*#\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

data = [re.sub(r"#metoo+", "metoo", sent) for sent in data]
data = [re.sub(r"# metoo+", "metoo", sent) for sent in data]
data = [re.sub(r"metoo\S+", "metoo", sent) for sent in data]
data = [re.sub(r"# \S+", "", sent) for sent in data]
data = [re.sub(r"#\S+", "", sent) for sent in data]

data = [re.sub(r"./\S+", "", sent) for sent in data]
data = [re.sub(r"@ \S+", "", sent) for sent in data]
data = [re.sub(r"@\S+", "", sent) for sent in data]

# Remove tweets with links and emails
data = [sent for sent in data if r'http\S+' not in sent]
data = [sent for sent in data if r'\w*twitter.co\w*' not in sent]
data = [sent for sent in data if r'\w*twitter.com\w*' not in sent]
data = [re.sub(r"http\S+", "", sent) for sent in data]
data = [re.sub("http", "", sent) for sent in data]
data = [re.sub(r'\w*twitter.co\w*', '', sent) for sent in data]
data = [re.sub(r"@\S+", "", sent) for sent in data]
data = [re.sub(r'\w*twitter.com\w*', '', sent) for sent in data]
data = [sent.strip() for sent in data]

# Clean empty 
def remove_empty(list):
        cleaned = []
        for d in list:
                if (len(d)>0):
                        cleaned.append(d)
        return cleaned

data = remove_empty(data)

open(directory+'cleaned_metoo_tweets.txt','w').close()
for sent in data:
    with open(directory+'cleaned_metoo_tweets.txt', 'a') as f:
        f.write(sent+'\n')

data2 = data
data = [re.sub("metoo", "", sent) for sent in data]

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence).lower(), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words[:1])

# Build the bigram and trigram models
# higher threshold fewer phrases.
bigram = gensim.models.Phrases(data_words, min_count=10, threshold=100) 
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
#print(data_words_nostops[:1])

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)

nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
data_lemmatized2 = data_lemmatized
data_lemmatized = remove_empty(data_lemmatized)

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
print(len(id2word))
# Create Corpus
texts = data_lemmatized
print(len(texts))

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

#print(corpus)
# View
print(corpus[:1])

[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

# tf-idf transformation
from gensim.models import TfidfModel
tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf,
                                           id2word=id2word,
                                           num_topics=35,
                                           random_state=100,
                                           iterations=100,
                                           update_every=1,
                                           chunksize=2000,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus_tfidf]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
# a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Visualize the topics
vis = pyLDAvis.gensim.prepare(lda_model, corpus_tfidf, id2word)
pyLDAvis.save_html(vis,'LDA.html')

mallet_path = directory+'/mallet-2.0.8/bin/mallet' # update this path

# def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
#     coherence_values = []
#     model_list = []
#     for num_topics in range(start, limit, step):
#         model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                                 id2word=dictionary,
#                                                 num_topics=num_topics,
#                                                 random_state=100,
#                                                 iterations=100,
#                                                 update_every=1,
#                                                 chunksize=2000,
#                                                 passes=10,
#                                                 alpha='auto',
#                                                 per_word_topics=True)
                                                                
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())

#     return model_list, coherence_values

# model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus_tfidf, texts=data_lemmatized, start=5, limit=100, step=5)
# # Show graph
# limit=100
# start=5
# step=5
# x = range(start, limit, step)
# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()

# Build Mallet model
# lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus_tfidf, num_topics=25, id2word=id2word)

# Show Topics
# pprint(lda_mallet.show_topics(formatted=False))

# Visualize the topics
# lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_mallet)
# vis = pyLDAvis.gensim.prepare(lda_model, corpus_tfidf, id2word)
# pyLDAvis.save_html(vis,'LDA.html')

# Compute Coherence Score
# coherence_model_lda_mallet = CoherenceModel(model=lda_mallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
# coherence_lda_mallet = coherence_model_lda_mallet.get_coherence()
# print('\nCoherence Score: ', coherence_lda_mallet)

def format_topics_sentences(ldamodel=lda_model, corpus=corpus_tfidf, texts=data_lemmatized):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = row[0]
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                        wp = ldamodel.show_topic(topic_num)
                        topic_keywords = ", ".join([word for word, prop in wp])
                        sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                        break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus_tfidf, texts=data_lemmatized)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Save
df_dominant_topic
data2 = [sent.replace(',','') for sent in data2]
raw_lemmatized = pd.DataFrame({'Tweet': data2, 'Text': data_lemmatized2})
df_dominant_topic.merge(raw_lemmatized,on='Text')
df_dominant_topic.to_csv(directory+'metoo_topics.csv')

for topic in range(35):
    df = df_dominant_topic.loc[df_dominant_topic['Dominant_Topic']==topic]
    open(directory+'clusters/cluster_{}.txt'.format(topic), 'w').close()
    for tweet in df['Tweet']:
        with open(directory+'clusters/cluster_{}.txt'.format(topic), 'a') as f:
            f.write(str(tweet).strip()+'\n')