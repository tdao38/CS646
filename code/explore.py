import os
import pytrec_eval ###https://awesomeopensource.com/project/cvangysel/pytrec_eval
import nltk
import string
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


# Read in the data
qrels = pd.read_csv('trec-covid-information-retrieval/qrels.csv')
topics = pd.read_csv('trec-covid-information-retrieval/topics-rnd3.csv')
metadata = pd.read_csv('trec-covid-information-retrieval/CORD-19/CORD-19/metadata.csv')
embeddings = pd.read_csv('trec-covid-information-retrieval/CORD-19/CORD-19/cord_19_embeddings_2020-05-19.csv')

def preprocess(text):
    print('preprocessing: ', text)
    # Lower case
    text = text.lower()

    # Rewrite some special terms so that they remain unchanged during cleaning later
    text = text.replace('covid-19', 'covid19')
    text = text.replace('SARS-CoV-2', 'sarscov2')

    # Remove punctuation
    for p in string.punctuation:
        if p in text:
            text = text.replace(p, " ")

    # Tokenize
    text_tokenized = nltk.word_tokenize(text)

    # Remove stop words or stand-alone numbers:
    stop_words = nltk.corpus.stopwords.words("english")
    stop_words.extend(['a', 'an', 'the'])
    text_tokenized_final = []
    for w in text_tokenized:
        #print(w, w.isnumeric())
        if w in stop_words or w.isnumeric():
            continue
        else:
            text_tokenized_final.append(w)

    processed_text = ' '.join(text_tokenized_final)

    return(processed_text)

# VECTOR SPACE MODEL COSINE SIMILARITY IMPLEMENTATION:
metadata = metadata.dropna(subset=['abstract']).reset_index()
for i in range(metadata.shape[0]):
    text = metadata.loc[i, 'abstract']
    # Only proceed with non-na cases
    if pd.isna(text):
        continue
    processed_text = preprocess(text)
    # Save to df
    metadata.loc[i, 'processed_abstract'] = processed_text

for i in range(topics.shape[0]):
    text = topics.loc[i, 'narrative']
    # Only proceed with non-na cases
    if pd.isna(text):
        continue
    processed_text = preprocess(text)
    # Save to df
    topics.loc[i, 'processed_query'] = processed_text

# COUNT EMBEDDING
cv = CountVectorizer()
doc_vecs = cv.fit_transform(metadata['processed_abstract'])
query_vecs = cv.transform(topics['processed_query'])

cosine = cosine_similarity(query_vecs, doc_vecs)

#index of largest cosine similarity:
results_all = pd.DataFrame()
for i in range(topics.shape[0]):
    idx = np.flip(np.argsort(cosine[i]))[:1000]
    cosine_scores = pd.Series(cosine[i][idx])
    cosine_docs = metadata.loc[idx]['cord_uid'].reset_index(drop=True)
    results = pd.DataFrame()
    results['cosine_score'] = cosine_scores
    results['cosine_doc'] = cosine_docs
    results['topic-id'] = topics.loc[i, 'topic-id']

    # save to big df
    if results_all.empty:
        results_all = results
    else:
        results_all = pd.concat([results_all, results])

# TF-IDF
tf_idf = TfidfVectorizer()
doc_vecs_tf_idf = tf_idf.fit_transform(metadata['processed_abstract'])
query_vecs_tf_idf = tf_idf.transform(topics['processed_query'])

cosine_tf_idf = cosine_similarity(doc_vecs_tf_idf, query_vecs_tf_idf)
#index of largest cosine similarity:
results_all_tf_idf = pd.DataFrame()
for i in range(topics.shape[0]):
    idx = np.flip(np.argsort(cosine_tf_idf[i]))[:100]
    cosine_scores = pd.Series(cosine_tf_idf[i][idx])
    cosine_docs = metadata.loc[idx]['cord_uid'].reset_index(drop=True)
    results = pd.DataFrame()
    results['cosine_score'] = cosine_scores
    results['cosine_doc'] = cosine_docs
    results['topic-id'] = topics.loc[i, 'topic-id']

    # save to big df
    if results_all.empty:
        results_all_tf_idf = results
    else:
        results_all_tf_idf = pd.concat([results_all_tf_idf, results])

### State of the art VSM
dl = doc_vecs.sum(axis = 1)
df = doc_vecs.sum(axis = 0)
avdl = dl.sum()/dl.shape[0]
C = dl.shape[0]

for i in range(topics.shape[0]):
    print(i)
    topic_ids = []
    doc_ids = []
    scores = []
    for j in range(doc_vecs.shape[0]):
        print(j)
        top = np.log(1 + np.log(1 + doc_vecs[j].toarray()))
        bottom = 1-0.5 + 0.5*(dl[j]/avdl)

        prod = query_vecs[i].toarray() * np.ravel(top/bottom[0]) * np.ravel(np.log((C+1)/df))
        score = np.sum(prod)

        if score > 0:
            topic_ids.append(topics.loc[i, 'topic-id'])
            doc_ids.append(metadata.loc[j, 'cord_uid'])
            scores.append(score)


results_vsm_all = pd.DataFrame()
# BM25