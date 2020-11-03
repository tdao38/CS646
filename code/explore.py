import os
import pytrec_eval ###https://awesomeopensource.com/project/cvangysel/pytrec_eval
import nltk
import string
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
from collections import Counter


# Read in the data
qrels = pd.read_csv(r'C:\Users\micha\OneDrive\fall2020\646\trec-covid-information-retrieval\qrels.csv')
topics = pd.read_csv(r'C:\Users\micha\OneDrive\fall2020\646\trec-covid-information-retrieval\topics-rnd3.csv')
metadata = pd.read_csv(r'C:\Users\micha\OneDrive\fall2020\646\trec-covid-information-retrieval\CORD-19\CORD-19\metadata.csv')
embeddings = pd.read_csv(r'C:\Users\micha\OneDrive\fall2020\646\trec-covid-information-retrieval\CORD-19\CORD-19\cord_19_embeddings_2020-05-19.csv')

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

cosine = cosine_similarity(doc_vecs_tf_idf, query_vecs_tf_idf)
#index of largest cosine similarity:
results_all_tf_idf = pd.DataFrame()
for i in range(topics.shape[0]):
    idx = np.flip(np.argsort(cosine[i]))[:100]
    cosine_scores = pd.Series(cosine[i][idx])
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



#change all 'judgements' from 2 -> 1 for simplicity sake
qrels['judgement'] = qrels['judgement'].replace([2],1)

#function to calculate MAP
def map_results_all(val):
    ap = pd.DataFrame()
    topic = 1
    for x in results_all_tf_idf['topic-id'].unique():
        cols = ['topic-id','iteration','cosine_doc','judgement']
        topic_qrels = qrels.loc[qrels['topic-id'] == x]
        results = results_all_tf_idf[results_all_tf_idf['topic-id'] == x]
        topic_qrels.columns = cols #change topic_qrels column name to make left join easier
        rel_judgement = results.merge(topic_qrels, on=['cosine_doc'], how = 'left')
        rel_judgement_map = rel_judgement.head(val)
        average_precision = rel_judgement_map['judgement'].sum() / val
        average_precision_ser = pd.Series(average_precision) #convert np to series
        average_precision_df = pd.DataFrame(average_precision_ser) #convert series to df to append
        ap = ap.append(average_precision_df)

    map = ap.sum() / len(ap.index)
    return map

map_results_all(30)




