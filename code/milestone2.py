import os
import pytrec_eval ###https://awesomeopensource.com/project/cvangysel/pytrec_eval
import nltk
import string
import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict


# Read in the data
qrels = pd.read_csv('trec-covid-information-retrieval/qrels.csv')
topics = pd.read_csv('trec-covid-information-retrieval/topics-rnd3.csv')
# metadata = pd.read_csv('trec-covid-information-retrieval/CORD-19/CORD-19/metadata.csv')
metadata = pd.read_csv('trec-covid-information-retrieval/CORD-19/CORD-19/metadata_preprocessed.csv')
embeddings = pd.read_csv('trec-covid-information-retrieval/CORD-19/CORD-19/cord_19_embeddings_2020-05-19.csv')
topics_expanded = pd.read_csv('topics_expanded.csv')

metadata = metadata.dropna(subset=['processed_abstract']).reset_index()

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

### VECTOR SPACE MODEL COSINE SIMILARITY IMPLEMENTATION:
# start_preprocess = datetime.now()
# metadata = metadata.dropna(subset=['abstract']).reset_index()
# for i in range(metadata.shape[0]):
#     text = metadata.loc[i, 'abstract']
#     # Only proceed with non-na cases
#     if pd.isna(text):
#         continue
#     processed_text = preprocess(text)
#     # Save to df
#     metadata.loc[i, 'processed_abstract'] = processed_text
#
# end_preprocess = datetime.now()
#
# metadata.to_csv('trec-covid-information-retrieval/CORD-19/CORD-19/metadata_preprocessed.csv')

# for i in range(topics.shape[0]):
#     text = topics.loc[i, 'narrative']
#     # Only proceed with non-na cases
#     if pd.isna(text):
#         continue
#     processed_text = preprocess(text)
#     # Save to df
#     topics.loc[i, 'processed_query'] = processed_text

for i in range(topics.shape[0]):
    text = topics_expanded.loc[i, 'query_expansion_wn_mesh']
    # Only proceed with non-na cases
    if pd.isna(text):
        continue
    processed_text = preprocess(text)
    # Save to df
    topics_expanded.loc[i, 'processed_query'] = processed_text

# COUNT EMBEDDING
cv_start = datetime.now()
cv = CountVectorizer()
doc_vecs = cv.fit_transform(metadata['processed_abstract'].values.astype('U'))
# query_vecs = cv.transform(topics['processed_query'])
query_vecs = cv.transform(topics_expanded['processed_query'])
cv_end = datetime.now()

cosine = cosine_similarity(query_vecs, doc_vecs)

cv_cosine_start = datetime.now()
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
cv_cosine_end = datetime.now()

# TF-IDF
tf_idf = TfidfVectorizer()
doc_vecs_tf_idf = tf_idf.fit_transform(metadata['processed_abstract'].values.astype('U'))
query_vecs_tf_idf = tf_idf.transform(topics['processed_query'])

cosine_tf_idf = cosine_similarity(query_vecs_tf_idf, doc_vecs_tf_idf)
#index of largest cosine similarity:
results_all_tf_idf = pd.DataFrame()
for i in range(topics.shape[0]):
    idx = np.flip(np.argsort(cosine_tf_idf[i]))[:1000]
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
# dl = doc_vecs.sum(axis = 1)
# df = doc_vecs.sum(axis = 0)
# avdl = dl.sum()/dl.shape[0]
# C = dl.shape[0]
#
# for i in range(topics.shape[0]):
#     print(i)
#     topic_ids = []
#     doc_ids = []
#     scores = []
#     for j in range(doc_vecs.shape[0]):
#         print(j)
#         top = np.log(1 + np.log(1 + doc_vecs[j].toarray()))
#         bottom = 1-0.5 + 0.5*(dl[j]/avdl)
#
#         prod = query_vecs[i].toarray() * np.ravel(top/bottom[0]) * np.ravel(np.log((C+1)/df))
#         score = np.sum(prod)
#
#         if score > 0:
#             topic_ids.append(topics.loc[i, 'topic-id'])
#             doc_ids.append(metadata.loc[j, 'cord_uid'])
#             scores.append(score)
#
#
# results_vsm_all = pd.DataFrame()

# BM25
k_1 = 1.2
b = 0.75

dl = doc_vecs.sum(axis = 1)
df = doc_vecs.sum(axis = 0)
df = np.ravel(df)
avdl = dl.sum()/dl.shape[0]
N = doc_vecs.shape[0] # total number of documents in the collection

# Calculate find common element:
results_bm25 = pd.DataFrame()

begin_bm25 = datetime.now()
for j in range(topics.shape[0]):
    results_q = {}
    for i in range(doc_vecs.shape[0]):
        print('topic: ', j, ' doc: ', i)
        q_index = np.nonzero(query_vecs[0])[1]
        d_index = np.nonzero(doc_vecs[i])[1]
        q_d_index = np.intersect1d(q_index, d_index)

        # Only proceed if there are intersection between query and doc
        if len(q_d_index) != 0:
            tf = doc_vecs[i, q_d_index].toarray()
            d_length = np.ravel(dl[i])[0]
            d_freq = df[q_d_index]

            top_left = (k_1 + 1) * tf
            bottom_left = (1 - b + b*(d_length/avdl)) + tf
            top_right = N - df[q_d_index] + 0.5
            bottom_right = df[q_d_index] + 0.5

            result = top_left/bottom_left + np.log(top_right/bottom_right)
            doc_id = metadata.loc[i, 'cord_uid']
            results_q[doc_id] = result.sum()
            print(doc_id, result.sum())

    top = sorted(results_q.items(), reverse=True)[:100]
    results_q_df = pd.DataFrame(top, columns=['cord-id', 'bm25_score'])
    results_q_df['topic-id'] = topics.loc[j, 'topic-id']

    if results_bm25.empty:
        results_bm25 = results_q_df
    else:
        results_bm25 = pd.concat([results_bm25, results_q_df])

end_bm25 = datetime.now()

results_bm25.columns = ['cosine_doc', 'score', 'topic-id']
results_bm25 = results_bm25.sort_values(by=['topic-id', 'score'], ascending=False)
#change all 'judgements' from 2 -> 1 for simplicity sake
qrels['judgement'] = qrels['judgement'].replace([2],1)

#function to calculate MAP
def map_results_all(val, df):
    ap = pd.DataFrame()
    topic = 1
    for x in topics['topic-id'].unique():
        cols = ['topic-id','iteration','cosine_doc','judgement']
        topic_qrels = qrels.loc[qrels['topic-id'] == x]
        results = df[df['topic-id'] == x]
        topic_qrels.columns = cols #change topic_qrels column name to make left join easier
        rel_judgement = results.merge(topic_qrels, on=['cosine_doc'], how = 'left')
        rel_judgement_map = rel_judgement.head(val)
        average_precision = rel_judgement_map['judgement'].sum() / val
        average_precision_ser = pd.Series(average_precision) #convert np to series
        average_precision_df = pd.DataFrame(average_precision_ser) #convert series to df to append
        ap = ap.append(average_precision_df)

    map = ap.sum() / len(ap.index)
    return map

start = datetime.now()

map_results_all(10,results_all)
map_results_all(20,results_all)
map_results_all(30,results_all)
map_results_all(10,results_all_tf_idf)
map_results_all(20,results_all_tf_idf)
map_results_all(30,results_all_tf_idf)

map_results_all(10, results_bm25)
map_results_all(20, results_bm25)
map_results_all(30, results_bm25)
map_results_all(100, results_bm25)

end = datetime.now()
print(end - start)

#############
# import Bio
# from Bio import Entrez, SeqIO
# from Bio import Medline
# from nltk.corpus import wordnet as wn
#
# Entrez.email = "tdao@umass.edu"
# handle = Entrez.esearch(db="pubmed", term="covid")
# record = Entrez.read(handle)
#
# record["DbList"]['pubmed']
#
# Entrez.esearch(db="pubmed", term="covid")
#
# handle = Entrez.efetch(db="pubmed", id='11472636', retmode="xml")
# print(handle.read())
#
# record = Entrez.read(Entrez.elink(dbfrom="pubmed", id='11472636'))
# for link in record[0]["LinkSetDb"][0]["Link"]:
#     print(link["Id"])
#
# synset = wn.synsets("dog")
#
# ['coronavirus', 'covid19', 'sars20']
#
# handle = Entrez.efetch(db="pubmed", id='14499001', rettype="medline", retmode="text")
# records = Medline.parse(handle)
# for record in records:
#     meshs = record.get("MH")
#     print(meshs)
#
#
# synonyms = []
# for syn in wn.synsets("origin"):
#     for l in syn.lemmas():
#         synonyms.append(l.name())
#
# list(set(synonyms))
