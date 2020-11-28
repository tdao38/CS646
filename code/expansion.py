import os
import pytrec_eval ###https://awesomeopensource.com/project/cvangysel/pytrec_eval
import nltk
from nltk.corpus import wordnet as wn
import string
import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict

topics = pd.read_csv('trec-covid-information-retrieval/topics-rnd3.csv')

# Wordnet expansion
topics['query_expansion_wn'] = np.nan
for i in range(topics.shape[0]):
    query = topics['query'][i]
    tokens = nltk.word_tokenize(query)
    query_expanded = []
    for w in tokens:
        syns = wn.synsets(w)
        if len(syns) == 0:
            query_expanded.append(w)
        else:
            for syn in syns:
                for l in syn.lemmas():
                    query_expanded.append(l.name().replace("_", " "))

    query_expanded_final = list(set(query_expanded)) # dedup
    query_expanded_final_dedup = list(set(nltk.word_tokenize(' '.join(query_expanded_final))))
    topics['query_expansion_wn'][i] = ' '.join(query_expanded_final_dedup)

# MeSH expansion
mesh_list = [
    ['covid19', 'sarscov2', 'ncov', 'ncov19', 'coronavirus', 'corona', 'sars', 'covid', 'sarscov'],
    ['die', 'death', 'mortality', 'survival'],
    ['diagnosis', 'finding', 'screening', 'symptom', 'sign'],
    ['serological', 'serodiagnosis', 'serologic'],
    ['quarantine', 'isolation'],
    ['sanitizer', 'antiseptics', 'disinfectants'],
    ['inhibitors', 'antagonists'],
    ['heart', 'cardiac'],
    ['biomarkers', 'biomarker', 'biologic marker', 'biologic markers'],
    ['hydroxychloroquine', 'oxychlorochin', 'plaquenil'],
    ['spike', 'glycoprotein', 'gp', 's1', 's2'],
    ['phylogeny', 'phylogenetic'],
    ['cytokine', 'cytokinesis'],
    ['mutation', 'mutate', 'mutant']
]

topics['query_expansion_wn_mesh'] = np.nan
for i in range(topics.shape[0]):
    query = topics['query_expansion_wn'][i]
    tokens = nltk.word_tokenize(query)
    query_expanded = []
    for w in tokens:
        query_expanded.append(w)
        for j in range(len(mesh_list)):
            if w in mesh_list[j]:
                query_expanded += mesh_list[j]

    query_expanded_final = list(set(query_expanded))
    query_expanded_final_dedup = list(set(nltk.word_tokenize(' '.join(query_expanded_final))))
    topics['query_expansion_wn_mesh'][i] = ' '.join(query_expanded_final_dedup)

topics.to_csv('topics_expanded.csv', index=False)