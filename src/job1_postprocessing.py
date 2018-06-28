#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 18:24:20 2018

@author: phoebeliu
"""
import pandas as pd
from collections import defaultdict
from gensim import corpora,matutils,models  
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from sklearn.cluster import KMeans,DBSCAN, AgglomerativeClustering,Birch
from sklearn import mixture
from operator import itemgetter
import jellyfish
from sklearn.preprocessing import scale,normalize,LabelEncoder
from sklearn import metrics
import itertools
#from langdetect import detect 


dataset_type = 'collected_data'
FILENAME = "Real_estate_utterances.csv"



def flag_bad_content (row):
   #if row['UTTERANCE'] in scenarios:
   #    print ("utterance %s" % row['UTTERANCE'])
   if len(row['UTTERANCE'].split()) <= 1 :
      return 'UNACCEPATABLE_CONTENT_ONE_WORD'
   elif row['UTTERANCE'] == row['SCENARIO']: 
      return 'UNACCEPATBLE_CONTENT_SAME_AS_SCENARIO'
   elif row['UTTERANCE'] in scenarios :
      return 'UNACCEPTABLE_CONTENT_OTHER_SCENARIO'
   elif any(row['UTTERANCE'] in s for s in scenarios):
       return 'UNACCEPTABLE_CONTENT_SUBSET_OF_OTHER_SCENARIO'
   else:
      return 'ACCEPTABLE_CONTENT'
  

def detect_language (row):
    language = detect(row['UTTERANCE'])
    #print (language)
    return language
 

#def determine_bad_clusters(intracluster_distances,threshold_value):
#    #threshold_value = 0.1    
#    
#    bad_clusters = []
#    for key, value in intracluster_distances.iteritems():  
#        #throw away bottom 20% clusters based on silhoutes. 
#        if value > 0 and value <  threshold_value:   
#           # if value > 0 and value < 0.05:   
#           bad_clusters.extend([key])
#    return bad_clusters

real_estate_user_df=  pd.read_csv('./'+dataset_type+'/'+ FILENAME)

utterance_dict = defaultdict(list)
for index, row in real_estate_user_df.iterrows():
    scenario = row['scenario'].lower().strip()
    response1 = re.sub('[\W_]+', ' ', row['response_1'].lower().strip(), flags=re.UNICODE) 
    response2 = re.sub('[\W_]+', ' ', row['response_2'].lower().strip(), flags=re.UNICODE) 
    response3 = re.sub('[\W_]+', ' ', row['response_3'].lower().strip(), flags=re.UNICODE) 
    utterance_dict[scenario].append(response1)
    utterance_dict[scenario].append(response2)
    utterance_dict[scenario].append(response3)
    
    
scenarios = utterance_dict.keys()
utterances = utterance_dict.values()    
utterances_df = pd.DataFrame({'UTTERANCE':[y for x in utterances for y in x]})
scenarios_flattened = list(itertools.chain.from_iterable(itertools.repeat(scenario, len(utterance_dict[scenario])) for scenario in scenarios))
utterances_df['SCENARIO'] = scenarios_flattened
   

    
#utterances_df['LANGUAGE'] = utterances_df.apply(lambda row: detect_language(row),axis=1)
utterances_df['FLAB_BAD_CONTENT'] = utterances_df.apply (lambda row: flag_bad_content (row),axis=1)
utterances_df = utterances_df.loc[utterances_df['FLAB_BAD_CONTENT'] == 'ACCEPTABLE_CONTENT']

#utterances_df = utterances_df.sample(frac=1)
utterances_df.to_csv('./'+dataset_type+'/Job1_postprocessed.csv',sep=',',encoding='utf-8',index=False)    

