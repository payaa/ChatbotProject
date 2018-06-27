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
from sklearn.decomposition import PCA

FILE_FOLDER = 'collected_data'
FILENAME = "Real_estate_utterances.csv"
OUTPUT_FILENAME = "%s_clustered.csv" % FILENAME.split('.')[0]
DOC2VEC_DIMENSION = 1000
#KEYWORD2VEC_DIMENSION = 200
N_CLUSTERS = 30
IS_SCENARIO_DEPENDENT = True
SCENARIO_INDEX = 6


def generate_utterance_dict():
    df=  pd.read_csv('./'+FILE_FOLDER+'/'+ FILENAME)
    utterance_dict = defaultdict(list)
    for index, row in df.iterrows():
        scenario = row['scenario']
        response1 = re.sub('[\W_]+', ' ', row['response_1'].lower(), flags=re.UNICODE) 
        response2 = re.sub('[\W_]+', ' ', row['response_2'].lower(), flags=re.UNICODE) 
        response3 = re.sub('[\W_]+', ' ', row['response_3'].lower(), flags=re.UNICODE) 
        utterance_dict[scenario].append(response1)
        utterance_dict[scenario].append(response2)
        utterance_dict[scenario].append(response3)
    return utterance_dict


def create_utterance_df(utterance_dict):
    scenarios = utterance_dict.keys()
    utterances = utterance_dict.values()    
    
    utterances_df = pd.DataFrame({'UTTERANCE':[y for x in utterances for y in x]})
    scenarios_flattened = list(itertools.chain.from_iterable(itertools.repeat(scenario, len(utterance_dict[scenario])) for scenario in scenarios))
    utterances_df['SCENARIO'] = scenarios_flattened
    utterances_df['FLAB_BAD_CONTENT'] = utterances_df.apply (lambda row: flag_bad_content (row),axis=1)
    return utterances_df
       


def clustering(vector, N_CLUSTERS):

    print ('clustering')
    # different clustering algorithms, trying with gaussian mixture model now, probably need to tune parameter
    model = KMeans(n_clusters=N_CLUSTERS, random_state=0)
    #model = mixture.GaussianMixture(n_components=N_CLUSTERS,covariance_type='spherical',random_state=42,verbose=1)
    #model= Birch(branching_factor=10, n_clusters=N_CLUSTERS, threshold=0.005,compute_labels=True)
    #model = AgglomerativeClustering(n_clusters=N_CLUSTERS,linkage="average", affinity='cosine')
    model.fit(vector)
    #cluster = model.labels_+1
    
    clusters = model.predict(vector) + 1
    cluster_series =  pd.Series(clusters)
    return model,cluster_series


## clustering
def calc_centroid (test_vec):
    # Sum the vectors in each cluster
    #lens = {}      # will contain the lengths for each cluster
    #centroid = {} # will contain the centroids of each cluster
    lens = 0
    centroid = np.zeros(test_vec.shape[1])
    for idx,clno in enumerate(test_vec):
        centroid+= test_vec[idx,:]
        lens += 1
    # Divide by number of observations in each cluster to get the centroid
    centroid  /= float(lens)
    return centroid

def calc_cluster_metrics(utterances, cluster_series, tfidf_vector):
    centroids = {}
    intracluster_distances = {}
    TSSs = {}
    typical_utterances = []
    
    for u in set(cluster_series):     
        print ("cluster id %s" % u)
        
        test_string = utterances[ cluster_series[cluster_series == u].index]   
        total_dist = {}
        for i,value in test_string.items():   
         
            total_dist1 = 0
            for index_j,value_j in test_string.items():     
                #print (value + " "+value_j)
                dist = jellyfish.damerau_levenshtein_distance(value, value_j) 
                total_dist1 += dist           
            total_dist[i]  = total_dist1
        typical_utterance_index = min(total_dist, key=total_dist.get)
        typical_utterances.append(typical_utterance_index)
      
        size = np.sum(cluster_series==u)        
    
        avg_dist = 0
      
        test_vec = tfidf_vector[ cluster_series[cluster_series == u].index]        
        test_vec= normalize(test_vec)    
        dist_matrix = 1-metrics.pairwise_distances(test_vec,metric='cosine')          
        customer_centroid_vectors = tfidf_vector[ cluster_series[cluster_series == u].index] 
        centroids[u] = calc_centroid( customer_centroid_vectors )
    #    utterance_distance={}
    #    for i in range(size):            
    #        utterance_distance[test_string.index[i]] = sum(dist_matrix[i,])   
    #    typical_utterance_index = max(utterance_distance, key=utterance_distance.get)
    #    typical_utterances.append(typical_utterance_index)
        
        per_sample = {}
        #plt.figure(figsize=(5, 4.5))
        if size > 1:
            for i in range(size):
                
                per_sample[i] = sum(dist_matrix[i,])
                
                for j in range(size):
                    avg_dist += dist_matrix[i,j]       
                    
            avg_dist /= (size * size)
            intracluster_distances[u] = avg_dist
            
       
    
    return intracluster_distances, typical_utterances
  
        
def save_result(utterances,scenarios,cluster,typical_utterances,bad_clusters):
   
    print ('saving results') 

    result = pd.DataFrame({'SPEECH':utterances})
    result['CLUSTER_ID']  =  cluster_series
    
    result['SCENARIO'] = scenarios

    result['TYPICAL_VECTOR'] = pd.Series('-')
    result.loc[typical_utterances,'TYPICAL_VECTOR'] = '*'      
    result['INTRACLUSTER_DISTANCE'] = pd.Series('')
    #result['TSS'] = pd.Series('')
      
    result['IS_BAD_CLUSTER'] = pd.Series('-')
    
    result.loc[result.CLUSTER_ID.isin(bad_clusters), 'IS_BAD_CLUSTER'] = "BAD"  
    for index, row in result.iterrows():  
       cluster_id = row['CLUSTER_ID']     
       result.loc[index, 'INTRACLUSTER_DISTANCE'] = intracluster_distances.get(cluster_id)        
       #result.loc[index, 'TSS'] = TSSs.get(cluster_id) 
    result = result.sort_values(['CLUSTER_ID'], ascending=[True])    
    result.to_csv('./'+FILE_FOLDER+'/'+OUTPUT_FILENAME,sep=',',encoding='utf-8')    
    print ('DONE')



def flag_bad_content (row):
   if len(row['UTTERANCE'].split()) <= 1 or row['UTTERANCE'] == row['SCENARIO'] :
      return 'UNACCEPTABLE_CONTENT'
   else:
      return 'ACCEPTABLE_CONTENT'
  

def determine_bad_clusters(intracluster_distances,threshold_value):
    #threshold_value = 0.1    
    
    bad_clusters = []
    for key, value in intracluster_distances.items():  
        #throw away bottom 20% clusters based on silhoutes. 
        if value > 0 and value <  threshold_value:   
           # if value > 0 and value < 0.05:   
           bad_clusters.extend([key])
    return bad_clusters

class UtteranceVectorizer(object):
   
    def __init__(self,utterance_df):
        self.utterance_df = utterances_df
        print ("vectorizing utterances")
        
    def tokenize(self,sent,delim= ' '):
        '''Return the tokens of a sentence including punctuation.
    
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
        '''   
        return [x.lower().strip() for x in re.split(delim, sent) if x.strip()]
     
    def vectorize_sentences(self):
        sentences, self.dictionary = self.get_dictionary(self.utterance_df['UTTERANCE'])
        self.tfidf_model,self.tfidf_vector, self.lsi_model, self.lsi_vector = self.get_lsi_model(sentences, DOC2VEC_DIMENSION)
        #return dictionary,tfidf_model,tfidf_vector,lsi_model, lsi_vector
    
    def ngrams(self,tokens, n,len_extra=0 ): 
        output = []  
        for i in range(len_extra,len(tokens)-n+1):
        #for i in range(len(tokens)-n-len_extra+1):
            g = '-'.join(tokens[i:i+n])
            output.extend([g]) 
        return (output)

    def tokenize_and_stem(self,text, is_tokenize=True):
        text = "".join(c for c in text if c not in ('!','.',',','?'))  
        tokens = [word.lower().strip() for word in text.split(" ")]
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z0-9]', token):
                filtered_tokens.append(token)  
        stems = [PorterStemmer().stem(t) for t in filtered_tokens]
        if is_tokenize:
            return stems
        else:
            return self.tokenize(text)
    
    def get_tokens(self,sentence,is_tokenize=True,use_ngram=True):
        tokens = self.tokenize_and_stem(sentence,is_tokenize)      
        if use_ngram: 
            bigram = self.ngrams(tokens,2)
            trigram = self.ngrams(tokens,3)      
            tokens.extend(bigram)
            tokens.extend(trigram)
        return tokens
    
    def get_lsi_model(self,sentences,dimension):
        corpus = [self.dictionary.doc2bow(s) for s in sentences]
        tfidf = models.TfidfModel(corpus,normalize=True)
        corpus_tfidf = tfidf[corpus]
        lsi = models.LsiModel(corpus_tfidf, onepass=False, power_iters=2,id2word=self.dictionary, num_topics=dimension)
        corpus_lsi = lsi[corpus_tfidf]
        corpus_lsi = np.asarray([matutils.sparse2full(vec, dimension) for vec in corpus_lsi])
        corpus_tfidf = np.asarray([matutils.sparse2full(vec, len(self.dictionary)) for vec in corpus_tfidf])  
        #dtm_model = gensim.models.DtmModel('dtm-linux64', corpus, my_timeslices, num_topics=20, id2word=dictionary)
        return (tfidf, corpus_tfidf, lsi, corpus_lsi)
    
    def get_dictionary(self,df,is_tokenize=True,use_ngram=True):
        sentences = []
        for index, sentence in enumerate(df):
            #print (sentence)
            tokens =self.get_tokens(sentence,is_tokenize,use_ngram) 
            sentences.append(tokens)
            
        dictionary = corpora.Dictionary(sentences)
        dictionary.filter_extremes(no_below=2, no_above=0.95, keep_n=5000)
        once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
        dictionary.filter_tokens(once_ids) # remove stop words and words that appear only once
        dictionary.compactify()   
        return (sentences, dictionary)    
            
        
    
    def get_vector_for_clustering(self,scenario=None):
    
        
        indices = self.utterance_df.index[self.utterance_df['SCENARIO'] == scenario].tolist()    
        lsi_vector_scenario_based = np.take(self.lsi_vector, indices,axis = 0)
        tfidf_vector_scenario_based = np.take(self.tfidf_vector, indices,axis = 0)
        
        utterances_flattened_scenario_based = self.utterance_df.loc[self.utterance_df['SCENARIO'] == scenario]['UTTERANCE']    
        utterances_flattened_scenario_based = utterances_flattened_scenario_based.reset_index(drop=True)
        
        current_scenario = self.utterance_df.loc[self.utterance_df['SCENARIO'] == scenario]['SCENARIO']
        current_scenario = current_scenario.reset_index(drop=True)
        
        return lsi_vector_scenario_based,tfidf_vector_scenario_based,utterances_flattened_scenario_based,current_scenario
        
 


def main():
    utterance_dict = generate_utterance_dict()    
    utterances_df = create_utterance_df(utterance_dict)
    
    vectorizer = UtteranceVectorizer(utterances_df)
    vectorizer.vectorize_sentences() 
    #user_dictionary,user_tfidf_model,user_tfidf_vector,user_lsi_model, user_lsi_vector = vectorizer.vectorize_sentences() 
    
    if IS_SCENARIO_DEPENDENT:        
      #  user_lsi_vector,user_tfidf_vector,utterances,current_scenario = get_vector_for_clustering(is_scenario_dependent = False)
        user_lsi_vector,user_tfidf_vector,utterances,current_scenario = vectorizer.get_vector_for_clustering(scenario =list(utterance_dict.keys())[SCENARIO_INDEX])
    else: 
        user_lsi_vector = vectorizer.lsi_vector
        user_tfidf_vector = vectorizer.tfidf_vector
        utterances = utterances_df['UTTERANCE']
        current_scenario = utterances_df['SCENARIO']
    
    
    model, cluster_series = clustering(user_lsi_vector,   N_CLUSTERS)
    intracluster_distances,typical_utterances = calc_cluster_metrics(utterances,cluster_series,user_tfidf_vector)
    
    #silhoutte_scores = metrics.silhouette_score(user_lsi_vector, cluster_series,
    #                                      metric='euclidean')
    
    # calcuate the silhoutte score
    #silhouette_scores = metrics.silhouette_samples(final_vecs, cluster_array, metric='cosine')
    bad_clusters = determine_bad_clusters(intracluster_distances,threshold_value = 0.1)   
    save_result(utterances,current_scenario,cluster_series,typical_utterances,bad_clusters)


if __name__ == '__main__':
    main()
