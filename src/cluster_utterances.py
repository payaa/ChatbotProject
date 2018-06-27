#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 18:24:20 2018

@author: phoebeliu
"""
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans,DBSCAN, AgglomerativeClustering,Birch
from sklearn import mixture
#from operator import itemgetter
import jellyfish
from sklearn.preprocessing import scale,normalize,LabelEncoder
from sklearn import metrics
import itertools
from sklearn.decomposition import PCA
from UtteranceVectorizer import *
import config 



def generate_utterance_dict():
    df=  pd.read_csv('./'+config.FILE_FOLDER+'/'+ config.FILENAME)
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
  
        
def save_result(utterances,scenarios,cluster,typical_utterances,bad_clusters,intracluster_distances):
   
    print ('saving results') 

    result = pd.DataFrame({'SPEECH':utterances})
    result['CLUSTER_ID']  =  cluster
    
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
    result.to_csv('./'+config.FILE_FOLDER+'/'+config.OUTPUT_FILENAME,sep=',',encoding='utf-8')    
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


 


def main():
    utterance_dict = generate_utterance_dict()    
    utterances_df = create_utterance_df(utterance_dict)
    
    vectorizer = UtteranceVectorizer(utterances_df)
    vectorizer.vectorize_sentences(dimension=config.DOC2VEC_DIMENSION) 
    
    if config.IS_SCENARIO_DEPENDENT:           
        user_lsi_vector,user_tfidf_vector,utterances,current_scenario = vectorizer.get_vector_for_clustering(scenario =list(utterance_dict.keys())[config.SCENARIO_INDEX])
    else: 
        user_lsi_vector = vectorizer.lsi_vector
        user_tfidf_vector = vectorizer.tfidf_vector
        utterances = utterances_df['UTTERANCE']
        current_scenario = utterances_df['SCENARIO']
    
    
    model, cluster_series = clustering(user_lsi_vector,   config.N_CLUSTERS)
    intracluster_distances,typical_utterances = calc_cluster_metrics(utterances,cluster_series,user_tfidf_vector)
    
    #silhoutte_scores = metrics.silhouette_score(user_lsi_vector, cluster_series,
    #                                      metric='euclidean')
    
    # calcuate the silhoutte score
    #silhouette_scores = metrics.silhouette_samples(final_vecs, cluster_array, metric='cosine')
    bad_clusters = determine_bad_clusters(intracluster_distances,threshold_value = 0.1)   
    save_result(utterances,current_scenario,cluster_series,typical_utterances,bad_clusters,intracluster_distances)


if __name__ == '__main__':
    main()
