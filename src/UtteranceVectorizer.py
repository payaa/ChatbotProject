#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:14:52 2018

@author: phoebeliu
"""


import re
from nltk.stem.porter import PorterStemmer
from gensim import corpora,matutils,models  
import numpy as np

class UtteranceVectorizer(object):
   
    def __init__(self,df):
        self.utterance_df = df
        print ("vectorizing utterances")
        
    def tokenize(self,sent,delim= ' '):
        '''Return the tokens of a sentence including punctuation.
    
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
        '''   
        return [x.lower().strip() for x in re.split(delim, sent) if x.strip()]
     
    def vectorize_sentences(self,dimension):
        sentences, self.dictionary = self.get_dictionary(self.utterance_df['UTTERANCE'])
        self.tfidf_model,self.tfidf_vector, self.lsi_model, self.lsi_vector = self.get_lsi_model(sentences, dimension)
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
        