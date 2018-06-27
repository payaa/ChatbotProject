#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:17:31 2018

@author: phoebeliu
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 18:24:20 2018

@author: phoebeliu
"""
import pandas as pd
from collections import defaultdict
import numpy as np

dataset_type = 'collected_data'
FILENAME_JOB2 = "Job2_postprocessed.csv"
FILENAME_JOB3_AGENT_DB= "Job3_ agent_template_data.csv"


df_job2=  pd.read_csv('./'+dataset_type+'/'+ FILENAME_JOB2)
df_job3_db = pd.read_csv('./'+dataset_type+'/'+ FILENAME_JOB3_AGENT_DB)

df_job3_preprocessed = df_job2[['UTTERANCE', 'SCENARIO']]

def assign_template_utterance(row,df_database,col_name):
    template_utterance = df_database.loc[df_database['SCENARIO'] == row['SCENARIO'],col_name]
    if len(template_utterance) == 0:
        return "NONE"     
    else: 
        return template_utterance.to_string(index=False) 
 
col_name = 'ANNOTATION_USER'
df_job3_preprocessed[col_name] = df_job3_preprocessed.apply (lambda row: assign_template_utterance (row,df_job3_db,col_name),axis=1)   

col_name = 'ANNOTATION_SYSTEM'
df_job3_preprocessed[col_name] = df_job3_preprocessed.apply (lambda row: assign_template_utterance (row,df_job3_db,col_name),axis=1)   

col_name = 'AGENT_TEMPLATE_UTTERANCE'
df_job3_preprocessed[col_name] = df_job3_preprocessed.apply (lambda row: assign_template_utterance (row,df_job3_db,col_name),axis=1)   

df_job3_preprocessed = df_job3_preprocessed.loc[df_job3_preprocessed['AGENT_TEMPLATE_UTTERANCE'] != 'NONE']

df_job3_preprocessed.to_csv('./'+dataset_type+'/Job3_preprocessed.csv',sep=',',encoding='utf-8',index=False)    



