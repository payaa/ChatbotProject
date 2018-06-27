#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:24:07 2018

@author: phoebeliu
"""

FILE_FOLDER = 'collected_data'
FILENAME = "Real_estate_utterances.csv"
OUTPUT_FILENAME = "%s_clustered.csv" % FILENAME.split('.')[0]
DOC2VEC_DIMENSION = 1000
#KEYWORD2VEC_DIMENSION = 200
N_CLUSTERS = 30
IS_SCENARIO_DEPENDENT = True
SCENARIO_INDEX = 6


FILENAME_CLUSTERED = "Real_estate_utterances_clustered.csv"
