'''
Created on Jan 22, 2016

@author: erica
'''

#from sklearn.externals import joblib
import pandas as pd
try:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter 
    import Tkinter as tk
except ImportError:
    # for Python3
    from tkinter import *   ## notice lowercase 't' in tkinter here
    import tkinter as tk
#from Tkinter import *
#import tkFont

import numpy as np
import pickle
from settings import *


FILE_FOLDER = 'collected_data'
FILENAME = "Real_estate_utterances_clustered.csv"

#dataset_type = 'chatbot_data'

cluster_definition = pd.read_csv('./'+FILE_FOLDER+'/'+FILENAME,sep=',',encoding='utf-8')
cluster_definition['CLUSTER_ID'] = cluster_definition['CLUSTER_ID'].astype('str')
#intracluster_distances = pickle.load( open( "./results/intercluster_distances.pkl", "rb" ) )
#TSSs = pickle.load( open( "./results/tss.pkl", "rb" ) )

intracluster_distances = cluster_definition['INTRACLUSTER_DISTANCE'][(pd.notnull(cluster_definition['INTRACLUSTER_DISTANCE'])) ] 

#TSSs = cluster_definition['TSS'][(pd.notnull(cluster_definition['TSS'])) ] 
cutoff_value = 0



def load_clusters(listbox):
    unique_clusters = pd.unique(cluster_definition.CLUSTER_ID.ravel())

    print ("unique clusters ",len(unique_clusters))
    global cutoff_value
    
    for u in unique_clusters:
        bad= ''
        if 'BAD' in set(cluster_definition[cluster_definition['CLUSTER_ID'] == u]['IS_BAD_CLUSTER']):
            bad = "bad"
        number_per_cluster = num_speeches_per_cluster(u)
        
        if  "silhoutte" in metric.get():     
           # metric_scores = silhoutte_in_cluster(u)
           # average_metric_score = np.average(metric_scores)    
#            cutoff_value = 0
#            if (average_metric_score < 0):           
#                listbox.insert(END,str(u) + " *"+str(number_per_cluster)+"* "+bad)  
#            else:
            listbox.insert(END,str(u) + " ["+str(number_per_cluster)+"] "+bad)
        elif "intracluster" in metric.get():    
            bottom20 = np.percentile(intracluster_distances.tolist(),int(percentileEntry.get()))          
            cutoff_value = bottom20
            if (intracluster_distances.get(int(u)) > bottom20):           
                listbox.insert(END,str(u) + " *"+str(number_per_cluster)+"* "+bad)  
            else:
                listbox.insert(END,str(u) + " ["+str(number_per_cluster)+"] "+bad)
#        elif "tss" in metric.get():         
#            bottom20 = np.percentile(TSSs.values(), int(percentileEntry.get())) 
#            cutoff_value = bottom20
#            if (TSSs.get(int(u)) > bottom20):           
#                listbox.insert(END,str(u) + " *"+str(number_per_cluster)+"* "+bad)  
#            else:
#                listbox.insert(END,str(u) + " ["+str(number_per_cluster)+"] "+bad) 
  
def onIDSelect(evt):
    w = evt.widget
    index = int(w.curselection()[0])
    value = w.get(index)       
    utteranceListBox.delete(0, END)   
 
    cluster_id = value.split()
    utterances = speeches_in_cluster(cluster_id[0])
    scenarios = scenarios_in_cluster(cluster_id[0])
    print (scenarios)
    #shopkeeper_type = shopkeeper_type_in_cluster(cluster_id[0])
    #silhoutte_scores = silhoutte_in_cluster(cluster_id[0])
    #max_silhouette= np.amax(silhoutte_scores,axis=0)    
    #max_index =  silhoutte_scores.index(max(silhoutte_scores))
    
#    if "intracluster" in metric.get():    
#        avg_score = intracluster_distances.get(int(cluster_id[0]))   
    
#    if  "silhoutte" in metric.get():           
#        #avg_score = np.average(silhoutte_in_cluster(cluster_id[0]))
#       # utteranceListBox.insert(END, "TYPICAL: " + utterances[max_index] + " ["+spatial_states[max_index]+"]")        
#    elif "intracluster" in metric.get():    
#        avg_score = intracluster_distances.get(int(cluster_id[0]))   
#    elif "tss" in metric.get():    
#        avg_score = TSSs.get(int(cluster_id[0]))   
    
    utteranceListBox.insert(END,"TYPICAL: "+typical_speech_in_cluster(cluster_id[0]))
    #utteranceListBox.insert(END, "cutoff value " + str(cutoff_value))
    utteranceListBox.insert(END, ""  )
#    utteranceListBox.insert(END, str(avg_score))
    utteranceListBox.insert(END, ""  )

    for index,u in enumerate(utterances):
        if  "silhoutte" in metric.get():   
            utteranceListBox.insert(END, "scenario: "+scenarios[index])
            utteranceListBox.insert(END, "\t"+u  ) 
            utteranceListBox.insert(END, "")
#            silhoutte_score = '%.3f' % silhoutte_scores[index]       
#            utteranceListBox.insert(END, "[" + str(silhoutte_score)+ "] "+u + " ["+spatial_states[index]+"]") 
        else:              
            utteranceListBox.insert(END, "scenario: "+scenarios[index])
            utteranceListBox.insert(END, "\t"+u  ) 
            utteranceListBox.insert(END, "")
   


def speeches_in_cluster(w):
    w = str(w)
    idx = cluster_definition[cluster_definition['CLUSTER_ID'] == w].index.tolist()
    row = list(cluster_definition.loc[idx].to_dict()['SPEECH'].values())
    return (row)    

def scenarios_in_cluster(w):
    w = str(w)
    idx = cluster_definition[cluster_definition['CLUSTER_ID'] == w].index.tolist()
    row = list(cluster_definition.loc[idx].to_dict()['SCENARIO'].values())
    return (row)    


    
def typical_speech_in_cluster(w):
    #w = w.split('_')[0]
    typical_speeches = cluster_definition.query('TYPICAL_VECTOR == "*"')
    idx = typical_speeches[typical_speeches['CLUSTER_ID'] == w].index.tolist()
    row = list(typical_speeches.loc[idx].to_dict()['SPEECH'].values())
    if len(row) > 0:        
        row = row[0]
    else: 
        row = "NONE"
    #idx = typical_speeches[typical_speeches['CLUSTER'] == w].index.tolist()
    #row = typical_speeches.loc[idx].to_dict()['RAW_SHOPKEEPER_SPEECH'].values()
    return row   
    
#def silhoutte_in_cluster(w):
#    w = str(w)
#    idx = cluster_definition[cluster_definition['CLUSTER_ID'] == w].index.tolist()
#    row = cluster_definition.loc[idx].to_dict()['SILHOUETTE'].values()
#    return (row)    
#    
#def average_silhoutte_value(w):
#    w = str(w)
#    idx = cluster_definition[cluster_definition['CLUSTER_ID'] == w].index.tolist()
#    row = cluster_definition.loc[idx].to_dict()['SILHOUETTE'].values()    
#    return (row)   
    


def num_speeches_per_cluster(w):
    idx = cluster_definition[cluster_definition['CLUSTER_ID'] == w].index.tolist() 
    return (len(idx)) 
    

top = tk.Tk()
layer0 = Frame(top)
layer0.pack(side=TOP)
layer1 = Frame(top)
layer1.pack(side=LEFT)
layer2 = Frame(top)
layer2.pack(side=RIGHT)


def load_metric():
    outputIds.delete(0, END)   
    utteranceListBox.delete(0, END)       
    load_clusters(outputIds)
   
    #elif "o" in metric.get():         
        
     

metric= StringVar()
metric.set("silhoutte")
Radiobutton(layer0, text="Silhoutte", variable=metric, value="silhoutte",command=load_metric).pack(side=LEFT,anchor=W)
Radiobutton(layer0, text="Intracluster_distance", variable=metric, value="intracluster",command=load_metric).pack(side=LEFT,anchor=W)
Radiobutton(layer0, text="TSS", variable=metric, value="tss",command=load_metric).pack(side=LEFT,anchor=W)
percentileLabel = Label(layer0, text="   Percentile")
percentileLabel.pack(side=LEFT)
percentileEntry = Entry(layer0, bd=1, width=10)
percentileEntry.pack(side=LEFT)
percentileEntry.insert(END, 80)
# ID scrollbarprint ("cluster id is "+cluster_id[0]+" o")
outputIds = Listbox(layer1, width=25,height=60)
idScroll = Scrollbar(layer1, command=outputIds.yview)
idScroll.pack(side=RIGHT, fill=BOTH)

outputIds.configure(yscrollcommand=idScroll.set)
outputIds.pack(side=RIGHT, fill=BOTH)

outputIds.bind('<<ListboxSelect>>', onIDSelect)
load_clusters(outputIds)
 
 # Utterance Clusters TODO: dynamic output and clickable Utterances
#spatialListBox = Listbox(layer2, width=10,height=60) 
#spatialScroll = Scrollbar(layer2, command=spatialListBox.yview)
#spatialScroll.pack(side=RIGHT, fill=BOTH) 
#spatialListBox.configure(yscrollcommand=spatialScroll.set)
#spatialListBox.pack(side=RIGHT, fill=BOTH)
# utterance scrollbar
 
# Utterance Clusters TODO: dynamic output and clickable Utterances
utteranceListBox = Listbox(top, width=120,height=60) 
utteranceScroll = Scrollbar(top, command=utteranceListBox.yview)
utteranceScroll.pack(side=RIGHT, fill=Y)
 
utteranceXScroll = Scrollbar(top, orient=HORIZONTAL,command=utteranceListBox.xview)
utteranceXScroll.pack(side = BOTTOM, fill =X)
 
utteranceListBox.configure(yscrollcommand=utteranceScroll.set)
utteranceListBox.configure(xscrollcommand=utteranceXScroll.set)
# TODO: fill utterances dynamically
# utteranceListBox.insert(1, "Id 1 Utterance 1")
utteranceListBox.pack(side=RIGHT, fill=BOTH,expand=True)

top.mainloop()
