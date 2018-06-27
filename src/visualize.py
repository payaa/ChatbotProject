# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:39:43 2017

@author: erica
"""
#import plotly.plotly as py
#py.sign_in('payaa','joHTtB54dA6BW5ZxpCPo')
#import plotly.graph_objs as go
#
#from plotly.graph_objs import *

from functions import *
from TrainedNeuralNetwork import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, show

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models.widgets import Panel,  Select,CheckboxGroup,Div, DataTable, DateFormatter, TableColumn
from bokeh.layouts import row, widgetbox


from bokeh.models import HoverTool,CustomJS,FixedTicker
from tsne import bh_sne

from bokeh.layouts import column, gridplot
from bokeh.palettes import RdYlGn6, RdYlGn9
from bokeh.charts import HeatMap, bins, output_file, show

from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN, AgglomerativeClustering,Birch
from sklearn import mixture
import collections
import itertools
import seaborn.apionly as sns
    
import colorsys



def __enum(**enums):
    return type('Enum', (), enums)


color_none = '#FFFFFF'
color_passive = '#FFEB69'
color_proactive ='#70F4FF'
color_annotate =  '#B24083'

class VisualizationDataManager(object):
    
    def __init__(self,dnn):       
        self.dnn = dnn    
        self.shift = np.arange(0, 1.1, 0.1)
        
        self.n_test, self.n_train = vi_dnn.get_n_training_test_examples()
        history_vector =  self.dnn.vectorizer_input.get_combined_history_vector()
        self.history_vector_test, self.history_vector_train = self.dnn.vectorizer_input.get_train_test_sample(history_vector)       
      
        #self.current_mode = current_mode
        self.rerun_training_data()
        self.perform_clustering(400)
        self.tsne_transformation()
        
        self.load_source_train()
        if current_mode is visualize_modes.AFTER_TRAINING:
            self.load_source_test()

    def rerun_training_data(self):
        self.activiation_array = np.empty((0,N_HIDDEN), 'float32')
        dnn.resetTrial()  
        passive_shopkeeper_type_vec = dnn.vectorizer_input.get_new_shopkeeper_type("PASSIVE")
        proactive_shopkeeper_type_vec = dnn.vectorizer_input.get_new_shopkeeper_type("PROACTIVE")
        self.controled_proactivity = []
        self.predicted_answers = []
        self.test_index = []
        
        if current_mode is visualize_modes.PRE_TRAINING:
            self.l_hid3_value =  read_trained_model('l_hid3_value')
            for i in range(self.n_train): 
                predicted_action_dnn_att_history = self.dnn.execute_sequence_predictor(classifers.DNN,self.history_vector_train[[i]]) 
                self.predicted_answers.extend(predicted_action_dnn_att_history.get_csv_result())
        elif current_mode is visualize_modes.AFTER_TRAINING_PROACTIVE: 
            for i in range(self.n_train):                 
                self.predict_by_shopkeeper_type(i,proactive_shopkeeper_type_vec)
            self.l_hid3_value = self.activiation_array
        elif current_mode is visualize_modes.AFTER_TRAINING_PASSIVE: 
            for i in range(self.n_train):                              
                self.predict_by_shopkeeper_type(i,passive_shopkeeper_type_vec)
            self.l_hid3_value = self.activiation_array
        elif current_mode is visualize_modes.AFTER_TRAINING: 
            for i in range(self.n_train): 
                self.add_to_activiation_value(self.history_vector_train[[i]]) 
            ### add to test value for activiation 
            for i in range(self.n_test):  
                self.shift_proactivity(i)
            self.l_hid3_value= self.activiation_array
        elif current_mode is visualize_modes.INPUT_VECTOR:            
            self.l_hid3_value = self.history_vector_train.reshape((self.history_vector_train.shape[0], -1))
            self.predicted_answers.extend([''] * self.n_train)
            

    def perform_clustering(self,n_clusters):
        print ("perform clustering")   
        #model = AgglomerativeClustering(n_clusters,linkage="average", affinity='cosine')
        model = mixture.GaussianMixture(n_components=n_clusters,covariance_type='diag',random_state=42,verbose=1)
        model.fit(self.l_hid3_value)
        cluster = model.predict(self.l_hid3_value) + 1   
        self.neuron_clusters = map(str, cluster)
        #self.neuron_clusters = map(str,model.labels_+1)
    
    def shift_proactivity(self,index):
        all_row = vi_dnn.readable_history_test[index]
        for i in self.shift:
            current_sequence_type = transform_to_vec(all_row,np.array([[i]]))
            self.controled_proactivity.append(str(i))
            self.add_to_activiation_value(current_sequence_type,True)
            self.test_index.append(str(index))
            

        
    def add_to_activiation_value(self,current_sequence,is_add_predicted_answer=True):
        predicted_action_dnn_att_history = self.dnn.execute_sequence_predictor(classifers.DNN,current_sequence)    
        l_value = self.dnn.l_hid3(current_sequence)
        self.activiation_array = np.vstack((self.activiation_array,l_value))   
        if is_add_predicted_answer:
            self.predicted_answers.extend(predicted_action_dnn_att_history.get_csv_result())   
        print ("DNN: "+predicted_action_dnn_att_history.get_csv_result()[0]  )

        
     
    def predict_by_shopkeeper_type(self,current_trial_index,current_shopkeeper_type_vec):      
        print()
        print ("INTERACTION: "+ str(current_trial_index))   
        all_row = vi_dnn.readable_history_train[current_trial_index]
        current_sequence_type = transform_to_vec(all_row,current_shopkeeper_type_vec)
        self.add_to_activiation_value(current_sequence_type)

     
    
    def tsne_transformation(self):   
        print ("tsne")      
        if load_tsne_values:
            self.tsne_hid3_values = read_trained_model('tsne_hid3_values')
        else:
            self.tsne_hid3_values = bh_sne(np.asarray(self.l_hid3_value).astype('float64'),perplexity=100)        
        #tsne_hid3_model = TSNE(n_components=2, perplexity =100,random_state=0) 
        #X_pca = PCA(n_components=50).fit_transform(self.l_hid3_value)
        #self.tsne_hid3_values = tsne_hid3_model.fit_transform(np.asarray(self.l_hid3_value).astype('float64'))     

   
  
    def get_readable_inputs(self,readable_form):
        readable_history_df= pd.DataFrame(columns=['SPEECH1','SPEECH2','SPEECH3'])
        
        for index,value in enumerate(readable_form):    
            if len( value[0]) >0 :
                readable_history_df.set_value(index,'SPEECH1',value[0]['SPATIAL_STATE'] + ' '+value[0]['STATE_TARGET'] + '; '+value[0]['CUSTOMER_SPEECH'])
            if len(value[1]) > 0:
                readable_history_df.set_value(index,'SPEECH2',value[1]['OUTPUT_SPATIAL_STATE'] + ' '+value[1]['OUTPUT_STATE_TARGET'] + '; '+value[1]['SHOPKEEPER_SPEECH'])
            if len(value[2]) > 0 :
                readable_history_df.set_value(index,'SPEECH3',value[2]['SPATIAL_STATE'] + ' '+value[2]['STATE_TARGET'] + '; '+value[2]['CUSTOMER_SPEECH'])

        readable_history_df.fillna('EMPTY',inplace=True)
        return readable_history_df

    def replicate_func(self,group):
        return pd.DataFrame(dict(SPEECH1=np.repeat(group.SPEECH1.values, len(self.shift)), SPEECH2=np.repeat(group.SPEECH2.values, len(self.shift)), SPEECH3=np.repeat(group.SPEECH3.values, len(self.shift))))



 
    def load_source_train(self):  
        self.annotation_tag = ['ASK_QUESTION','SOCIAL_ACKNOWLEDGEMENT','OTHER'] #WAITING
        _, df_train = vi_dnn.get_ground_truth(self.annotation_tag + ['CAMERA_FEATURE','SHOPKEEPER_SPEECH','SHOPKEEPER','SPATIAL_STATE','STATE_TARGET','CLUSTER_ID_SHOPKEEPER'])
        df_train = df_train.reset_index(drop=True)
        #_,annotation_train_df['ground_truth_shopkeeper_speech_train'] = vi_dnn.get_ground_truth('SHOPKEEPER_SPEECH') 
        #_, ground_truth_proactivity_train= vi_dnn.get_ground_truth('SHOPKEEPER')

        _, Y_shopkeeper_train = vi_dnn.get_train_test_sample(vi_dnn.Y_shopkeeper_good)
        predicted_cluster_ids = vi_dnn.encoder_shopkeeper.inverse_transform(Y_shopkeeper_train)  
        df_train['ground_truth_typical_utterance'] =[ vi_dnn.get_typical_speech(str(p),"shopkeeper")    for p in predicted_cluster_ids]
        
        df_train[['SPEECH1','SPEECH2','SPEECH3']]= self.get_readable_inputs( vi_dnn.readable_history_train  )
        
        df_train[['x','y']] = pd.DataFrame(self.tsne_hid3_values[:self.n_train,])
        df_train['fill_colors'] = color_none
        df_train['fill_alpha'] = 1
        df_train['line_color'] = '#565656'
        df_train['line_width'] = 1
        df_train['predicted_answers'] = self.predicted_answers[:self.n_train]
        df_train['neuron_clusters'] = self.neuron_clusters[:self.n_train]
        
        spatial_formations = []
        annotate_spatial_colors = []
        for index,value in  enumerate(df_train['SPATIAL_STATE']):
            if value == 'PRESENT_X':
                spatial_formations.extend([value+'-'+df_train['STATE_TARGET'].iloc[index]])
                if df_train['STATE_TARGET'].iloc[index] == 'NIKON':                   
                    if df_train['SHOPKEEPER'].iloc[index] == 'PROACTIVE':
                        annotate_spatial_colors.extend(['#4d9de0'])
                    elif df_train['SHOPKEEPER'].iloc[index] == 'PASSIVE':
                        annotate_spatial_colors.extend(['#73C3FF'])
                elif df_train['STATE_TARGET'].iloc[index] == 'CANON' :
                    if df_train['SHOPKEEPER'].iloc[index] == 'PROACTIVE':                    
                        annotate_spatial_colors.extend(['#e15554'])
                    elif df_train['SHOPKEEPER'].iloc[index] == 'PASSIVE':
                        annotate_spatial_colors.extend(['#FFBBBA'])
                elif df_train['STATE_TARGET'].iloc[index] == 'SONY' :                  
                    if df_train['SHOPKEEPER'].iloc[index] == 'PROACTIVE':                    
                        annotate_spatial_colors.extend(['#3bb273'])
                    elif df_train['SHOPKEEPER'].iloc[index] == 'PASSIVE':
                        annotate_spatial_colors.extend(['#A1FFD9'])
            else:
                spatial_formations.extend([value])
                if value == 'FACE_TO_FACE':
                    if df_train['SHOPKEEPER'].iloc[index] == 'PROACTIVE':                    
                        annotate_spatial_colors.extend(['#7768ae'])        
                    elif df_train['SHOPKEEPER'].iloc[index] == 'PASSIVE':
                        annotate_spatial_colors.extend(['#DDCEFF'])
                                 
                elif value == 'WAITING':
                    if df_train['SHOPKEEPER'].iloc[index] == 'PROACTIVE':                    
                        annotate_spatial_colors.extend(['#e1bc29'])
                    elif df_train['SHOPKEEPER'].iloc[index] == 'PASSIVE':
                        annotate_spatial_colors.extend(['#FFFF8F'])
                            
                else:
                    annotate_spatial_colors.extend(['#FFFFFF'])
        df_train['SPATIAL_FORMATION'] = spatial_formations
        df_train['annotate_spatial_colors'] = annotate_spatial_colors
        
         
        ### COLOR BASED ON CAMERA AND ANNOTATION
        #state_targets =['CANON','NIKON','SONY']
        #count =  len(self.annotation_tag) * len(state_targets)
        
        ncolors = len(self.annotation_tag) 
        palette = get_N_HexCol(ncolors)
     
        palette = [ '#fa7921','#0b032d', '#7cffc4']
        
        df_train['annotation_colors'] = [unicode('#cecccc')] * len(df_train)
        df_train['annotation_labels'] = ['NONE'] * len(df_train)
        count = 0
        while (count < ncolors):
            
            for index,value in enumerate(self.annotation_tag):   
                #for index2,value2 in enumerate(state_targets):        
                    color_index = df_train[(df_train[value] == 1) ].index.tolist()
                    print (str(count) + ": "+value)
                    for i in color_index:            
                        df_train.set_value(i,'annotation_colors',palette[count])
                        df_train.set_value(i,'annotation_labels',  value)
                    count = count+1
        
        n_proactive,n_passive = vi_dnn.get_n_examples_proactive_passive(df_train)
        df_proactive = df_train[:n_proactive]
        df_passive = df_train[-n_passive:]
        
        self.source = ColumnDataSource(df_train)
        self.source_proactive = ColumnDataSource(df_proactive)
        self.source_passive = ColumnDataSource(df_passive)
        
        self.source_cluster = ColumnDataSource(data=dict(CLUSTER_ID=[""]*self.n_train,CUSTOMER_UTTERANCE = [""]*self.n_train,GROUND_TRUTH = [""]*self.n_train))
        self.source_selection = ColumnDataSource(data=dict(SPATIAL_FORMATION=[""],STATE_TARGET=[""],ANNOTATION_CUSTOMER=[""],ANNOTATION_FEATURE=[""]))
     
            
    def load_source_test(self):      
        df_test = self.get_readable_inputs( vi_dnn.readable_history_test)        
        df_test= df_test.groupby(level=0).apply(self.replicate_func).reset_index(drop=True)
        
        ground_truth_shopkeeper_speech_test,_= vi_dnn.get_ground_truth('SHOPKEEPER_SPEECH')
        ground_truth_shopkeeper_speech_test = ground_truth_shopkeeper_speech_test.tolist() 
        df_test['ground_truth'] = list(itertools.chain.from_iterable(itertools.repeat(x, len(self.shift)) for x in ground_truth_shopkeeper_speech_test))

        df_test[['x','y']] = pd.DataFrame(self.tsne_hid3_values[-len(self.shift)*self.n_test:,])
        df_test['fill_colors'] = linear_gradient("#FFEB69","#70F4FF",len(self.shift))*self.n_test
        df_test['test_index'] = self.test_index
        df_test['controled_proactivity'] = self.controled_proactivity                     
        df_test['line_width'] = 1
        df_test['predicted_answers'] = self.predicted_answers[-len(self.shift)*self.n_test:]
           
        self.source_test = ColumnDataSource(df_test)             



def get_N_HexCol(N):

    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in xrange(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x*255),colorsys.hsv_to_rgb(*rgb))
    
        hex_out.append('#'+"".join(map(lambda x:chr(x).encode('hex'),rgb)))
    return hex_out

    
def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

    
def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return [RGB_to_hex(RGB) for RGB in gradient]



def linear_gradient(start_hex, finish_hex, n):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)
  return color_dict(RGB_list)



class NeuronPanel(object):
    
    def __init__(self):           
        self.initializePanel()
        
    def initializePanel(self):
        output_file("./results/"+current_mode+".html2")     
        TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select" 
        self.p = figure(plot_width=2000, plot_height=2000, tools=[TOOLS],
                   title="Last hidden layer")
           
        tooltips=[("index", "$index"),
                  ("(x,y)", "($x, $y)"),             
                  ("customer1", "@SPEECH1"),
                  ("shopkeeper2", "@SPEECH2"),
                  ("customer3", "@SPEECH3"),
                  ("ground_truth_cluster_id","@CLUSTER_ID"),            
                  ("ground_truth_typical_utterance", "@ground_truth_typical_utterance"), 
                  ("ground_truth", "@SHOPKEEPER_SPEECH"), 
                  ("predicted_answer","@predicted_answers"),
                  ("",""),]      
       
        tooltips_test=[("index", "$index"),
                  ("(x,y)", "($x, $y)"), 
                  ("test_index","@test_index"),
                  ("customer1", "@SPEECH1"),
                  ("shopkeeper2", "@SPEECH2"),
                  ("customer3", "@SPEECH3"),
                  ("controled_proactivity", "(@controled_proactivity)"), 
                  ("ground_truth", "@SHOPKEEPER_SPEECH") ,
                  ("predicted_answer","@predicted_answers"),
                    ]       
      
        self.r = self.p.circle('x', 'y',fill_color ='fill_colors',line_color = None, line_width = 'line_width',size=10, visible=False,source=visualizer.source)   
        self.p.add_tools(HoverTool(tooltips=tooltips , point_policy = "follow_mouse", renderers=[self.r])) 
        
        self.proactive_r = self.p.circle('x', 'y',fill_color ='fill_colors',line_color = 'line_color', line_width = 'line_width',fill_alpha = 'fill_alpha',size=12, source=visualizer.source_proactive)   
        self.passive_r= self.p.triangle('x', 'y',fill_color ='fill_colors', line_color = 'line_color',line_width = 'line_width',fill_alpha = 'fill_alpha',size=12, source=visualizer.source_passive)   
       
           
        if current_mode is visualize_modes.AFTER_TRAINING:
            self.test_r = self.p.diamond('x', 'y',fill_color ='fill_colors',line_color= 'black',line_width = 'line_width',size=15, source=visualizer.source_test)           
            self.p.add_tools(HoverTool(tooltips=tooltips_test , point_policy = "follow_mouse", renderers=[self.test_r])) 
 

    
def initalizeMenu():      
  
    
    current_state_text = Div(text="""None<br>""")  
    #current_source_text = Div(text="""None<br>""",source=visualizer.source_current)  
    columns =  [TableColumn(field="SPATIAL_STATE", title="SPATIAL_STATE"),
                TableColumn(field="STATE_TARGET", title="STATE_TARGET"),
                TableColumn(field="ANNOTATION_CUSTOMER", title="ANNOTATION_CUSTOMER"),
                TableColumn(field="ANNOTATION_FEATURE", title="ANNOTATION_FEATURE"),]
    data_table_selection = DataTable(source=visualizer.source_selection,columns =  columns, width=400, height=280)
    data_table_cluster = DataTable(source=visualizer.source_cluster,columns =  [ TableColumn(field="CLUSTER_ID", title="CLUSTER_ID",width = 20),
                                                                                 TableColumn(field="CUSTOMER_UTTERANCE", title="CUSTOMER_UTTERANCE",width = 200),
                                                                                 TableColumn(field="GROUND_TRUTH", title="SHOPKEEPER_UTTERANCE_GROUND_TRUTH",width=200),], width=800, height=1000)
    
    
    visualizer.source.callback = callback_cluster_visualize(data_table_cluster)
    #visualizer.source_passive.callback = callback_source(data_table_cluster)           

    unique_spatial_state= list(set(visualizer.source.data['SPATIAL_FORMATION']))
    select_spatial_state= Select( title='Spatial Formation',value ='Reset',options= ["Reset"]+unique_spatial_state,callback=callback_spatial_state(current_state_text))
    
    #unique_targets = list(set(visualizer.source.data['STATE_TARGET']))
    #select_state_target = Select( title='State Target ',value ='Reset',options= ["Reset"]+unique_targets,callback=callback_customer_location(current_state_text))
   
    display_spatial_formation= CheckboxGroup(labels=["Display Spatial Formation"], active=[],callback=callback_display_spatial_formation())
    
    display_annotation = CheckboxGroup(labels=["Display"], active=[],callback=callback_display_annotation(current_state_text))
    select_customer_annotation = Select(title='Annotation (Customer)', value='None', options=["None"]+visualizer.annotation_tag, callback=callback_customer_annotation(current_state_text))
    
    select_customer_annotation_feature= Select(title='Feature (Customer)', value='NONE', options=list(set(visualizer.source.data['CAMERA_FEATURE'])),callback=callback_customer_feature(current_state_text))
    
    unique_cluster_id= list(sorted(set(visualizer.source.data['CLUSTER_ID'])))
    select_shopkeeper_cluster = Select(title='Shopkeeper Cluster ID', value='Reset', options=["Reset"]+unique_cluster_id, callback=callback_shopkeeper_clusters())  
    
    
    #counter=collections.Counter(visualizer.source.data['neuron_clusters'])
    unique_neuron_cluster_id= list(sorted(set(visualizer.source.data['neuron_clusters'])))
    select_neuron_cluster = Select(title='Neuron Cluster ID', value='Reset', options=["Reset"]+unique_neuron_cluster_id,callback=callback_neuron_clusters())   
    
    menus = [current_state_text,display_spatial_formation,select_spatial_state,display_annotation,select_customer_annotation,select_customer_annotation_feature,select_shopkeeper_cluster,select_neuron_cluster]
    
    if current_mode is visualize_modes.AFTER_TRAINING:      
        test_index_select = Select(title='Test Index', value='0', options=map(str,range (visualizer.n_test)),callback=callback_test_examples())
        visibility_checkbox = CheckboxGroup(labels=["Proactive","Passive", "Test"], active=[0,1,2],callback=callback_visibility())
        menus.extend([test_index_select,visibility_checkbox])
    controls = widgetbox(menus, width=200)
  
    p1 = row(controls,  neuron_panel.p, data_table_cluster)
    return p1





#==============================================================================
# def callback_cluster_visualize(data_table):
#     callback = CustomJS(args=dict(current_source = visualizer.source_cluster,data_table=data_table), code="""
#         var inds = cb_obj.selected['1d'].indices;
#         var d1 = cb_obj.data;
#         console.log("length1 ",inds.length)
#         var d2 = current_source.data;
#         d2['CLUSTER_ID'] = [""]
#         d2['TYPICAL_UTTERANCE'] = [""]
#         for (i = 0; i < inds.length; i++) {
#            if (d2['CLUSTER_ID'].indexOf(d1['CLUSTER_ID'][inds[i]]) == -1){
#                d2['CLUSTER_ID'].push(d1['CLUSTER_ID'][inds[i]])
#                d2['TYPICAL_UTTERANCE'].push(d1['ground_truth_typical_utterance'][inds[i]])
#            }           
#         }
#         
#      
#         current_source.trigger('change');
#         data_table.trigger('change');
#     """)
#     return callback
#==============================================================================


def callback_cluster_visualize(data_table):
    callback = CustomJS(args=dict(current_source = visualizer.source_cluster,data_table=data_table), code="""
        var inds = cb_obj.selected['1d'].indices;
        var d1 = cb_obj.data;
        console.log("length1 ",inds.length)
        var d2 = current_source.data;
        d2['CLUSTER_ID'] = [""]
        d2['CUSTOMER_UTTERANCE'] = [""]
        d2['GROUND_TRUTH'] = [""]
        for (i = 0; i < inds.length; i++) {
         
           d2['CLUSTER_ID'].push(d1['CLUSTER_ID'][inds[i]])
           d2['CUSTOMER_UTTERANCE'].push(d1['SPEECH3'][inds[i]])
           d2['GROUND_TRUTH'].push(d1['SHOPKEEPER_SPEECH'][inds[i]])
                  
        }
        
     
        current_source.trigger('change');
        data_table.trigger('change');
    """)
    return callback





def callback_visibility():
    callback_visibility= CustomJS(args=dict(line0=neuron_panel.proactive_r, line1=neuron_panel.passive_r, line2=neuron_panel.test_r), code="""
    console.log(cb_obj.active);
    line0.visible = false;
    line1.visible = false;
    line2.visible = false;
    for (i in cb_obj.active) {
        //console.log(cb_obj.active[i]);
        if (cb_obj.active[i] == 0) {
            line0.visible = true;
        }else if (cb_obj.active[i] == 1) {
            line1.visible = true;
        } else if (cb_obj.active[i] == 2) {
            line2.visible = true;
        }
    }
    """)
    return callback_visibility  



def callback_test_examples():
    callback_test_examples = CustomJS(args=dict(source=visualizer.source_test), code="""    
        var test_index = source.data['test_index']
        var value = cb_obj.value
       
      
        for (i =0; i < test_index.length ; i++) {
            source.data['line_width'][i]= 1   
        }
        for (i =value*11; i < (value*11)+10 ; i++) {
            source.data['line_width'][i] = 3 
        }
        source.trigger('change')
    """)
    return callback_test_examples  



def callback_neuron_clusters():       
 
    callback_clusterid = CustomJS(args=dict(source_proactive=visualizer.source_proactive,source_passive=visualizer.source_passive), code="""    
     
        var source = new Array();
        source.push(source_proactive);
        source.push(source_passive);
        var color_annotate= '#B24083'
        var color_none = '#FFFFFF'       
    
        var value = cb_obj.value        
        
      
        source.forEach( function (key){  
            var cluster_id= key.data['neuron_clusters']
            if (value == "Reset"){
                for (i = 0; i < cluster_id.length; i++) {
                    key.data['fill_colors'][i] =color_none
                    key.data['line_width'][i] = 1
                }
            }else{
                for(i=0; i<cb_obj.options.length;i++){           
                    if (value ==  cb_obj.options[i]){
                        console.log("current ",value)
                        for (i = 0; i < cluster_id.length; i++) {
                            if (cluster_id[i] == value){
                                key.data['fill_colors'][i] = color_annotate
                                key.data['line_width'][i] = 2
                            }else{
                                key.data['fill_colors'][i] = color_none
                                key.data['line_width'][i] = 1
                            }  
                        }                                  
                    }    
                }           
            }  
            key.trigger('change')         
        })                    
      
    """)    
    return callback_clusterid




def callback_shopkeeper_clusters():       
 
    callback_clusterid = CustomJS(args=dict(source_proactive=visualizer.source_proactive,source_passive=visualizer.source_passive), code="""    
     
        var source = new Array();
        source.push(source_proactive);
        source.push(source_passive);
        var color_annotate= '#B24083'
        var color_none = '#FFFFFF'       
    
        var value = cb_obj.value        
        
      
        source.forEach( function (key){  
            var cluster_id= key.data['CLUSTER_ID']
            if (value == "Reset"){
                for (i = 0; i < cluster_id.length; i++) {
                    key.data['fill_colors'][i] =color_none
                    key.data['line_width'][i] = 1
                }
            }else{
                for(i=0; i<cb_obj.options.length;i++){           
                    if (value ==  cb_obj.options[i]){
                        console.log("current ",value)
                        for (i = 0; i < cluster_id.length; i++) {
                            if (cluster_id[i] == value){
                                key.data['fill_colors'][i] = color_annotate
                                key.data['line_width'][i] = 2
                            }else{
                                key.data['fill_colors'][i] = color_none
                                key.data['line_width'][i] = 1
                            }  
                        }                                  
                    }    
                }           
            }  
            key.trigger('change')         
        })                    
    
    """)    
    return callback_clusterid


def callback_spatial_state(current_state_text):       
 
    callback_location = CustomJS(args=dict(source_proactive=visualizer.source_proactive,source_passive=visualizer.source_passive,current_state_text = current_state_text), code="""    
     
        var source = new Array();
        source.push(source_proactive);
        source.push(source_passive);
        var color_annotate= '#B24083'
        var color_none = '#FFFFFF'       
    
        var value = cb_obj.value        
        
      
        source.forEach( function (key){  
            var state_target= key.data['SPATIAL_FORMATION']
            if (value == "Reset"){
                for (i = 0; i < state_target.length; i++) {
                    key.data['fill_colors'][i] =color_none
                    key.data['line_width'][i] = 1
                }
            }else{
                for(i=0; i<cb_obj.options.length;i++){           
                    if (value ==  cb_obj.options[i]){
                        console.log("current ",value)
                        for (i = 0; i < state_target.length; i++) {
                            if (state_target[i] == value){
                                key.data['fill_colors'][i] = color_annotate
                                key.data['line_width'][i] = 1
                            }else{
                                key.data['fill_colors'][i] = color_none
                                key.data['line_width'][i] = 1
                            }  
                        }                                  
                    }    
                }           
            }  
            key.trigger('change')         
        })                    
        current_state_text.text = value
    """)    
    return callback_location

  





def callback_display_annotation(current_state_text):
    callback_visibility= CustomJS(args=dict(source_proactive=visualizer.source_proactive,source_passive=visualizer.source_passive,current_state_text=current_state_text), code="""
    console.log(cb_obj.active);
    var source = new Array();
    source.push(source_proactive);
    source.push(source_passive);
  
    
    source.forEach( function (key){          
        var annotation_colors = key.data['annotation_colors']
        var state_target = key.data['SPATIAL_FORMATION']
        for (j = 0; j < annotation_colors.length; j++) {
            key.data['fill_colors'][j] = '#FFFFFF'          
            key.data['fill_alpha'][j] = 1
          
         
            
        }
        if (cb_obj.active[0] == 0) {         
            for (j = 0; j < annotation_colors.length; j++) {
                key.data['fill_colors'][j] = annotation_colors[j]
                if (state_target[j] == current_state_text.text){
                    key.data['fill_alpha'][j] = 1              
                   
                }else{
                    key.data['fill_alpha'][j] = 0.6
                       
                    
                }
            }
        }        
        key.trigger('change')     
    
    })

    """)
    return callback_visibility  


def callback_display_spatial_formation():
    callback_visibility= CustomJS(args=dict(source_proactive=visualizer.source_proactive,source_passive=visualizer.source_passive), code="""
    console.log(cb_obj.active);
    var source = new Array();
    source.push(source_proactive);
    source.push(source_passive);
    color_annotate = '#85939b'
    
    source.forEach( function (key){          
        var annotation_colors = key.data['annotate_spatial_colors']   
        
        for (j = 0; j < annotation_colors.length; j++) {
            key.data['fill_colors'][j] = '#FFFFFF' 
        }
        if (cb_obj.active[0] == 0) {         
            for (j = 0; j < annotation_colors.length; j++) {
                key.data['fill_colors'][j] = annotation_colors[j]
            }
        }        
        key.trigger('change')     
    
    })

    """)
    return callback_visibility  



def callback_customer_feature(current_state_text):       
     
    callback_feature = CustomJS(args=dict(source_proactive=visualizer.source_proactive,source_passive=visualizer.source_passive,current_state_text = current_state_text), code="""    
        var source = {"source_proactive": source_proactive, "source_passive": source_passive };    
        
       
        var color_annotate= '#f90010'
        var color_annotate_light = '#ff8c93'
        var color_none = '#FFFFFF'
        
        var value = cb_obj.value
        
        for(i=0; i<cb_obj.options.length;i++){           
            if (value ==  cb_obj.options[i]){
                console.log("current selection ",value)
                for (var key1 in source) {
                    var key = source[key1]
                    console.log("key now ",key1)
                    var camera_feature= key.data['CAMERA_FEATURE']
                    var state_target = key.data['SPATIAL_FORMATION'];
                    for (j = 0; j < camera_feature.length; j++) {
                        if (camera_feature[j] == value ){
                           if (current_state_text.text == 'Reset'){
                            
                              if (key1 == 'source_proactive'){
                                 key.data['fill_colors'][j] = color_annotate
                              }else{
                                 key.data['fill_colors'][j] = color_annotate_light
                              }
                              key.data['line_width'][j] = 1
                           }else{
                               if (state_target[j] == current_state_text.text){
                                  if (key1 == 'source_proactive'){
                                     key.data['fill_colors'][j] = color_annotate
                                  }else{
                                     key.data['fill_colors'][j] = color_annotate_light
                                  }
                                  key.data['line_width'][j] = 1
                               }
                           }                            
                        }else{
                            key.data['fill_colors'][j] = color_none
                            key.data['line_width'][j] = 1
                        }  
                    }  
                    key.trigger('change')    
                }
            }    
        }        
    """)    
    return callback_feature

def callback_customer_annotation(current_state_text):
        
     
    callback_annotation = CustomJS(args=dict(source_proactive=visualizer.source_proactive,source_passive=visualizer.source_passive,current_state_text=current_state_text), code="""    
        console.log("current location ",current_state_text.text)
        var source = {"source_proactive": source_proactive, "source_passive": source_passive };    

        var color_annotate =  ['#FFFFFF','#f90010','#003d05', '#003b68']
        var color_annotate_light =  ['#FFFFFF','#ff8c93','#609664', '#7fcddb']
        var color_none = '#FFFFFF'
        
        var value = cb_obj.value
        
        for (var key1 in source) {
            var key = source[key1]
      
            var state_target = key.data['SPATIAL_FORMATION'];
            if (value == "None"){
                for (i = 0; i < state_target.length; i++) {
                    key.data['fill_colors'][i] =color_none
                    key.data['line_width'][i] = 1
                }
            }else{
                for(i=0; i<cb_obj.options.length;i++){           
                    if (value ==  cb_obj.options[i]){
                        var name = key.data[ cb_obj.options[i]];
                        console.log("current ",  cb_obj.options[i])
                       
                        for (j = 0; j < name.length; j++) {
                            if (name[j] == 1  ){
                                if (current_state_text.text == 'Reset'){
                                    if (key1 == 'source_proactive'){ 
                                      key.data['fill_colors'][j] = color_annotate[i]
                                    }else{
                                      key.data['fill_colors'][j] = color_annotate_light[i]
                                    }
                                    key.data['line_width'][j] =1
                                }else{
                                    if (state_target[j] == current_state_text.text){
                                       if (key1 == 'source_proactive'){ 
                                          key.data['fill_colors'][j] = color_annotate[i]
                                       }else{
                                          key.data['fill_colors'][j] = color_annotate_light[i]
                                       }
                                       key.data['line_width'][j] = 1
                                    }
                                }
                             
                            }else{
                                key.data['fill_colors'][j] = color_none
                                key.data['line_width'][j] = 1
                            }  
                        }
                    }       
                }      
            }        
            key.trigger('change')        
        }
       
    """)    
    return callback_annotation


if __name__ == '__main__':   

    visualize_modes = __enum(PRE_TRAINING='pretrained_lhid3', AFTER_TRAINING_PROACTIVE='aftertrained_proactive_lhid3',AFTER_TRAINING_PASSIVE='aftertrained_passive_lhid3',AFTER_TRAINING= 'aftertrained_new_lhid3',INPUT_VECTOR= 'input_vector')
    current_mode = visualize_modes.PRE_TRAINING     
    visualizer = VisualizationDataManager(dnn)
    neuron_panel = NeuronPanel()
    show (initalizeMenu())

         
                                                                            
#==============================================================================
# def callback_annotation_shopkeeper():
#         
#     callback_annotation = CustomJS(args=dict(source=visualizer.source), code="""    
#         
#         var value = cb_obj.value
#         console.log(value);   
#        
#         var ask_queston = source.data['annotate_ask_shopkeeper']
#         var answer_question = source.data['annotate_answer_shopkeeper']
#         var new_feature = source.data['annotate_new_feature_shopkeeper']
#         var new_camera = source.data['annotate_new_camera_shopkeeper']
#         var approach = source.data['annotate_approach_shopkeeper']
#         var farewell = source.data['annotate_farewell_shopkeeper']
#         var statement = source.data['annotate_statement_shopkeeper']
#         
#         
#         if (value == "ask_question"){  
#          
#             for (i = 0; i < ask_queston.length; i++) {
#                 if (ask_queston[i] == 1){              
#                     source.data['fill_colors'][i] = 'blue'
#                 }else{
#                     source.data['fill_colors'][i] = '#D0D0D0'
#                 }  
#             }
#         }else if (value == "answer_question"){
#              for (i = 0; i < answer_question.length; i++) {
#                 if (answer_question[i] == 1){              
#                     source.data['fill_colors'][i] = 'blue'
#                 }else{
#                     source.data['fill_colors'][i] = '#D0D0D0'
#                 }  
#             }
#         }else if (value == "new_feature"){
#              for (i = 0; i < new_feature.length; i++) {
#                 if (new_feature[i] == 1){              
#                     source.data['fill_colors'][i] = 'blue'
#                 }else{
#                     source.data['fill_colors'][i] = '#D0D0D0'
#                 }  
#             }
#         }else if (value == "new_camera"){
#              for (i = 0; i < new_camera.length; i++) {
#                 if (new_camera[i] == 1){              
#                     source.data['fill_colors'][i] = 'blue'
#                 }else{
#                     source.data['fill_colors'][i] = '#D0D0D0'
#                 }  
#             }
#         }else if (value == "approach"){
#              for (i = 0; i < approach.length; i++) {
#                 if (approach[i] == 1){              
#                     source.data['fill_colors'][i] = 'blue'
#                 }else{
#                     source.data['fill_colors'][i] = '#D0D0D0'
#                 }  
#             }
#         }else if (value == "farewell"){
#              for (i = 0; i < farewell.length; i++) {
#                 if (farewell[i] == 1){              
#                     source.data['fill_colors'][i] = 'blue'
#                 }else{
#                     source.data['fill_colors'][i] = '#D0D0D0'
#                 }  
#             }
#         }else if (value == "statement"){
#              for (i = 0; i < statement.length; i++) {
#                 if (statement[i] == 1){              
#                     source.data['fill_colors'][i] = 'blue'
#                 }else{
#                     source.data['fill_colors'][i] = '#D0D0D0'
#                 }  
#             }
#         }else{
#              for (i = 0; i < statement.length; i++) {
#                      source.data['fill_colors'][i] = '#D0D0D0'
#              }
#         }
#                 
#       
#       
#         source.trigger('change')
#     """)    
#     return callback_annotation
#==============================================================================


    
#==============================================================================
# def callback_annotation_heatmap():
#         
#     callback_annotation = CustomJS(args=dict(source=visualizer.source), code="""    
#         
#         var value = cb_obj.value
#         console.log(value);   
#        
#         var ask_queston = source.data['annotate_ask_queston']
#         var statement = source.data['annotate_statement']
#         var answer_question = source.data['annotate_answer_question']
#         var polite = source.data['annotate_polite']
#         var attention = source.data['annotate_attention']
#         var comparison = source.data['annotate_comparison']
#         var time = source.data['annotate_time']
#         var request = source.data['annotate_request']
#         var other = source.data['annotate_other']
#         
#         if (value == "ask_question"){  
#          
#             for (i = 0; i < ask_queston.length; i++) {
#                 if (ask_queston[i] == 1){              
#                     source.data['colors'][i] = 'blue'
#                 }else{
#                     source.data['colors'][i] = '#D0D0D0'
#                 }  
#             }
#         }else if (value == "statement"){
#              for (i = 0; i < statement.length; i++) {
#                 if (statement[i] == 1){              
#                     source.data['colors'][i] = 'blue'
#                 }else{
#                     source.data['colors'][i] = '#D0D0D0'
#                 }  
#             }
#         }else if (value == "answer_question"){
#              for (i = 0; i < answer_question.length; i++) {
#                 if (answer_question[i] == 1){              
#                     source.data['colors'][i] = 'blue'
#                 }else{
#                     source.data['colors'][i] = '#D0D0D0'
#                 }  
#             }
#         }else if (value == "polite"){
#              for (i = 0; i < polite.length; i++) {
#                 if (polite[i] == 1){              
#                     source.data['colors'][i] = 'blue'
#                 }else{
#                     source.data['colors'][i] = '#D0D0D0'
#                 }  
#             }
#         }else if (value == "get_attention"){
#              for (i = 0; i < attention.length; i++) {
#                 if (attention[i] == 1){              
#                     source.data['colors'][i] = 'blue'
#                 }else{
#                     source.data['colors'][i] = '#D0D0D0'
#                 }  
#             }
#         }else if (value == "comparison"){
#              for (i = 0; i < comparison.length; i++) {
#                 if (comparison[i] == 1){              
#                     source.data['colors'][i] = 'blue'
#                 }else{
#                     source.data['colors'][i] = '#D0D0D0'
#                 }  
#             }
#         }else if (value == "time"){
#              for (i = 0; i < time.length; i++) {
#                 if (time[i] == 1){              
#                     source.data['colors'][i] = 'blue'
#                 }else{
#                     source.data['colors'][i] = '#D0D0D0'
#                 }  
#             }
#         }else if (value == "request"){
#              for (i = 0; i < request.length; i++) {
#                 if (request[i] == 1){              
#                     source.data['colors'][i] = 'blue'
#                 }else{
#                     source.data['colors'][i] = '#D0D0D0'
#                 }  
#             }
#         }else if (value == "other"){
#              for (i = 0; i < other.length; i++) {
#                 if (other[i] == 1){              
#                     source.data['colors'][i] = 'blue'
#                 }else{
#                     source.data['colors'][i] = '#D0D0D0'
#                 }  
#             }
#         } else{
#              for (i = 0; i < other.length; i++) {
#                      source.data['colors'][i] = '#D0D0D0'
#              }
#         }
#                 
#       
#       
#         source.trigger('change')
#     """)    
#     return callback_annotation  
#==============================================================================
 

#==============================================================================
# def generate_annoated_heatmaps():    
#     test = pd.DataFrame(dnn.activiation_array)
#     #test.reset_index(level=0, inplace=True)
#     test['example_index'] = test.index+1    
#    
#         
#     test1 = pd.melt(test, var_name='activiation_index', value_name='activiation_value', id_vars=['example_index'])
#     # Start counting from 0
#     heatmaps = {}
#     mean_square_error_dict = calc_mean_square_error(10)
#     
#     for i in mean_square_error_dict:
#         print (i)
#         print (mean_square_error_dict[i])
#         partial_data = test1.loc[test1['activiation_index'].isin(mean_square_error_dict[i])]
#         partial_data = partial_data.assign(activiation_index= 'act_num'+  partial_data.activiation_index.map(str) )  
#         
#         annoated_index = list(visualizer.annotation_train_df[ visualizer.annotation_train_df[i] == 1].index)
#      
#         partial_data['line_color'] = None 
#         partial_data.set_value(partial_data[partial_data['example_index'].isin(annoated_index)].index.tolist(),'line_color','red')   
#         
#         partial_data['SPEECH1'] = visualizer.readable_history_train_df['SPEECH1'][(partial_data['example_index']-1).tolist()].tolist()
#         partial_data['SPEECH2'] = visualizer.readable_history_train_df['SPEECH2'][(partial_data['example_index']-1).tolist()].tolist()
#         partial_data['SPEECH3'] = visualizer.readable_history_train_df['SPEECH3'][(partial_data['example_index']-1).tolist()].tolist()
#         partial_data['GROUND_TRUTH'] = visualizer.ground_truth_shopkeeper_speech_train.iloc[partial_data['example_index']-1].tolist()
# 
#         partial_data = partial_data.sort_values(['activiation_index', 'activiation_value'], ascending=[1, 0])
#         
#         partial_data['sorted_index'] = range(1,test.shape[0]+1) * len(mean_square_error_dict[i])
#       
#             
#         
#         source = ColumnDataSource(
#             data=dict(
#                     example_index=partial_data['example_index'], 
#                     activiation_index=partial_data['activiation_index'], 
#                     activiation_value=partial_data['activiation_value'],
#                     line_color = partial_data['line_color'],
#                     speech1 = partial_data['SPEECH1'] ,    
#                     speech2 = partial_data['SPEECH2'] ,   
#                     speech3 = partial_data['SPEECH3'],
#                     ground_truth = partial_data['GROUND_TRUTH'],
#                     sorted_index = partial_data['sorted_index'])
#         )
#         
#         TOOLS = "hover,save,pan,box_zoom,wheel_zoom"
#         
#         heatmap = figure(title="test",
#                    x_range=partial_data['example_index'].map(str).unique().tolist(), y_range=partial_data['activiation_index'].map(str).unique().tolist(),
#                    plot_width=2000, plot_height=2000,
#                    tools=TOOLS)
#         
#         heatmap.grid.grid_line_color = None
#         heatmap.axis.axis_line_color = None
#         heatmap.axis.major_tick_line_color = None
#         heatmap.axis.major_label_text_font_size = "5pt"
#         heatmap.axis.major_label_standoff = 0
#         
#         
#         heatmap.xaxis[0].ticker=FixedTicker(ticks=annoated_index)
#        
#         heatmap.rect(x="sorted_index", y="activiation_index", width=1, height=1,
#                source=source,
#                alpha = 'activiation_value',
#                fill_color='#1f78b4',
#                line_color='line_color')
#         
#         heatmap.select_one(HoverTool).tooltips = [ 
#             ('index', '@example_index'),
#             ('value', '@activiation_value'),
#             ("customer1", "@speech1"),                
#             ("shopkeeper2", "@speech2"),             
#             ('customer3','@speech3'),
#             ('ground_truth','@ground_truth')
#         ]
# 
# 
#         heatmaps[i] = heatmap    
# 
#     return heatmaps
# 
# 
# def generate_all_heatmaps():    
#     test = pd.DataFrame(dnn.activiation_array)
#     #test.reset_index(level=0, inplace=True)
#     test['example_index'] = test.index+1    
#    
#         
#     test1 = pd.melt(test, var_name='activiation_index', value_name='activiation_value', id_vars=['example_index'])
#     # Start counting from 0
#     heatmaps = {}
#     # Start counting from 1
#     n = 40
#     for count, element in enumerate(dnn.activiation_array.T, 1):         
#         if count % n == 0  :           
#             print (str(count-n) + '  '+str(count) )
#             partial_data = test1.loc[(test1['activiation_index'] >= count-n) & (test1['activiation_index'] <= count )]                
#   
#             partial_data = partial_data.assign(activiation_index= 'act_num'+  partial_data.activiation_index.map(str) )  
#             
#             #annoated_index = list(visualizer.annotation_train_df[ visualizer.annotation_train_df[i] == 1].index)
#          
#             #partial_data['line_color'] = None 
#             #partial_data.set_value(partial_data[partial_data['example_index'].isin(annoated_index)].index.tolist(),'line_color','red')   
#             
#             partial_data['SPEECH1'] = visualizer.readable_history_train_df['SPEECH1'][(partial_data['example_index']-1).tolist()].tolist()
#             partial_data['SPEECH2'] = visualizer.readable_history_train_df['SPEECH2'][(partial_data['example_index']-1).tolist()].tolist()
#             partial_data['SPEECH3'] = visualizer.readable_history_train_df['SPEECH3'][(partial_data['example_index']-1).tolist()].tolist()
#             partial_data['GROUND_TRUTH'] = visualizer.ground_truth_shopkeeper_speech_train.iloc[partial_data['example_index']-1].tolist()
#     
#             partial_data = partial_data.sort_values(['activiation_index', 'activiation_value'], ascending=[1, 0])
#             
#             if count == dnn.activiation_array.shape[1]:
#                 partial_data['sorted_index'] = range(1,test.shape[0]+1) * (n)
#             else:
#                 partial_data['sorted_index'] = range(1,test.shape[0]+1) * (n+1)
#           
#               
#             source = ColumnDataSource(
#                 data=dict(
#                         example_index=partial_data['example_index'], 
#                         activiation_index=partial_data['activiation_index'], 
#                         activiation_value=partial_data['activiation_value'],                      
#                         speech1 = partial_data['SPEECH1'] ,    
#                         speech2 = partial_data['SPEECH2'] ,   
#                         speech3 = partial_data['SPEECH3'],
#                         ground_truth = partial_data['GROUND_TRUTH'],
#                         sorted_index = partial_data['sorted_index'])
#             )
#             
#             TOOLS = "hover,save,pan,box_zoom,wheel_zoom"
#             
#             heatmap = figure(title="test",
#                        x_range=partial_data['example_index'].map(str).unique().tolist(), y_range=partial_data['activiation_index'].map(str).unique().tolist(),
#                        plot_width=2000, plot_height=2000,
#                        tools=TOOLS)
#             
#             heatmap.grid.grid_line_color = None
#             heatmap.axis.axis_line_color = None
#             heatmap.axis.major_tick_line_color = None
#             heatmap.axis.major_label_text_font_size = "5pt"
#             heatmap.axis.major_label_standoff = 0        
#             
#            
#            
#             heatmap.rect(x="sorted_index", y="activiation_index", width=1, height=1,
#                    source=source,
#                    alpha = 'activiation_value',
#                    fill_color='#1f78b4',
#                    line_color=None)
#             
#             heatmap.select_one(HoverTool).tooltips = [ 
#                 ('index', '@example_index'),
#                 ('value', '@activiation_value'),
#                 ("customer1", "@speech1"),                
#                 ("shopkeeper2", "@speech2"),             
#                 ('customer3','@speech3'),
#                 ('ground_truth','@ground_truth')
#             ]
#     
#     
#             heatmaps[count] = heatmap    
# 
#     return heatmaps
#==============================================================================



#==============================================================================
# def calc_mean_square_error(n):  
#     mean_square_error_dict = {}
#     mean_square_error = pd.DataFrame()
#     
# 
#     for column2 in visualizer.annotation_train_df:   
#       #  print (column2)
#         mean_square_error[column2] =[mean_squared_error(visualizer.annotation_train_df[column2], column) for column in dnn.activiation_array.T] 
#         #mean_square_error[column2] =[np.correlate(np.asarray(visualizer.annotation_train_df[column2]).astype('float32'), column) for column in dnn.activiation_array.T] 
#     
#     for column in mean_square_error:
#         mean_square_error_dict[column] = mean_square_error[column].argsort()[:n].values      
#         #print(mean_square_error[column].argsort()[:10].values) #get 3 smallest for each activiation array
#     return mean_square_error_dict
#==============================================================================




#==============================================================================
#     heatmaps = generate_annoated_heatmaps()
#     #heatmaps = generate_all_heatmaps()
#     
#     for count, h in heatmaps.iteritems():   
#          print (count)
#          output_file("./results/heatmap_"+str(count)+".html", title="heatmap_"+str(count))
#          p2 =  heatmaps[count]   
#          show (p2)
#==============================================================================

