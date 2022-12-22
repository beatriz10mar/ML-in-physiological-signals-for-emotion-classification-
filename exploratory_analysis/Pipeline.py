


# In[1]:
import pkg_resources
# pkg_resources.require("seaborn==0.11.2")
# pkg_resources.require("pandas==1.3.4")
# pkg_resources.require("neurokit2==0.1.5")
# pkg_resources.require("numpy==1.20.3")
  
import warnings
from Neurokit_Function import *
from Stationarity import *
from Reading import *
from Exploratory import *


import seaborn as sns
import neurokit2 as nk
import pandas as pd
import numpy as np
import csv
import sys
import pickle
import os

from statsmodels.tsa.seasonal import seasonal_decompose
#from scipy import signal


sys.path.append(
    "C:\\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques")
#from scipy.interpolate import PchipInterpolator

# Acessing files functions
#from exploratory_analysis_function import *


# In[2]:


path_excel = r'C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Processing the Signals'
path = r'C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Database'


# ### Reading Data

# In[3]:


def open_all(csvfile, path):

    # Save participants information
    id_information = pd.DataFrame([])

    emotional_sequence = []
    seq_list = []
    id_code = []  # List of id code of all participants
    dates_S1 = []  # List of session1 date of all participants
    dates_S2 = []  # List of session2 date of all participants
    times_S1 = []  # List of session1 time of all participants
    times_S2 = []  # List of session2 time of all participants
    session1_filesname = []  # List of session1 filesnames of all participants
    session2_filesname = []  # List of session2 filesnames of all participants

    all_dataS1 = []  # List of session1 data of all participants
    all_dataS2 = []  # List of session2 data of all participants
    index_start_S1 = []  # List of session1 start indices of all participants
    index_end_S1 = []  # List of session1 end indices of all participants
    index_start_S2 = []  # List of session2 start indices of all participants
    index_end_S2 = []  # List of session2 end indices of all participants
    index_triggers_S1 = []  # List of session1 triggers indices of all participants
    index_triggers_S2 = []  # List of session2 triggers indices of all participants

    # Save Data
    dict_data = {}

    b_pre_dict = {}
    f_pre_dict = {}
    h_pre_dict = {}
    n_pre_dict = {}
    baseline_list = []
    fear_list = []
    happy_list = []
    neutral_list = []
    dict_emotion = {}

    all_s1 = {}

    with open(csvfile, newline='') as csvfile:
        next(csvfile)  # ignoring the first line
        spamreader = csv.reader(csvfile, delimiter=';')
        for row in spamreader:
            
            # gather information from the csv file
            code = row[0]  # ID code of the participant in line
            print(code)
            session1_name = row[1]  # Session1 filename
            session2_name = row[2]  # Session2 filename
            # List of the sequence of emotional conditions
            sequence = ['Baseline', row[3], row[4], row[5]]
            seq = row[3][0]+row[4][0]+row[5][0]

            id_code.append(code)  # Adding ID code
            sequence = ['Baseline', row[3], row[4], row[5]]  # Adding sequence
            emotional_sequence.append(sequence)  # Adding sequence
            seq_list.append(seq)
            
            # Adding session1 filename
            session1_filesname.append(session1_name)
            
            # Adding session2 filename
            session2_filesname.append(session2_name)

            ############################################ FIRST SESSION ############################################

            ######### Read the data of first session #########

            # path=r'C:\Users\Acer\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Material_Carolina\Data'
            filepath = path+'\\'+code+'\\'+session1_name+'.txt'

            # Using reading function to read the data file
            dataS1, headerS1, sampling_rateS1, resolutionS1, dateS1, timeS1, sensorS1, labelS1, columnS1, sleeve_colorS1 = reading_function(
                filepath)
            dates_S1.append(dateS1)  # Adding session1 date
            times_S1.append(timeS1)  # Adding session1 time

            ######### Sensor confirmation #########
            x_nSeq_S1, x_trig_S1, x_EMG_MF_S1, x_EMG_TR_S1, x_EDA_S1, x_ECG_S1 = sensor_confirmation(
                sensorS1, sleeve_colorS1, labelS1, columnS1)
            
           
            trigger = dataS1[:, x_trig_S1]  # Acessing to triggers column
            EMG_MF = dataS1[:, x_EMG_MF_S1]  # Acessing to EMG_MF column
            EMG_TR = dataS1[:, x_EMG_TR_S1]  # Acessing to EMG_TR column
            EDA = dataS1[:, x_EDA_S1]  # Acessing to EDA column
            ECG = dataS1[:, x_ECG_S1]  # Acessing to ECG column

            ######### Find triggers and confirmation #########
            index_trig_true, index_trig = trigger_function(trigger, sampling_rateS1,sequence)
            index_start_S1.append(index_trig[0:4])  # Triggers when set begins
            index_end_S1.append(index_trig[1:5])  # Triggers when set ends
            index_triggers_S1.append(index_trig)  # All triggers

            print(index_trig)
            print(index_trig_true)
            trig_error={}
            for i in range(0,len(index_trig_true)-1):
                trig_error[i]= (abs(index_trig[i]-index_trig_true[i])/index_trig[i])*100
            print([code,trig_error])
            
    #         ######### Rearreange session data #########
    #         # Rearrange data (always the same sequence)
    #         concatenated_dataS1 = np.column_stack((EMG_MF, EMG_TR, EDA, ECG))

    #         all_s1[code] = concatenated_dataS1
            
    #         if len(index_trig) > 4:
    #             dict_data[code] = concatenated_dataS1[index_trig[0]:index_trig[4], :]  # retirei index_trig[4]+1

    #         ######### List emotional condition #########

    #         for i in range(0, len(sequence)):
    #             if sequence[i] == 'Baseline':
    #                 b_start = index_trig[i]
    #                 b_end = index_trig[i+1]
    #                 # retirei index_trig[4]+1
    #                 b_pre_dict[code] = concatenated_dataS1[b_start:b_end, :]

    #             elif sequence[i] == 'Fear':
    #                 f_start = index_trig[i]
    #                 f_end = index_trig[i+1]
    #                 # retirei index_trig[4]+1
    #                 f_pre_dict[code] = concatenated_dataS1[f_start:f_end, :]

    #             elif sequence[i] == 'Happy':
    #                 h_start = index_trig[i]
    #                 h_end = index_trig[i+1]
    #                 # retirei index_trig[4]+1
    #                 h_pre_dict[code] = concatenated_dataS1[h_start:h_end, :]

    #             elif sequence[i] == 'Neutral':
    #                 n_start = index_trig[i]
    #                 n_end = index_trig[i+1]
    #                 # retirei index_trig[4]+1
    #                 n_pre_dict[code] = concatenated_dataS1[n_start:n_end, :]

    # id_information = pd.DataFrame({'Participant ID': id_code, 'Emotional sequence': emotional_sequence, 'Sequence': seq_list,
    #                               'Triggers index': index_triggers_S1, 'Triggers start': index_start_S1, 'Triggers end': index_end_S1, 'Date S1': dates_S1, 'Time S1': times_S1})

    # baseline_list.append(b_pre_dict)
    # fear_list.append(f_pre_dict)
    # happy_list.append(h_pre_dict)
    # neutral_list.append(n_pre_dict)

    # dict_emotion['Baseline'] = baseline_list
    # dict_emotion['Fear'] = fear_list
    # dict_emotion['Happy'] = happy_list
    # dict_emotion['Neutral'] = neutral_list

    return id_information, dict_emotion, dict_data, index_trig_true, index_trig, trig_error

  

# In[4]:


# Run only one time
id_information, dict_emotion, dict_data, index_trig_true, index_trig , trig_error= open_all('data_id.csv', path)

# Only for 2 participants
#id_information, dict_emotion, baseline_list, dict_data, all_s1 = open_all('teste.csv')


# In[5]:


filename = 'id_information_file'
file = open(filename, 'wb')
pickle.dump(id_information, file)
file.close()


# In[6]:


filename = 'dict_data_file'
file = open(filename, 'wb')
pickle.dump(dict_data, file)
file.close()


# In[7]:


filename = 'dict_emotion_file'
file = open(filename, 'wb')
pickle.dump(dict_emotion, file)
file.close()


# ### Acessing participants information and data

# In[8]:


id_information = pickle.load(open('id_information_file', 'rb'))
dict_data = pickle.load(open('dict_data_file', 'rb'))
dict_emotion = pickle.load(open('dict_emotion_file', 'rb'))


# In[9]:


# ### Preprocessing Features

# In[10]:


def all_preprocessed_features(dict_emotion, condition, sampling_rate):

    dic_features = {}

    for participant in dict_emotion[condition][0].keys():

        print(participant)

        # if (condition=='Happy' and participant=='C028'):
        #    a=605829
        # elif (condition=='Happy' and participant=='C026'):
        #    a=60695
        # elif (condition=='Neutral' and participant=='C014'):
        #    a=667227
        # else:
        a = len(dict_emotion[condition][0][participant][:, 0])

        # print(a)
        # Features
        # MF_duration_activity, MF_peak_activity, MF_mean_activity, MF_area_activity, MF_amplitude_activity = signal_process(
        #     dict_emotion[condition][0][participant][:, 0], 'EMG', sampling_rate, a)
        # # print("a")
        # TR_duration_activity, TR_peak_activity, TR_mean_activity, TR_area_activity, TR_amplitude_activity = signal_process(
        #     dict_emotion[condition][0][participant][:, 1], 'EMG', sampling_rate, len(dict_emotion[condition][0][participant][:, 1]))
        # # print("b")
        # eda_symp, scr_height, scr_amplitude, scr_risetime, scr_recoverytime, eda_tonic, eda_phasic = signal_process(
        #     dict_emotion[condition][0][participant][:, 2], 'EDA', sampling_rate, len(dict_emotion[condition][0][participant][:, 2]))
        # # print("C")
        ecg_Rpeaks, ecg_rate, t_duration, st_interval, p_duration, tp_interval , pr_interval, qrs_duration, rr_pre, rr_pos = signal_process(
            dict_emotion[condition][0][participant][:, 3], 'ECG', sampling_rate, len(dict_emotion[condition][0][participant][:, 3]))

        dic_features[participant] = {#'MF_duration_activity': MF_duration_activity, 'MF_peak_activity': MF_peak_activity,
        #                              'MF_mean_activity': MF_mean_activity, 'MF_area_activity': MF_area_activity,
        #                              'MF_amplitude_activity': MF_amplitude_activity,
        #                              'TR_duration_activity': TR_duration_activity, 'TR_peak_activity': TR_peak_activity,
        #                              'TR_mean_activity': TR_mean_activity, 'TR_area_activity': TR_area_activity,
        #                              'TR_amplitude_activity': TR_amplitude_activity,
        #                              'scr_height': scr_height, 'scr_amplitude': scr_amplitude, 'scr_risetime': scr_risetime,
        #                              'scr_recoverytime': scr_recoverytime, 'eda_tonic': eda_tonic, 'eda_phasic': eda_phasic, 'eda_symp': eda_symp,
                                     'ecg_rate': ecg_rate, 't_duration': t_duration, 'p_duration': p_duration,
                                     'st_interval': st_interval, 'tp_interval': tp_interval, 'pr_interval': pr_interval,
                                     'qrs_duration': qrs_duration, 'rr_pre': rr_pre, 'rr_pos': rr_pos,
                                      'ecg_Rpeaks': ecg_Rpeaks}
    #Errors detected
    #if participant == 'B013':
    #    dic_baseline['B013']['ecg_Rpeaks'].loc[137632]=0
        
    return dic_features


# Creates a dictionary where the keys are the several participants. Inside each key there is other dictionary with features keys and the correspondent values

# In[11]:


sampling_rate = 1000


# As 8 linhas seguintes são para por em comentario no final

# In[12]:


dic_features_b = all_preprocessed_features(
    dict_emotion, 'Baseline', sampling_rate)


# In[13]:


filename = 'dic_baseline_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(dic_features_b, file)
file.close()


# In[14]:


dic_features_f = all_preprocessed_features(dict_emotion, 'Fear', sampling_rate)


# In[15]:


filename = 'dic_fear_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(dic_features_f, file)
file.close()


# In[16]:


dic_features_h = all_preprocessed_features(
    dict_emotion, 'Happy', sampling_rate)


# In[17]:


filename = 'dic_happy_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(dic_features_h, file)
file.close()


# In[18]:


dic_features_n = all_preprocessed_features(
    dict_emotion, 'Neutral', sampling_rate)


# In[19]:


filename = 'dic_neutral_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(dic_features_n, file)
file.close()


# ### Acessing dict with pre-processed features

# In[20]:


dic_baseline = pickle.load(open(path_excel+'/'+'dic_baseline_file', 'rb'))
dic_fear = pickle.load(open(path_excel+'/'+'dic_fear_file', 'rb'))
dic_happy = pickle.load(open(path_excel+'/'+'dic_happy_file', 'rb'))
dic_neutral = pickle.load(open(path_excel+'/'+'dic_neutral_file', 'rb'))


# In[Baseline]

mean_b, s_b, conf_neg_b, conf_pos_b, lin_b, lin_time_b, df_b, df_norm_b,df_interpol_b=graph_profile(dic_baseline, dic_baseline, "Baseline", sampling_rate, ["B","C"])

rr_b, rr2_b, retlist_b, percent_b, index_b = count_zeros(lin_b, "Baseline", conf_pos_b, conf_neg_b)
 
# fig_mean_b = fig_mean("Baseline",lin_time_b,mean_b,conf_pos_b,conf_neg_b)
# fig_CI_b = fig_CI("Baseline",lin_time_b,conf_pos_b,conf_neg_b)
# fig_scatter_b = fig_scatter("Baseline",mean_b,conf_pos_b,conf_neg_b,rr_b)
# fig_histo_b,mean_zeros_b,mean_non_zeros_b = fig_histo("Baseline",mean_b,rr_b,rr2_b) 

# In[clust]

hr_clust_neg_b,hr_clust_pos_b, hr_clust_null_b, name_clust_neg_b,name_clust_pos_b, name_clust_null_b=clustering(dic_baseline,df_b,df_norm_b,mean_b,rr_b, rr2_b, 7)

# In[]

#Import ElbowVisualizer

from yellowbrick.cluster import KElbowVisualizer

model = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30),timings= True)
# metric='calinski_harabasz',
# metric='silhouette',
visualizer.fit(np.asarray(hr_clust_neg_b))        # Fit data to visualizer

visualizer.show()   

# # In[]
# visualizer = KElbowVisualizer(model, k=(2,30), metric='calinski_harabasz',timings= True)
# # metric='calinski_harabasz',
# # metric='silhouette',
# visualizer.fit(np.asarray(mean_clust_neg_f))        # Fit data to visualizer
# visualizer.show()   

   
# In[kmeans_pos_b]:
    
y_kmeans_pos_b,clust_pos_b, count_names_pos_b, count_time_pos_b, X_pos_b, percent_pos_b, slope_pos_b= kmeans_clust("Baseline","pos",hr_clust_pos_b,5,7,name_clust_pos_b)
y_kmeans_neg_b,clust_neg_b, count_names_neg_b, count_time_neg_b, X_neg_b, percent_neg_b, slope_neg_b= kmeans_clust("Baseline","neg",hr_clust_neg_b,5,7,name_clust_neg_b)

# In[Neutral]

mean_n,s_n, conf_neg_n, conf_pos_n, lin_n, lin_time_n, df_n, df_norm_n,df_interpol_n = graph_profile(
    dic_neutral, dic_baseline, "Neutral", sampling_rate, ["B","C"])

rr_n, rr2_n, retlist_n, percent_n, index_n = count_zeros(lin_n, "Neutral", conf_pos_n, conf_neg_n)

# fig_mean_n=fig_mean("Neutral",lin_time_n,mean_n,conf_pos_n,conf_neg_n)
# fig_CI_n=fig_CI("Neutral",lin_time_n,conf_pos_n,conf_neg_n)
# fig_scatter_n=fig_scatter("Neutral",mean_n,conf_pos_n,conf_neg_n,rr_n)
# fig_histo_n,mean_zeros_n,mean_zeros_new_n=fig_histo("Neutral",mean_n,rr_n,rr2_n)

# In[]
hr_clust_neg_n,hr_clust_pos_n, hr_clust_null_n, name_clust_neg_n,name_clust_pos_n, name_clust_null_n=clustering(dic_neutral, df_n, df_norm_n, mean_n,rr_n, rr2_n, 7)

# In[]: 
y_means_pos_n,clust_pos_n, count_names_pos_n, count_time_pos_n, X_pos_n, percent_pos_n, slope_pos_n= kmeans_clust("Neutral","pos",hr_clust_pos_n,7,7,name_clust_pos_n)
y_means_neg_n,clust_neg_n, count_names_neg_n, count_time_neg_n, X_neg_n, percent_neg_n, slope_neg_n= kmeans_clust("Neutral","neg",hr_clust_neg_n,7,7,name_clust_neg_n)

# In[Happy]

mean_h, s_h, conf_neg_h, conf_pos_h, lin_h, lin_time_h, df_h, df_norm_h, df_interpol_n = graph_profile(
    dic_happy, dic_baseline, "Happy", sampling_rate, ["B","C"])

rr_h, rr2_h, retlist_h, percent_h, index_h = count_zeros(lin_h, "Happy", conf_pos_h, conf_neg_h)

# fig_mean_h=fig_mean("Happy",lin_time_h,mean_h,conf_pos_h,conf_neg_h)
# fig_CI_h=fig_CI("Happy",lin_time_h,conf_pos_h,conf_neg_h)
# fig_scatter_h=fig_scatter("Happy",mean_h,conf_pos_h,conf_neg_h,rr_h)
# fig_histo_f,mean_zeros_h,mean_zeros_new_h =fig_histo("Happy",mean_h,rr_h,rr2_h)

# In[]
hr_clust_neg_h,hr_clust_pos_h,hr_clust_null_h,name_clust_neg_h,name_clust_pos_h, name_clust_null_h =clustering(dic_happy, df_h, df_norm_h, mean_h,rr_h, rr2_h, 7)

# In[]:    
    
y_means_pos_h,clust_pos_h, count_names_pos_h, count_time_pos_h, X_pos_h, percent_pos_h, slope_pos_h= kmeans_clust("Happy","pos",hr_clust_pos_h,6,7,name_clust_pos_h)
y_means_neg_h,clust_neg_h, count_names_neg_h, count_time_neg_h, X_neg_h, percent_neg_h, slope_neg_h= kmeans_clust("Happy","neg",hr_clust_neg_h,6,7,name_clust_neg_h)

# In[Fear]

mean_f, s_f, conf_neg_f, conf_pos_f, lin_f, lin_time_f, df_f, df_norm_f, df_interpol_f= graph_profile(
    dic_fear, dic_baseline, "Fear", sampling_rate, ["B","C"])

rr_f, rr2_f, retlist_f, percent_f, index_f = count_zeros(lin_f, "Fear", conf_pos_f, conf_neg_f)

# fig_mean_f=fig_mean("Fear",lin_time_f,mean_f,conf_pos_f,conf_neg_f)
# fig_CI_f=fig_CI("Fear",lin_time_f,conf_pos_f,conf_neg_f)
# fig_scatter_f=fig_scatter("Fear",mean_f,conf_pos_f,conf_neg_f,rr_f)
# fig_histo_f,mean_zeros_f,mean_zeros_new_f = fig_histo("Fear",mean_f,rr_f,rr2_f)

# In[]
hr_clust_neg_f,hr_clust_pos_f, hr_clust_null_f,name_clust_neg_f,name_clust_pos_f, name_clust_null_f = clustering(dic_fear, df_f, df_norm_f, mean_f,rr_f, rr2_f, 7)

# In[]:    
y_means_pos_f,clust_pos_f, count_names_pos_f, count_time_pos_f, X_pos_f, percent_pos_f, slope_pos_f= kmeans_clust("Fear","pos",hr_clust_pos_f,6,7,name_clust_pos_f)
y_means_neg_f,clust_neg_f, count_names_neg_f, count_time_neg_f, X_neg_f, percent_neg_f, slope_neg_f= kmeans_clust("Fear","neg",hr_clust_neg_f,7,7,name_clust_neg_f)

# In[save_clust]
file = open(path_excel+'/'+'clust_baseline_file', 'wb')
pickle.dump([clust_pos_b, clust_neg_b, slope_pos_b, slope_neg_b, name_clust_pos_b, name_clust_neg_b, name_clust_null_b], file)
file.close()

file = open(path_excel+'/'+'clust_neutral_file', 'wb')
pickle.dump([clust_pos_n,clust_neg_n, slope_pos_n, slope_neg_n, name_clust_pos_n, name_clust_neg_n, name_clust_null_n], file)
file.close()

file = open(path_excel+'/'+'clust_fear_file', 'wb')
pickle.dump([clust_pos_f,clust_neg_f, slope_pos_f, slope_neg_f, name_clust_pos_f, name_clust_neg_f, name_clust_null_f], file)
file.close()

file = open(path_excel+'/'+'clust_happy_file', 'wb')
pickle.dump([clust_pos_h,clust_neg_h, slope_pos_h, slope_neg_h, name_clust_pos_h, name_clust_neg_h, name_clust_null_h], file)
file.close()



# ### Stationarity Analysis

# In[]:
    
[clust_pos_b, clust_neg_b, slope_pos_b, slope_neg_b, name_clust_pos_b, name_clust_neg_b, name_clust_null_b] = pickle.load(open(path_excel+'/'+'clust_baseline_file', 'rb'))
[clust_pos_f, clust_neg_f, slope_pos_f, slope_neg_f, name_clust_pos_f, name_clust_neg_f, name_clust_null_f] = pickle.load(open(path_excel+'/'+'clust_fear_file', 'rb'))
[clust_pos_h, clust_neg_h, slope_pos_h, slope_neg_h, name_clust_pos_h, name_clust_neg_h, name_clust_null_h] = pickle.load(open(path_excel+'/'+'clust_happy_file', 'rb'))
[clust_pos_n, clust_neg_n, slope_pos_n, slope_neg_n, name_clust_pos_n, name_clust_neg_n, name_clust_null_n] = pickle.load(open(path_excel+'/'+'clust_neutral_file', 'rb'))
# In[21]:

dic_baseline = pickle.load(open(path_excel+'/'+'dic_baseline_file', 'rb'))
dic_fear = pickle.load(open(path_excel+'/'+'dic_fear_file', 'rb'))
dic_happy = pickle.load(open(path_excel+'/'+'dic_happy_file', 'rb'))
dic_neutral = pickle.load(open(path_excel+'/'+'dic_neutral_file', 'rb'))

# In[]:
def all_stationarity(dic):

    d = {}
    d_kpss = {}
    status = {}
    diff = {}
    res = {}
    detrended = {}
    detrended_df = {}

    for participant in dic.keys():
        d[participant] = {}
        d_kpss[participant] = {}
        status[participant] = {}
        diff[participant] = {}
        res[participant] = {}
        detrended[participant] = {}
        detrended_df[participant] = {}

    for participant in dic.keys():
        for feature in dic[participant].keys():
            
            # Do this analysis just to some features and with a len bigger than 2
            # Tonic e Phasic têm número de amostras igual ao sinal de 10 min

            if feature != 'eda_symp' and feature != 'eda_tonic' and feature != 'eda_phasic' and feature != 'ecg_Rpeaks' and feature != 'rr_pos' and feature != 'rr_pre' and feature != 'scr_height' and feature != 'scr_amplitude' and feature != 'scr_risetime' and feature != 'scr_recoverytime' and len(dic[participant][feature]) > 2:

                dic[participant][feature] = [
                    x for x in dic[participant][feature] if str(x) != 'nan']

                sTest = StationarityTests()
                
                #ADF Test
                results_adf = ADF_Stationarity_Test(
                    sTest, dic[participant][feature])
                l = [sTest.isStationary, results_adf['ADF Test Statistic'], results_adf['P-Value'],
                     results_adf['# Lags Used'], results_adf['# Observations Used']]

                #KPSS Test
                results_kpss = KPSS_Stationarity_Test(
                    sTest, dic[participant][feature])
                f = [sTest.isStationary, results_kpss[0],
                     results_kpss[1], results_kpss[2], results_kpss[3]]
                
                d[participant][feature] = l
                d_kpss[participant][feature] = f
                
            else:
                #Just save the lenght of the features not analysed
                l = [len(feature)]
                d[participant][feature] = l
                d_kpss[participant][feature] = l
                continue

            #Conditions of concordance to both tests
            if l[0] == True and f[0] == False:
                status[participant][feature] = 'Non-stationarity'

            elif l[0] == False and f[0] == True:
                status[participant][feature] = 'Stationarity'

            elif l[0] == False and f[0] == False:
                
                
                res[participant][feature] = seasonal_decompose(pd.DataFrame.from_dict(
                    dic[participant][feature]), model='additive', extrapolate_trend='freq', period=len(dic[participant][feature])//2)

                detrended[participant][feature] = (pd.DataFrame.from_dict(
                    dic[participant][feature])).subtract(res[participant][feature].trend, axis=0)
                
                #ADF Test
                results_adf_detrend = ADF_Stationarity_Test(
                    sTest, detrended[participant][feature])
                l_trend = [sTest.isStationary, results_adf_detrend['ADF Test Statistic'], results_adf_detrend['P-Value'],
                            results_adf_detrend['# Lags Used'], results_adf_detrend['# Observations Used']]

                #KPSS Test
                results_kpss_detrend = KPSS_Stationarity_Test(
                    sTest, detrended[participant][feature])
                f_trend = [sTest.isStationary, results_kpss_detrend[0],
                            results_kpss_diff[1], results_kpss_detrend[2], results_kpss_detrend[3]]

                if l_trend[0]==False and f_trend[0]==True:
                    status[participant][feature]='Stationarity after detrending'
                elif l_trend[0]==True and f_trend[0]==False:
                    status[participant][feature]='Non-stationarity after detrending'
                elif l_trend[0]==False and f_trend[0]==False:
                    status[participant][feature]='Inconclusive with KPSS non-stationary after detrending'
                elif l_trend[0]==True and f_trend[0]==True:
                    status[participant][feature]='Inconclusive with KPSS Stationary after detrending'

            elif l[0] == True and f[0] == True:
                
                #Differenciate the time serie
                diff[participant][feature] = dic[participant][feature] - \
                    np.roll(dic[participant][feature], 1)
                
                #ADF Test
                results_adf_diff = ADF_Stationarity_Test(
                    sTest, diff[participant][feature])
                l_diff = [sTest.isStationary, results_adf_diff['ADF Test Statistic'], results_adf_diff['P-Value'],
                          results_adf_diff['# Lags Used'], results_adf_diff['# Observations Used']]

                #KPSS Test
                results_kpss_diff = KPSS_Stationarity_Test(
                    sTest, diff[participant][feature])
                f_diff = [sTest.isStationary, results_kpss_diff[0],
                          results_kpss_diff[1], results_kpss_diff[2], results_kpss_diff[3]]

                if l_diff[0] == False and f_diff[0] == True:
                    status[participant][feature] = 'Stationarity after differencing'
                elif l_diff[0] == True and f_diff[0] == False:
                    status[participant][feature] = 'Non-stationarity after differencing'
                elif l_diff[0] == False and f_diff[0] == False:
                    status[participant][feature] = 'Inconclusive with KPSS non-stationary after differencing'
                elif l_diff[0] == True and f_diff[0] == True:
                    status[participant][feature] = 'Inconclusive with KPSS Stationary after differencing'

    d = pd.DataFrame.from_dict(d)
    d_kpss = pd.DataFrame.from_dict(d_kpss)
    status = pd.DataFrame.from_dict(status)
    detrend = pd.DataFrame.from_dict(detrended)

    #Calculate the percentage of each result 
    number = status.apply(pd.Series.value_counts, axis=0).sum(axis=1)
    total_number = status.size
    percent_number = 100*number/total_number

    return d, status, d_kpss, percent_number

 # In[22]:


warnings.filterwarnings("ignore")

stationarity_results_b, status_b, kpss_b, percent_b= all_stationarity(
    dic_baseline)

print(percent_b)

filename = 'percent_baseline_file'
file=open(path_excel+'/'+filename,'wb')
pickle.dump(percent_b,file)
file.close()
# In[23]:

stationarity_results_f, status_f, kpss_f, percent_f = all_stationarity(
    dic_fear)

print(percent_f)

filename = 'percent_fear_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(percent_f, file)
file.close()
# In[24]:

stationarity_results_h, status_h, kpss_h, percent_h = all_stationarity(
    dic_happy)

print(percent_h)

filename = 'percent_happy_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(percent_h, file)
file.close()
# In[25]:

stationarity_results_n, status_n, kpss_n, percent_n = all_stationarity(
    dic_neutral)

print(percent_n)

filename = 'percent_neutral_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(percent_n, file)
file.close()


# In[26]:

filepath_stat_adf = path_excel+'/'+'stationarity_results_adf.xlsx'
writer = pd.ExcelWriter(filepath_stat_adf, engine='xlsxwriter')
stationarity_results_b.to_excel(writer, sheet_name='Baseline', engine='xlsxwriter')
stationarity_results_f.to_excel(writer, sheet_name='Fear', engine='xlsxwriter')
stationarity_results_h.to_excel(writer, sheet_name='Happy', engine='xlsxwriter')
stationarity_results_n.to_excel(writer, sheet_name='Neutral', engine='xlsxwriter')
writer.save()

# In[ ]
filepath_stat_kpss = path_excel+'/'+'stationarity_results_kpss.xlsx'
writer = pd.ExcelWriter(filepath_stat_kpss, engine='xlsxwriter')
kpss_b.to_excel(writer, sheet_name='Baseline', engine='xlsxwriter')
kpss_f.to_excel(writer, sheet_name='Fear', engine='xlsxwriter')
kpss_h.to_excel(writer, sheet_name='Happy', engine='xlsxwriter')
kpss_n.to_excel(writer, sheet_name='Neutral', engine='xlsxwriter')
writer.save()

# In[27]

detr= ["Stationarity after detrending", 'Non-stationarity after detrending','Inconclusive with KPSS non-stationary after detrending', 'Inconclusive with KPSS Stationary after detrending']
dif= ["Stationarity after differencing", 'Non-stationarity after differencing','Inconclusive with KPSS non-stationary after differencing', 'Inconclusive with KPSS Stationary after differencing']

status_b_style = (status_b.style
                  .applymap(lambda x: 'background-color : %s' % 'blue' if x == 'Non-stationarity' else '')
                  .applymap(lambda x: 'background-color : %s' % 'green' if x == 'Stationarity' else '')
                  .applymap(lambda x: 'background-color : %s' % 'red' if x in detr else '')
                  .applymap(lambda x: 'background-color : %s' % 'orange' if x in dif else ''))

status_f_style = (status_f.style
                  .applymap(lambda x: 'background-color : %s' % 'red' if x == 'Non-stationarity' else '')
                  .applymap(lambda x: 'background-color : %s' % 'green' if x == 'Inconclusive with KPSS Stationary' else '')
                  .applymap(lambda x: 'background-color : %s' % 'blue' if x == 'Inconclusive with KPSS non-Stationary' else ''))

status_h_style = (status_h.style
                  .applymap(lambda x: 'background-color : %s' % 'red' if x == 'Non-stationarity' else '')
                  .applymap(lambda x: 'background-color : %s' % 'green' if x == 'Inconclusive with KPSS Stationary' else '')
                  .applymap(lambda x: 'background-color : %s' % 'blue' if x == 'Inconclusive with KPSS non-Stationary' else ''))

status_n_style = (status_n.style
                  .applymap(lambda x: 'background-color : %s' % 'red' if x == 'Non-stationarity' else '')
                  .applymap(lambda x: 'background-color : %s' % 'green' if x == 'Inconclusive with KPSS Stationary' else '')
                  .applymap(lambda x: 'background-color : %s' % 'blue' if x == 'Inconclusive with KPSS non-Stationary' else ''))

filepath_status = path_excel+'/'+'stationarity_comparation.xlsx'
writer = pd.ExcelWriter(filepath_status, engine='xlsxwriter')

status_b_style.to_excel(writer, sheet_name='Baseline', engine='xlsxwriter')
status_f_style.to_excel(writer, sheet_name='Fear', engine='xlsxwriter')
status_h_style.to_excel(writer, sheet_name='Happy', engine='xlsxwriter')
status_n_style.to_excel(writer, sheet_name='Neutral', engine='xlsxwriter')
writer.save()

# NOTA: correu pela última vez no dia 06/01/2022 à 21:00 (este é o documento original)


# ### Features Extraction

# In[28]:


def all_features(dic_features, sampling_rate):

    lista = []

    for participant in dic_features.keys():

        # Features
        features_emg_mf = feature_extraction(
            dic_features[participant], 'EMG_MF', sampling_rate)
        features_emg_tr = feature_extraction(
            dic_features[participant], 'EMG_TR', sampling_rate)
        features_eda = feature_extraction(
            dic_features[participant], 'EDA', sampling_rate)
        features_ecg = feature_extraction(
            dic_features[participant], 'ECG', sampling_rate)

        part = pd.DataFrame({'ID participant': [participant]})
        features_ = pd.concat(
            [features_emg_mf, features_emg_tr, features_eda, features_ecg], axis=1)
        lista.append(features_)

    return lista


# In[29]:


b_features = all_features(dic_baseline, sampling_rate)
b_df = pd.DataFrame()
baseline_features = b_df.append(other=b_features, ignore_index=True)


# In[30]:


f_features = all_features(dic_fear, sampling_rate)
f_df = pd.DataFrame()
fear_features = f_df.append(other=f_features, ignore_index=True)


# In[31]:


h_features = all_features(dic_happy, sampling_rate)
h_df = pd.DataFrame()
happy_features = h_df.append(other=h_features, ignore_index=True)


# In[32]:


n_features = all_features(dic_neutral, sampling_rate)
n_df = pd.DataFrame()
neutral_features = n_df.append(other=n_features, ignore_index=True)


# In[33]:


filepath_feat = path_excel+'/'+'conditions_features.xlsx'
writer = pd.ExcelWriter(filepath_feat, engine='xlsxwriter')
baseline_features.to_excel(writer, sheet_name='Baseline', na_rep='nan')
fear_features.to_excel(writer, sheet_name='Fear', na_rep='nan')
happy_features.to_excel(writer, sheet_name='Happy', na_rep='nan')
neutral_features.to_excel(writer, sheet_name='Neutral', na_rep='nan')
writer.save()


# NOTA: correu pela última vez no dia 24/02/2021 à 01:02 (este é o documento original)

# ### Acessing features extraction dataframe

# In[34]:


filepath_feat = path_excel+'/'+'conditions_features.xlsx'


# In[35]:


baseline = pd.read_excel(filepath_feat, sheet_name='Baseline',
                         usecols=lambda x: 'Unnamed' not in x)
fear = pd.read_excel(filepath_feat, sheet_name='Fear',
                     usecols=lambda x: 'Unnamed' not in x)
happy = pd.read_excel(filepath_feat, sheet_name='Happy',
                      usecols=lambda x: 'Unnamed' not in x)
neutral = pd.read_excel(filepath_feat, sheet_name='Neutral',
                        usecols=lambda x: 'Unnamed' not in x)


# ### Dividing signals in excerpts

# #### Spliting signals into small series

# In[36]:


dict_data = pickle.load(open('dict_data_file', 'rb'))
dict_emotion = pickle.load(open('dict_emotion_file', 'rb'))
sampling_rate = 1000


# In[37]:


dict_split = dict_emotion


# In[38]:


def spliting_signals(signal, sampling_rate):

    size = len(signal)

    splits = {}

    increment = round(size/5)

    for i in range(0, 5):
        st = i * increment
        en = (i+1)*increment
        small_signal = signal[st:en]
        splits[i+1] = small_signal

    return splits


# In[ 39]:


def spliting_emotion(Emotion, dict_split, sampling_rate):

    for part in dict_split[Emotion][0]:
        signal = dict_split[Emotion][0][part]

        splits = spliting_signals(signal, sampling_rate)
        dict_split[Emotion][0][part] = splits

    return dict_split

# In[]:
def splinting_data(condition, clust, data):
    dict_split={}
    
    for part in clust: 
        dict_split[condition][part]=[]
        
        for i in range(len(clust)):
            dict_split[condition][part].append(data[condition][0][clust[0][i][0]][clust[0][i][1]])
        
    return dict_split

# 4 linhas a seguir em #

# In[40]:
dict_split = spliting_emotion('Fear', dict_split, sampling_rate)
#dict_split = spliting_data('Fear',name_clust_pos_b ,dict_emotion)

# In[41]:


dict_split = spliting_emotion('Happy', dict_split, sampling_rate)


# In[42]:


dict_split = spliting_emotion('Neutral', dict_split, sampling_rate)


# In[43]:


pickle.dump(dict_split, open("dict_split", "wb"))


# #### Acessing splited data

# In[44]:


dict_split = pickle.load(open("dict_split", "rb"))
sampling_rate = 1000



# #### Preprocesing signals

# In[ 45]:


def all_preprocessed_features(dict_emotion, condition, sampling_rate):

    dic_features = {}

    for part in dict_emotion[condition][0].keys():
        print(part)
        dic_features[part] = {}

        for split in dict_emotion[condition][0][part].keys():

            # if (condition=='Fear') & (part=='C007') & (split==5):
            #    a = 118629
            #     b = len(dict_emotion[condition][0][part][split][:,1])

            # elif (condition=='Fear') & (part=='C011') & (split==1):
            #     a=118534
            #     b = len(dict_emotion[condition][0][part][split][:,1])

            # elif (condition=='Happy') & (part=='C008') & (split==4):
            #     a=121243
            #     b = len(dict_emotion[condition][0][part][split][:,1])

            # elif (condition=='Happy') & (part=='C026') & (split==5):
            #     a=119540
            #     b = len(dict_emotion[condition][0][part][split][:,1])

            # elif (condition=='Neutral') & (part=='C010') & (split==2):
            #     a=len(dict_emotion[condition][0][part][split][:,0])
            #     b=131074

            # elif (condition=='Neutral') & (part=='C014') & (split==5):
            #     a=132533
            #     b = len(dict_emotion[condition][0][part][split][:,1])

            # else:
            #     a = len(dict_emotion[condition][0][part][split][:,0])
            #     b = len(dict_emotion[condition][0][part][split][:,1])

            a = len(dict_emotion[condition][0][part][split][:, 0])
            b = len(dict_emotion[condition][0][part][split][:, 1])
            # print(split)

            # Features
            MF_duration_activity, MF_peak_activity, MF_mean_activity, MF_area_activity, MF_amplitude_activity = signal_process(
                dict_emotion[condition][0][part][split][:, 0], 'EMG', sampling_rate, a)
            TR_duration_activity, TR_peak_activity, TR_mean_activity, TR_area_activity, TR_amplitude_activity = signal_process(
                dict_emotion[condition][0][part][split][:, 1], 'EMG', sampling_rate, b)
            eda_symp, scr_height, scr_amplitude, scr_risetime, scr_recoverytime, eda_tonic, eda_phasic = signal_process(
                dict_emotion[condition][0][part][split][:, 2], 'EDA', sampling_rate, len(dict_emotion[condition][0][part][split][:, 2]))
            ecg_Rpeaks, ecg_rate, t_duration = signal_process(
                dict_emotion[condition][0][part][split][:, 3], 'ECG', sampling_rate, len(dict_emotion[condition][0][part][split][:, 3]))

            dic_features[part][split] = {'MF_duration_activity': MF_duration_activity, 'MF_peak_activity': MF_peak_activity,
                                         'MF_mean_activity': MF_mean_activity, 'MF_area_activity': MF_area_activity,
                                         'MF_amplitude_activity': MF_amplitude_activity,
                                         'TR_duration_activity': TR_duration_activity, 'TR_peak_activity': TR_peak_activity,
                                         'TR_mean_activity': TR_mean_activity, 'TR_area_activity': TR_area_activity,
                                         'TR_amplitude_activity': TR_amplitude_activity,
                                         'scr_height': scr_height, 'scr_amplitude': scr_amplitude, 'scr_risetime': scr_risetime,
                                         'scr_recoverytime': scr_recoverytime, 'eda_tonic': eda_tonic, 'eda_phasic': eda_phasic,
                                         'ecg_rate': ecg_rate, 't_duration': t_duration,
                                         'eda_symp': eda_symp, 'ecg_Rpeaks': ecg_Rpeaks}

    return dic_features


# * Correr durante a noite

# In[46 ]:


dict_split_fear = all_preprocessed_features(dict_split, 'Fear', sampling_rate)


# In[47 ]:


dict_split_happy = all_preprocessed_features(
    dict_split, 'Happy', sampling_rate)


# In[48 ]:


dict_split_neutral = all_preprocessed_features(
    dict_split, 'Neutral', sampling_rate)


# comentario

# In[49]:


# filename = 'dict_split_fear'
# file=open(path_excel+'/'+filename,'wb')
# pickle.dump(dict_split_fear,file)
# file.close()

# filename = 'dict_split_happy'
# file = open(path_excel+'/'+filename, 'wb')
# pickle.dump(dict_split_happy, file)
# file.close()

# filename = 'dict_split_neutral'
# file=open(path_excel+'/'+filename,'wb')
# pickle.dump(dict_split_neutral,file)
# file.close()


# #### Acessing preprocessed splited data

# In[50]:


#dict_split_fear = pickle.load(open(path_excel+'/'+'dict_split_fear', 'rb'))
#dict_split_happy = pickle.load(open(path_excel+'/'+'dict_split_happy', 'rb'))
#dict_split_neutral = pickle.load(open(path_excel+'/'+'dict_split_neutral', 'rb'))


# #### Feature Extraction

# In[51]:


def all_features(dic_features, sampling_rate):
    lista = []

    for participant in dic_features.keys():

        for split in dic_features[participant].keys():

            # Features
            features_emg_mf = feature_extraction(
                dic_features[participant][split], 'EMG_MF', sampling_rate)
            features_emg_tr = feature_extraction(
                dic_features[participant][split], 'EMG_TR', sampling_rate)
            features_eda = feature_extraction(
                dic_features[participant][split], 'EDA', sampling_rate)
            features_ecg = feature_extraction(
                dic_features[participant][split], 'ECG', sampling_rate)

            part = pd.DataFrame(
                {'ID participant': [participant], 'Excerpt': [split]})
            features_ = pd.concat(
                [part, features_emg_mf, features_emg_tr, features_eda, features_ecg], axis=1)
            lista.append(features_)

    return lista


# estas tbm em comentario menos os fear_features_df e iguais

# In[52]:


fear_features = all_features(dict_split_fear, sampling_rate)
fear_df = pd.DataFrame()
fear_features_df = fear_df.append(other=fear_features, ignore_index=True)


# In[53]:


fear_features_df


# In[54]:


happy_features = all_features(dict_split_happy, sampling_rate)
happy_df = pd.DataFrame()
happy_features_df = happy_df.append(other=happy_features, ignore_index=True)


# In[55]:


happy_features_df


# In[56]:


neutral_features = all_features(dict_split_neutral, sampling_rate)
neutral_df = pd.DataFrame()
neutral_features_df = neutral_df.append(
    other=neutral_features, ignore_index=True)


# In[57]:


neutral_features_df


# In[58]:


filepath_feat = path_excel+'/'+'conditions_features_excerpts.xlsx'
writer = pd.ExcelWriter(filepath_feat, engine='xlsxwriter')
#fear_features_df.to_excel(writer, sheet_name='Fear', na_rep='nan')
happy_features_df.to_excel(writer, sheet_name='Happy', na_rep='nan')
#neutral_features_df.to_excel(writer, sheet_name='Neutral', na_rep='nan')
writer.save()



# In[]:
# ### Feature Selection

dict_emotion = pickle.load(open('dict_emotion_file', 'rb'))

[clust_pos_b, clust_neg_b, slope_pos_b, slope_neg_b, name_clust_pos_b, name_clust_neg_b, name_clust_null_b] = pickle.load(open(path_excel+'/'+'clust_baseline_file', 'rb'))
[clust_pos_f, clust_neg_f, slope_pos_f, slope_neg_f, name_clust_pos_f, name_clust_neg_f, name_clust_null_f] = pickle.load(open(path_excel+'/'+'clust_fear_file', 'rb'))
[clust_pos_h, clust_neg_h, slope_pos_h, slope_neg_h, name_clust_pos_h, name_clust_neg_h, name_clust_null_h] = pickle.load(open(path_excel+'/'+'clust_happy_file', 'rb'))
[clust_pos_n, clust_neg_n, slope_pos_n, slope_neg_n, name_clust_pos_n, name_clust_neg_n, name_clust_null_n] = pickle.load(open(path_excel+'/'+'clust_neutral_file', 'rb'))

sampling_rate = 1000

# In[? ]:
#Create the matrix for the classification model

def matrix_features(dic_emotion, clust, slope, sampling_rate, cond, cond2):

    # matrix_f=pd.DataFrame(index=range(len(clust)),columns=['id','rate_median','rate_mean','rate_mean_norm','rate_median_norm', 'diff_rate', 'diff_rate_norm','emotion'])
    # df_data=pd.DataFrame(index=range(len(mean_clust)),columns=['rate','rate_norm','diff_rate','diff_rate_norm'])
    # X=0            
    dic_features= {}
    sorted_slope = {}
    lista=pd.DataFrame()
    count=0
    
    #Sort the subclasses by the mean of the slope.
    # In the positive, the higher the slope the higher the degree of the subclass.
    if cond2 == "pos":  
        subclass=[]
        sorted_keys = sorted(slope, key=slope.get)

        for w in sorted_keys:
            sorted_slope[w] = slope[w]
            [subclass.append('+'*i) for i in range(1,len(slope.keys())+1)]
    
    # In the negative, the lower the slope the higher the degree of the subclass    
    if cond2 == "neg":
        subclass=[]
        sorted_keys = sorted(slope, key=slope.get, reverse=True)

        for w in sorted_keys:
            sorted_slope[w] = slope[w]       
            [subclass.append('-'*i) for i in range(1,len(slope.keys())+1)]
    
    
    for clust_i,clust_val in enumerate(sorted_slope.keys()):
        dic_features[clust_val] = {}
        #print(clust_val)

        for i,v in enumerate(clust[clust_val]):
            
            participant=v[1][0]
            idx_i=v[1][3]
            idx_f=v[1][4]
            t=v[1][1]
            #print([participant,idx_i, idx_f])
            
            #A case with a error in the detection
            if (participant=='C012') and (cond=='Fear') and (idx_i==74894):
                continue
                
            ecg_Rpeaks, ecg_rate, T_duration, ST_interval, P_duration, TP_interval , PR_interval, QRS_duration, RR_Pre, RR_Pos = signal_process(
                dic_emotion[cond][0][participant][idx_i: idx_f][:, 3], 'ECG', sampling_rate, len(dic_emotion[cond][0][participant][idx_i: idx_f][:, 3]))

            dic_features[clust_val][i] = {'Participant': participant, 
                                           'Idx_i': idx_i,
                                           'ecg_Rpeaks': ecg_Rpeaks,
                                           'ecg_rate': ecg_rate, 
                                           'T_duration': T_duration,  
                                           'P_duration': P_duration, 
                                           'QRS_duration': QRS_duration,
                                           'ST_interval': ST_interval, 
                                           'TP_segment': TP_interval,
                                           'PR_interval': PR_interval,
                                           'RR_Pre': RR_Pre,
                                           'RR_Pos': RR_Pos
                                           } 
            
            #Detect moments with one feature with all elements nan and eleminate them
            for v in set(dic_features[clust_val][i]) - set(['ecg_Rpeaks', 'Participant']):
                if np.isnan(dic_features[clust_val][i][v]).all():   
                    del dic_features[clust_val][i]
                    count=count+1
                    break
           
            #Extract statitical features
            else:
                features_ecg = feature_extraction_ecg (
                    dic_features[clust_val][i], sampling_rate)
                
                part = pd.DataFrame(
                    {'Emotion': [cond], 
                     'Class': [cond2],
                      'Subclass': [subclass[clust_i]],
                      'ID participant': [participant],
                      'Idx_i': [idx_i]
                      }) 
                
                features_ = pd.concat(
                    [part, features_ecg], axis=1)
                
                lista=pd.concat([lista, features_],ignore_index=True)
                continue
            
    return dic_features, lista, count

# In[]: 
#Same procedure as before for the null subclass of emotion

def matrix_features_null (dic_emotion, name_clust, sampling_rate, cond, cond2):
    dic_features= {}
    sorted_slope = {}
    lista=pd.DataFrame()
    count=0
    
    if cond2 == "null":
        for i in name_clust:
            participant=i[0]
            idx_i=i[3]
            idx_f=i[4]
            t=i[1]
            
            # if np.isnan(dic_emotion[cond][0][participant][idx_i: idx_f][:, 3]).any():
            #print([participant,idx_i, idx_f])
            ecg_Rpeaks, ecg_rate, T_duration, ST_interval, P_duration, TP_interval , PR_interval, QRS_duration, RR_Pre, RR_Pos = signal_process(
                dic_emotion[cond][0][participant][idx_i: idx_f][:, 3], 'ECG', sampling_rate, len(dic_emotion[cond][0][participant][idx_i: idx_f][:, 3]))

            dic_features[t] = {'Participant': participant, 
                                'Idx_i': idx_i,
                                'ecg_Rpeaks': ecg_Rpeaks,
                                'ecg_rate': ecg_rate, 
                                'T_duration': T_duration,  
                                'P_duration': P_duration, 
                                'QRS_duration': QRS_duration,
                                'ST_interval': ST_interval, 
                                'TP_segment': TP_interval,
                                'PR_interval': PR_interval,
                                'RR_Pre': RR_Pre,
                                'RR_Pos': RR_Pos
                                } 
            
           # print(dic_features[t])          
            for v in set(dic_features[t]) - set(['ecg_Rpeaks', 'Participant']):

                #if ((v != 'ecg_Rpeaks') and (v != 'Participant')):              
                if np.isnan(dic_features[t][v]).all():
                #    print(dic_features[t][v])
                    del dic_features[t]
                    count=count+1
                    break
            else:
                features_ecg = feature_extraction_ecg (
                    dic_features[t], sampling_rate)
                
                part = pd.DataFrame(
                    {'Emotion': [cond], 
                      'Class': [cond2],
                      'Subclass': ['0'],
                      'ID participant': [participant],
                      'Idx_i': [idx_i]
                      }) 
                
                features_ = pd.concat(
                    [part, features_ecg], axis=1)
                
                lista=pd.concat([lista, features_],ignore_index=True)
                continue

    return dic_features, lista  , count    
            
# In[ ]:
# Features for baseline
dic_features_pos_b, lista_pos_b, count_pos_b = matrix_features (dict_emotion, clust_pos_b, slope_pos_b, sampling_rate, "Baseline", "pos")
dic_features_neg_b, lista_neg_b, count_neg_b = matrix_features (dict_emotion, clust_neg_b, slope_neg_b, sampling_rate, "Baseline", "neg")
dic_features_null_b, lista_null_b, count_null_b = matrix_features_null (dict_emotion, name_clust_null_b, sampling_rate, "Baseline", "null")

# In[]: 
# Features for fear    
dic_features_pos_f, lista_pos_f, count_pos_f = matrix_features (dict_emotion, clust_pos_f, slope_pos_f, sampling_rate, "Fear", "pos")
dic_features_neg_f, lista_neg_f, count_neg_f = matrix_features (dict_emotion, clust_neg_f, slope_neg_f, sampling_rate, "Fear", "neg")
dic_features_null_f, lista_null_f, count_null_f = matrix_features_null (dict_emotion, name_clust_null_f, sampling_rate, "Fear", "null")

# In[]: 
# Features for neutral   
dic_features_pos_n, lista_pos_n, count_pos_n = matrix_features (dict_emotion, clust_pos_n, slope_pos_n, sampling_rate, "Neutral", "pos")
dic_features_neg_n, lista_neg_n, count_neg_n = matrix_features (dict_emotion, clust_neg_n, slope_neg_n, sampling_rate, "Neutral", "neg")
dic_features_null_n, lista_null_n, count_null_n = matrix_features_null (dict_emotion, name_clust_null_n, sampling_rate, "Neutral", "null")

# In[]: 
# Features for happy    
dic_features_pos_h, lista_pos_h, count_pos_h = matrix_features (dict_emotion, clust_pos_h, slope_pos_h, sampling_rate, "Happy", "pos")
dic_features_neg_h, lista_neg_h, count_neg_h = matrix_features (dict_emotion, clust_neg_h, slope_neg_h, sampling_rate, "Happy", "neg")
dic_features_null_h, lista_null_h, count_null_h = matrix_features_null (dict_emotion, name_clust_null_h, sampling_rate, "Happy", "null")

# In[]:
# Save features

fear_features = [lista_pos_f, lista_neg_f, lista_null_f]
filename = 'features_f_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(fear_features, file)
file.close()

# In[]:
b_features = [lista_pos_b, lista_neg_b, lista_null_b]
filename = 'features_b_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(b_features, file)
file.close()

# In[]:
h_features = [lista_pos_h, lista_neg_h, lista_null_h]
filename = 'features_h_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(h_features, file)
file.close()

# In[]:
n_features = [lista_pos_n, lista_neg_n, lista_null_n]
filename = 'features_n_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(n_features, file)
file.close()
 # In[]:
    
lista_pos_f, lista_neg_f, lista_null_f = pickle.load(open("features_f_file", "rb"))
lista_pos_b, lista_neg_b, lista_null_b = pickle.load(open("features_b_file", "rb"))
lista_pos_n, lista_neg_n, lista_null_n = pickle.load(open("features_n_file", "rb"))
lista_pos_h, lista_neg_h, lista_null_h = pickle.load(open("features_h_file", "rb"))



# In[]:
def color_line(condition):

    if condition == "Fear":
        line_c= 'rgb(34,154,0)'
        line_b='green'
        
    elif condition == "Happy":
        line_c= 'rgb(255, 174, 66)'
        line_b='orange'
        
    elif condition == "Neutral":
        line_c= 'rgb(119,136,153)'
        line_b='darkslategrey'
        
    elif condition == "Baseline":
        line_c= 'rgb(47,79,79)'
        line_b='black'
        
    return line_c, line_b

# In[]:
def boxplot(data, condition, j):
    line_c, line_b = color_line(condition)
    
    if j == 0:
        cond2= 'pos'
        
    if j == 1:
        cond2= 'neg'
        
    if j == 2:
        cond2= '0'
        
    for i in set(dados[j]) - set(['Class', 'ID participant', 'Emotion', 'Idx_i', 'Subclass']):
        fig=go.Figure()
        fig.add_trace(go.Box(x=dados['Subclass'], y=dados[i],
                        line=dict(color=line_b),
                        fillcolor=line_c)
                        )
        fig.update_layout(xaxis_title='Emotion',
                    yaxis_title=str(i)+' value',
                    font=dict(
                        family="Times New Roman", size=30),
                    width=1000,
                    height=1000)
        fig.show()
        
        plotly.offline.plot(
            fig, filename=r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Boxplot\Boxplot_"+str(condition)+"_"+str(cond2)+"_"+str(i)+".html", auto_open=False)

def boxplot_subclass(dados, condition):
    line_c, line_b= color_line(condition)
    for i in set(dados) - set(['Class', 'ID participant', 'Emotion', 'Idx_i', 'Subclass']):
        fig=go.Figure()
        fig.add_trace(go.Box(x=dados['Subclass'], y=dados[i],
                        line=dict(color=line_b),
                        fillcolor=line_c)
                        )
        fig.update_layout(xaxis_title='Emotion',
                    yaxis_title=str(i)+' value',
                    font=dict(
                        family="Times New Roman", size=30),
                    width=1000,
                    height=1000)
        fig.update_yaxes(range=[min(dados[i]), max(dados[i])])
        fig.show()
        
        plotly.offline.plot(
            fig, filename=r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Boxplot\Boxplot_"+str(condition)+"_subclass_"+str(i)+".html", auto_open=False)



def boxplot_class(dados, condition):
    line_c, line_b= color_line(condition)     
    for i in set(dados) - set(['Class', 'ID participant', 'Emotion', 'Idx_i','Subclass']):
        # fig = px.box(dados, x='Class', y=i, 
        #              line=dict(color='black'),
        #              fillcolor=dict(color=line_c))
        fig=go.Figure()
        fig.add_trace(go.Box(x=dados['Class'], y=dados[i],
                        line=dict(color=line_b),
                        fillcolor=line_c)
                        )
        
        fig.update_layout(xaxis_title='Emotion',
                    yaxis_title=str(i)+' value',
                    font=dict(
                        family="Times New Roman", size=30),
                    width=1000,
                    height=1000)
        fig.update_yaxes(range=[min(dados[i]), max(dados[i])])
        fig.show()
        
        plotly.offline.plot(
            fig, filename=r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Boxplot\Boxplot_"+str(condition)+"_class_"+str(i)+".html", auto_open=False)


def boxplot_norm(dados, dados_range, condition):
    line_c, line_b= color_line(condition)
    for i in set(dados) - set(['Class', 'ID participant', 'Emotion', 'Idx_i','Subclass', 'Label']):
        fig=go.Figure()
        fig.add_trace(go.Box(x=dados['Subclass'], y=dados[i],
                        line=dict(color=line_b),
                        fillcolor=line_c)
                        )
        fig.update_layout(xaxis_title='Emotion',
                    yaxis_title=str(i)+' value',
                    font=dict(
                        family="Times New Roman", size=30),
                    width=1000,
                    height=1000)
        print(max(dados_range[i]))
        fig.update_yaxes(range=[min(dados_range[i]), max(dados_range[i])])
        fig.show()
        
        plotly.offline.plot(
            fig, filename=r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Boxplot\Boxplot_"+str(condition)+"_norm_"+str(i)+".html", auto_open=False)


#boxplot_emotion(data)
# In[]:
#lista_pos_h, lista_neg_h, lista_pos_f, lista_neg_f, lista_pos_n, lista_neg_n] #,lista_pos_b, lista_neg_b, lista_null_b,lista_pos_n, lista_neg_n, lista_null_n, lista_pos_h, lista_neg_h, lista_null_h]



# In[]:
def dados(lista,j):
    
    ext=[]
    metrics_train={}
    metrics_test={}
    
    test_metrics= pd.DataFrame()
    train_metrics = pd.DataFrame()
    
    #Dected nulls and fill with mean value 
    for j in lista:  
        #ext.append(j['Subclass'].unique()[-1])
    
        for i in j.columns:
            if j[i].isnull().values.any():
                j[i] = j[i].fillna(value=j[i].mean())
    
    data = pd.concat(lista, ignore_index=True)
    #Randomize positions of the observations
    
    #Dataset only with wave interval
    data_2= pd.DataFrame()
    data_2=data[['Emotion', 'Class', 'Subclass','ID participant', 'Idx_i','ECG_Rate','QRS_duration', 'P_duration', 'T_duration', 'ST_interval', 'TP_segment', 'PR_interval']].copy()
    
    #Dataset with wave interval divided by the mean of the array
    l = ['ECG_Rate','QRS_duration', 'P_duration', 'T_duration', 'ST_interval', 'TP_segment', 'PR_interval']
    data_3= pd.DataFrame()
    data_3=data[['Emotion', 'Class', 'Subclass','ID participant', 'Idx_i']].copy()
    for keyo in l:
        data_3[str('Result_'+ keyo)] = data[keyo]/data[str(keyo+'_Mean')]
    
    #Dataset with waveintervall multiply by heart rate
    j = ['QRS_duration', 'P_duration', 'T_duration', 'ST_interval', 'TP_segment', 'PR_interval']
    data_4= pd.DataFrame()
    data_4=data[['Emotion', 'Class', 'Subclass','ID participant', 'Idx_i', 'ECG_Rate']].copy()
    for keyo in j:
        data_4[str('Result_'+keyo)] = data[keyo] * data['ECG_Rate']
    
    #Dataset with extremes 
    lst=[]
    for index, row in data_2.iterrows():
        if (row['Subclass'] == '++++++') and (row['Emotion'] == 'Happy'):
            lst.append(row)
        if (row['Subclass'] == '------') and (row['Emotion'] == 'Happy'):
            lst.append(row)
        if (row['Subclass'] == '++++++') and (row['Emotion'] == 'Fear'):
                lst.append(row)
        if (row['Subclass'] == '-------') and (row['Emotion'] == 'Fear'):
                lst.append(row)
        if (row['Subclass'] == '++++++') and (row['Emotion'] == 'Neutral'):
                lst.append(row)
        if (row['Subclass'] == '-------') and (row['Emotion'] == 'Neutral'):         
                lst.append(row)
        if (row['Subclass'] == '+++++++') and (row['Emotion'] == 'Neutral'):         
                    lst.append(row)
    
    data_5 = pd.DataFrame(lst, columns=data_2.columns)
    data_5['Label']=data_5['Subclass']+data_5['Emotion']
    
    return data_2, data_5
# In[]:
lista=[lista_pos_f, lista_neg_f, lista_null_f]
data=pd.concat(lista)
condition='Fear'
j=0

data_2, data_5 = dados (lista,j)

boxplot_class(data_2, 'Fear')
boxplot_subclass(data_2, 'Fear')
boxplot_norm(data_5, data_2, 'Fear')

# In[Happy]

lista=[lista_pos_h, lista_neg_h, lista_null_h]
data=pd.concat(lista)
condition='Happy'
j=0

data_2, data_5 = dados(lista,j)
boxplot_class(data_2, 'Happy')
boxplot_subclass(data_2, 'Happy')
boxplot_norm(data_5, data_2, 'Happy')

# In[Neutral]

lista=[lista_pos_n, lista_neg_n, lista_null_n]
data=pd.concat(lista)
condition='Neutral'
j=0

data_2, data_5 = dados (lista,j)

boxplot_class(data_2, 'Neutral')
boxplot_subclass(data_2, 'Neutral')
boxplot_norm(data_5, data_2, 'Neutral')
# In[]
    
# y=dic_baseline['B001']['ecg_rate'][0:7]
# x2=x[0][0:7]
# markers_on = dic_baseline['B001']['ecg_rate'][1]
# plt.figure(1)
# plt.plot(x2,y,'-', color='black')
# plt.plot(x2,y,'o', color='blue')

# plt.xlabel("sample")
# plt.ylabel("heart rate value")
# plt.show()



#pos
def aaa(lista):
    del lista['Class']
    #lista.rename(columns = {'Sublass':'Class'}, inplace = True) 
    classes=['Positive']*len(lista)    
    lista.insert(1, "Class", classes)     
    return lista

lista_pos_b=aaa(lista_pos_b)

