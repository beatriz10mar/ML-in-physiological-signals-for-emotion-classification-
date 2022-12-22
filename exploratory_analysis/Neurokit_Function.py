#!/usr/bin/env python
# coding: utf-8

# In[3]:


import neurokit2 as nk
import pandas as pd
from scipy.stats import skew, kurtosis
import numpy as np


# In[4]:


import pickle
dict_emotion = pickle.load(open('dict_emotion_file', 'rb'))


# In[7]:


sampling_rate=1000


# In[11]:
# Meti as linhas a baio em comentario

#signal=dict_emotion['Happy'][0]['C026'][:,0]


# In[12]:


#clean = nk.emg_clean(signal,sampling_rate=sampling_rate)
#amplitude = nk.emg_amplitude(clean)


# In[13]:


#len(clean)


# In[42]:


#activity, info_act = nk.emg_activation(emg_cleaned=clean[0:606905],sampling_rate=sampling_rate,method='biosppy')


# ### Pre-processing

# In[9]:


def signal_process (signal, label, sampling_rate, a):
    
    if label == 'EMG':
        clean = nk.emg_clean(signal,sampling_rate=sampling_rate)
        amplitude = nk.emg_amplitude(clean)
        activity, info_act = nk.emg_activation(emg_cleaned=clean[0:a],sampling_rate=sampling_rate,method='biosppy')
        
        # Duration of Activity, Max peak of activity, Mean peaks of activity
        duration_activity=[]
        peak_activity=[]
        mean_activity=[]
        area_activity=[]
        amplitude_activity=[]
        
        for i in range(0,len(info_act['EMG_Offsets'])):
            duration=(info_act['EMG_Offsets'][i]-info_act['EMG_Onsets'][i])/sampling_rate # in seconds
            peak=max(amplitude[info_act['EMG_Onsets'][i]:info_act['EMG_Offsets'][i]])
            mean=np.mean(amplitude[info_act['EMG_Onsets'][i]:info_act['EMG_Offsets'][i]])
            area=np.trapz(amplitude[info_act['EMG_Onsets'][i]:info_act['EMG_Offsets'][i]])            
            duration_activity.append(duration) #The duration of the muscle activity
            peak_activity.append(peak) #The maximum peak of the muscle activity
            mean_activity.append(mean) #The mean of peaks of the muscle activity
            area_activity.append(area) #The area of muscle activity
            amplitude_activity.append(amplitude[info_act['EMG_Onsets'][i]:info_act['EMG_Offsets'][i]]) # redundante
        
       
        amplitude_activity = [item for sublist in amplitude_activity for item in sublist]
    
        return duration_activity, peak_activity, mean_activity, area_activity, amplitude_activity
                    
    elif label == 'EDA':
        eda_feat, info_eda=nk.eda_process(eda_signal=signal, sampling_rate=sampling_rate)
        eda_symp = nk.eda_sympathetic(eda_feat['EDA_Raw'],sampling_rate=sampling_rate,frequency_band=[0.045,0.25],method='posada')
        #raw_signal=eda_feat['EDA_Raw']
        eda_tonic = eda_feat['EDA_Tonic'] # The tonic component of eda
        eda_phasic = eda_feat['EDA_Phasic'] # The phasic component of eda
        scr_height=info_eda['SCR_Height'] #The SCR amplitude including tonic component
        scr_amplitude=info_eda['SCR_Amplitude'] #The SCR amplitude excluding tonic component
        scr_risetime=info_eda['SCR_RiseTime'] #The time is taken for SCR onset to reach peak amplitude
        scr_recoverytime=info_eda['SCR_RecoveryTime'] #The time it takes for SCR to decrease to half amplitude.
        
        return eda_symp, scr_height, scr_amplitude, scr_risetime, scr_recoverytime, eda_tonic, eda_phasic
        
    elif label == 'ECG':
        if len(signal) > 4500:
            ecg_clean = nk.ecg_clean(signal,sampling_rate=sampling_rate)  
            ecg_Rpeaks, info_Rpeaks = nk.ecg_peaks(ecg_clean,sampling_rate=sampling_rate) #signal to ecg_clean
    
            if len(info_Rpeaks['ECG_R_Peaks']) >= 4 :
    
                #peaks_cwt=nk.ecg_delineate(ecg_clean, info_peaks['ECG_R_Peaks'], sampling_rate, method='cwt')
                #peaks_dwt=nk.ecg_delineate(ecg_clean, info_peaks['ECG_R_Peaks'], sampling_rate, method='dwt')
                ecg_peaks, info_peaks = nk.ecg_delineate(ecg_clean, info_Rpeaks['ECG_R_Peaks'], sampling_rate)
                ecg_rate = nk.ecg_rate(ecg_Rpeaks, sampling_rate=sampling_rate, interpolation_method='monotone_cubic')
                
                T_duration=[]
                ST_interval=[]
                P_duration=[]
                PR_interval=[]
                TP_interval=[]
                QRS_duration=[]
                
                RR_Pre = info_Rpeaks['ECG_R_Peaks'][1]-info_Rpeaks['ECG_R_Peaks'][0]
                RR_Pos = info_Rpeaks['ECG_R_Peaks'][2]-info_Rpeaks['ECG_R_Peaks'][1]
                
                for i in range(0,len(info_Rpeaks['ECG_R_Peaks'])-1):
                    duration_t=(info_peaks['ECG_T_Offsets'][i]-info_peaks['ECG_T_Onsets'][i])/1000 # in seconds
                    T_duration.append(duration_t) #The duration between the T peak and the offset
                    
                    st_interval=(info_peaks['ECG_T_Offsets'][i]-info_peaks['ECG_R_Offsets'][i])/1000 # in seconds
                    ST_interval.append(st_interval) #The duration between the T peak and the offset         
                         
                    duration_p=(info_peaks['ECG_P_Offsets'][i]-info_peaks['ECG_P_Onsets'][i])/1000 # in seconds
                    P_duration.append(duration_p) #The duration between the P peak and the offset
                    
                    pr_interval=(info_peaks['ECG_R_Onsets'][i]-info_peaks['ECG_P_Onsets'][i])/1000 # in seconds
                    PR_interval.append(pr_interval) #The duration between the P peak and the offset
                    
                    duration_qrs=(info_peaks['ECG_R_Offsets'][i]-info_peaks['ECG_R_Onsets'][i])/1000 # in seconds
                    QRS_duration.append(duration_qrs) #The duration between the T peak and the offset
                   
                    tp_interval=(info_peaks['ECG_P_Onsets'][i+1]-info_peaks['ECG_T_Offsets'][i])/1000 # in seconds
                    TP_interval.append(tp_interval)
                    
            else:
                ecg_rate = np.nan
                T_duration = np.nan
                ST_interval = np.nan
                P_duration = np.nan
                TP_interval = np.nan
                PR_interval = np.nan
                QRS_duration = np.nan
                RR_Pre = np.nan
                RR_Pos = np.nan
                
                print("Error message: The lenght of the signal does not correspond to the expected, probably due to lack of data.")

                            
        else:
            ecg_Rpeaks = np.nan
            ecg_rate = np.nan
            T_duration = np.nan
            ST_interval = np.nan
            P_duration = np.nan
            TP_interval = np.nan
            PR_interval = np.nan
            QRS_duration = np.nan
            RR_Pre = np.nan
            RR_Pos = np.nan
            
        return ecg_Rpeaks, ecg_rate, T_duration, ST_interval, P_duration, TP_interval , PR_interval, QRS_duration, RR_Pre, RR_Pos



# ### Feature Extraction

# In[47]:


def feature_extraction (dic, label, sampling_rate):
    
    if label == 'EMG_MF':
        mean_dur, standard_deviation_dur, variance_dur, skewness_dur, kurtis_dur = statistics_f (dic['MF_duration_activity'])
        mean_peak, standard_deviation_peak, variance_peak, skewness_peak, kurtis_peak = statistics_f (dic['MF_peak_activity'])
        mean_act, standard_deviation_act, variance_act, skewness_act, kurtis_act = statistics_f (dic['MF_mean_activity'])
        mean_amp, standard_deviation_amp, variance_amp, skewness_amp, kurtis_amp = statistics_f (dic['MF_amplitude_activity'])
        mean_area, standard_deviation_area, variance_area, skewness_area, kurtis_area = statistics_f (dic['MF_area_activity'])
        
        EMG_Activations_N=len(dic['MF_duration_activity'])
                
        features_emg=pd.DataFrame({'EMG_MF_Activations_N':[EMG_Activations_N],
                                   'EMG_MF_Duration_Mean':[mean_dur],'EMG_MF_Duration_Std':[standard_deviation_dur],
                                   'EMG_MF_Duration_Var':[variance_dur],'EMG_MF_Duration_Skew':[skewness_dur],
                                   'EMG_MF_Duration_Kurt':[kurtis_dur],
                                   'EMG_MF_MaxPeakAct_Mean':[mean_peak],'EMG_MF_MaxPeakAct_Std':[standard_deviation_peak],
                                   'EMG_MF_MaxPeakAct_Var':[variance_peak],'EMG_MF_MaxPeakAct_Skew':[skewness_peak],
                                   'EMG_MF_MaxPeakAct_Kurt':[kurtis_peak],
                                   'EMG_MF_MeanPeaksAct_Mean':[mean_act],'EMG_MF_MeanPeaksAct_Std':[standard_deviation_act],
                                   'EMG_MF_MeanPeaksAct_Var':[variance_act],'EMG_MF_MeanPeaksAct_Skew':[skewness_act],
                                   'EMG_MF_MeanPeaksAct_Kurt':[kurtis_act],
                                   'EMG_MF_all_Amplitude_Mean':[mean_amp],'EMG_MF_all_Amplitude_Std':[standard_deviation_amp],
                                   'EMG_MF_all_Amplitude_Var':[variance_amp],'EMG_MF_all_Amplitude_Skew':[skewness_amp],
                                   'EMG_MF_all_Amplitude_Kurt':[kurtis_amp],
                                   'EMG_MF_Area_Mean':[mean_amp],'EMG_MF_Area_Std':[standard_deviation_amp],
                                   'EMG_MF_Area_Var':[variance_amp],'EMG_MF_Area_Skew':[skewness_amp],
                                   'EMG_MF_Area_Kurt':[kurtis_amp]})
        
        
        return features_emg
    
    elif label == 'EMG_TR':
        mean_dur, standard_deviation_dur, variance_dur, skewness_dur, kurtis_dur = statistics_f (dic['TR_duration_activity'])
        mean_peak, standard_deviation_peak, variance_peak, skewness_peak, kurtis_peak = statistics_f (dic['TR_peak_activity'])
        mean_act, standard_deviation_act, variance_act, skewness_act, kurtis_act = statistics_f (dic['TR_mean_activity'])
        mean_amp, standard_deviation_amp, variance_amp, skewness_amp, kurtis_amp = statistics_f (dic['TR_amplitude_activity'])
        mean_area, standard_deviation_area, variance_area, skewness_area, kurtis_area = statistics_f (dic['TR_area_activity'])
        
        EMG_Activations_N=len(dic['TR_duration_activity'])
                
        features_emg=pd.DataFrame({'EMG_TR_Activations_N':[EMG_Activations_N],
                                   'EMG_TR_Duration_Mean':[mean_dur],'EMG_TR_Duration_Std':[standard_deviation_dur],
                                   'EMG_TR_Duration_Var':[variance_dur],'EMG_TR_Duration_Skew':[skewness_dur],
                                   'EMG_TR_Duration_Kurt':[kurtis_dur],
                                   'EMG_TR_MaxPeakAct_Mean':[mean_peak],'EMG_TR_MaxPeakAct_Std':[standard_deviation_peak],
                                   'EMG_TR_MaxPeakAct_Var':[variance_peak],'EMG_TR_MaxPeakAct_Skew':[skewness_peak],
                                   'EMG_TR_MaxPeakAct_Kurt':[kurtis_peak],
                                   'EMG_TR_MeanPeaksAct_Mean':[mean_act],'EMG_TR_MeanPeaksAct_Std':[standard_deviation_act],
                                   'EMG_TR_MeanPeaksAct_Var':[variance_act],'EMG_TR_MeanPeaksAct_Skew':[skewness_act],
                                   'EMG_TR_MeanPeaksAct_Kurt':[kurtis_act],
                                   'EMG_TR_all_Amplitude_Mean':[mean_amp],'EMG_TR_all_Amplitude_Std':[standard_deviation_amp],
                                   'EMG_TR_all_Amplitude_Var':[variance_amp],'EMG_TR_all_Amplitude_Skew':[skewness_amp],
                                   'EMG_TR_all_Amplitude_Kurt':[kurtis_amp],
                                   'EMG_TR_Area_Mean':[mean_amp],'EMG_TR_Area_Std':[standard_deviation_amp],
                                   'EMG_TR_Area_Var':[variance_amp],'EMG_TR_Area_Skew':[skewness_amp],
                                   'EMG_TR_Area_Kurt':[kurtis_amp]})
        
        return features_emg
    
    elif label=='EDA':
        symp=pd.DataFrame({'EDA_Symp':[dic['eda_symp']['EDA_Symp']],'EDA_SympN':[dic['eda_symp']['EDA_SympN']]})
        
        mean_ton, standard_deviation_ton, variance_ton, skewness_ton, kurtis_ton = statistics_f (dic['eda_tonic'])
        mean_pha, standard_deviation_pha, variance_pha, skewness_pha, kurtis_pha = statistics_f (dic['eda_phasic'])
        
        mean_hei, standard_deviation_hei, variance_hei, skewness_hei, kurtis_hei = statistics_f (dic['scr_height'])
        mean_amp, standard_deviation_amp, variance_amp, skewness_amp, kurtis_amp = statistics_f (dic['scr_amplitude'])
        mean_rise, standard_deviation_rise, variance_rise, skewness_rise, kurtis_rise = statistics_f (dic['scr_risetime'])
        mean_rec, standard_deviation_rec, variance_rec, skewness_rec, kurtis_rec = statistics_f (dic['scr_recoverytime'])
        
        SCR_Peaks_N=len(dic['scr_height'])
        
        statistics_eda=pd.DataFrame({'SCR_Peaks_N':[SCR_Peaks_N],
                                     'EDA_Tonic_Mean':[mean_ton],'EDA_Tonic_Std':[standard_deviation_ton],
                                     'EDA_Tonic_Var':[variance_ton],'EDA_Tonic_Skew':[skewness_ton],
                                     'EDA_Tonic_Kurt':[kurtis_ton],
                                     'EDA_Phasic_Mean':[mean_pha],'EDA_Phasic_Std':[standard_deviation_pha],
                                     'EDA_Phasic_Var':[variance_pha],'EDA_Phasic_Skew':[skewness_pha],
                                     'EDA_Phasic_Kurt':[kurtis_pha],
                                     'SCR_Height_Mean':[mean_hei],'SCR_Height_Std':[standard_deviation_hei],
                                     'SCR_Height_Var':[variance_hei],'SCR_Height_Skew':[skewness_hei],
                                     'SCR_Height_Kurt':[kurtis_hei],
                                     'SCR_Amplitude_Mean':[mean_amp],'SCR_Amplitude_Std':[standard_deviation_amp],
                                     'SCR_Amplitude_Var':[variance_amp],'SCR_Amplitude_Skew':[skewness_amp],
                                     'SCR_Amplitude_Kurt':[kurtis_amp],
                                     'SCR_RiseTime_Mean':[mean_rise],'SCR_RiseTime_Std':[standard_deviation_rise],
                                     'SCR_RiseTime_Var':[variance_rise],'SCR_RiseTime_Skew':[skewness_rise],
                                     'SCR_RiseTime_Kurt':[kurtis_rise],
                                     'SCR_RecoveryTime_Mean':[mean_rec],'SCR_RecoveryTime_Std':[standard_deviation_rec],
                                     'SCR_RecoveryTime_Var':[variance_rec],'SCR_RecoveryTime_Skew':[skewness_rec],
                                     'SCR_RecoveryTime_Kurt':[kurtis_rec]})
        
        features_eda=pd.concat([symp, statistics_eda],axis=1)
        
        return features_eda
    
    elif label == 'ECG':      
        hrv_feat = nk.hrv(dic['ecg_Rpeaks'], sampling_rate=sampling_rate)

        mean_rate, standard_deviation_rate, variance_rate, skewness_rate, kurtis_rate = statistics_f (dic['ecg_rate'])
        mean_dur_t, standard_deviation_dur_t, variance_dur_t, skewness_dur_t, kurtis_dur_t = statistics_f (dic['t_duration'])
        mean_dur_p, standard_deviation_dur_p, variance_dur_p, skewness_dur_p, kurtis_dur_p = statistics_f (dic['p_duration'])
        mean_dur_r, standard_deviation_dur_r, variance_dur_r, skewness_dur_r, kurtis_dur_r = statistics_f (dic['p_duration'])
        mean_dur_qrs, standard_deviation_dur_qrs, variance_dur_qrs, skewness_dur_qrs, kurtis_dur_qrs = statistics_f (dic['qrs_duration'])

        statistics_ecg=pd.DataFrame({'ECG_Rate_Mean':[mean_rate],'ECG_Rate_Std':[standard_deviation_rate],
                                     'ECG_Rate_Var':[variance_rate],'ECG_Rate_Skew':[skewness_rate],
                                     'ECG_Rate_Kurt':[kurtis_rate],
                                     'ECG_Tduration_Mean':[mean_dur_t],'ECG_Tduration_Std':[standard_deviation_dur_t],
                                     'ECG_Tduration_Var':[variance_dur_t],'ECG_Tduration_Skew':[skewness_dur_t],
                                     'ECG_Tduration_Kurt':[kurtis_dur_t],
                                     'ECG_Pduration_Mean':[mean_dur_p],'ECG_Pduration_Std':[standard_deviation_dur_p],
                                     'ECG_Pduration_Var':[variance_dur_p],'ECG_Pduration_Skew':[skewness_dur_p],
                                     'ECG_Pduration_Kurt':[kurtis_dur_p],
                                     'ECG_Rduration_Mean':[mean_dur_r],'ECG_Rduration_Std':[standard_deviation_dur_r],
                                     'ECG_Rduration_Var':[variance_dur_r],'ECG_Rduration_Skew':[skewness_dur_r],
                                     'ECG_Rduration_Kurt':[kurtis_dur_r],
                                     'ECG_QRSduration_Mean':[mean_dur_qrs],'ECG_QRSduration_Std':[standard_deviation_dur_qrs],
                                     'ECG_QRSduration_Var':[variance_dur_qrs],'ECG_QRSduration_Skew':[skewness_dur_qrs],
                                     'ECG_QRSduration_Kurt':[kurtis_dur_qrs]})
        
        features_ecg=pd.concat([statistics_ecg, hrv_feat],axis=1)
           
        return features_ecg
    


# ### Statistics

# In[18]:
def feature_extraction_ecg (dic, sampling_rate):

    mean_rate, standard_deviation_rate, variance_rate = statistics_f (dic['ecg_rate'])
    mean_dur_t, standard_deviation_dur_t, variance_dur_t = statistics_f (dic['T_duration'])
    mean_dur_p, standard_deviation_dur_p, variance_dur_p= statistics_f (dic['P_duration'])
    mean_dur_qrs, standard_deviation_dur_qrs, variance_dur_qrs = statistics_f (dic['QRS_duration'])
    mean_st, standard_deviation_st, variance_st = statistics_f (dic['ST_interval'])
    mean_tp, standard_deviation_tp, variance_tp = statistics_f (dic['TP_interval'])
    mean_pr, standard_deviation_pr, variance_pr = statistics_f (dic['PR_interval'])


    statistics_ecg=pd.DataFrame({'ECG_Rate':[dic['ecg_rate'][0]],
                                 'ECG_Rate_Mean':[mean_rate],
                                 'ECG_Rate_Std':[standard_deviation_rate],
                                 'ECG_Rate_Var':[variance_rate],
                                 
                                 'T_duration':[dic['T_duration'][0]],
                                 'T_duration_Mean':[mean_dur_t],
                                 'T_duration_Std':[standard_deviation_dur_t],
                                 'T_duration_Var':[variance_dur_t],
                                 
                                 'P_duration':[dic['P_duration'][0]],
                                 'P_duration_Mean':[mean_dur_p],
                                 'P_duration_Std':[standard_deviation_dur_p],
                                 'P_duration_Var':[variance_dur_p],
                                 
                                 'QRS_duration':[dic['QRS_duration'][0]],
                                 'QRS_duration_Mean':[mean_dur_qrs],
                                 'QRS_duration_Std':[standard_deviation_dur_qrs],
                                 'QRS_duration_Var':[variance_dur_qrs],
                                 
                                 'ST_interval':[dic['ST_interval'][0]],
                                 'ST_interval_Mean':[mean_st],
                                 'ST_interval_Std':[standard_deviation_st],
                                 'ST_interval_Var':[variance_st],  
                                 
                                 'TP_interval':[dic['TP_interval'][0]],
                                 'TP_interval_Mean':[mean_tp],
                                 'TP_interval_Std':[standard_deviation_tp],
                                 'TP_interval_Var':[variance_tp],
                                 
                                 'PR_interval':[dic['PR_interval'][0]],
                                 'PR_interval_Mean':[mean_pr],
                                 'PR_interval_Std':[standard_deviation_pr],
                                 'PR_interval_Var':[variance_pr],
                                 
                                 'RR_Pre': [dic['RR_Pre']], 
                                 'RR_Pos': [dic['RR_Pos']]              
                                 })
    
    #frames=[statistics_ecg, rr_pre, rr_pos]
    #features_ecg=pd.concat(frames, axis=1)
           
    return statistics_ecg
# In[19]:

def statistics_f (a):
    mean=np.nanmean(a)
    standard_deviation=np.nanstd(a)
    variance=np.nanvar(a)
    #skewness = (3*(mean-np.nanmedian(a)))/standard_deviation
    ##skewness=skew(a, nan_policy='omit')
    #kurtis=kurtosis(a, nan_policy='omit')
    
    return mean, standard_deviation, variance