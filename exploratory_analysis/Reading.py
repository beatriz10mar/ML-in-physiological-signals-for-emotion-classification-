#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import json
import csv


# In[8]:


def reading_function (file_path):
    
    # Opening the file and create a list of the header 
    with  open(file_path,'r') as file:
        header=[]
        # Adding the header to the list 'header'
        for line in file: # Go trough the lines of the file until is the end of the header 
            if line!='# EndOfHeader\n':
                line=line.replace('# ','\t') # Removing the # of the line
                header.append(line.strip()) # Adding the line to the list without spaces
            else:
                break
        # Creating a numpy.array with the columns of the file that contains nSeq, triggers and sensors
        data=np.genfromtxt(file_path) 
    ####################################### DATA #######################################
    
    # check if some column have Nan or only zeros
    for column in data.T:
        if np.isnan(column).any() or np.all((column==0)):
            print('ATTENTION: data contains zeros or nan values')
    
    ####################################### HEADER #######################################
   
    # Transforming the string into a dictionary (header format for easier access to variables)
    _header=json.loads(header[1])
    # Accessing the key of the header which is the mac address of the device used to collect the data
    mac=next(iter(_header.keys()))
    # The value of the 'mac' key is a dictionary ('dict_header') with several keys that contains information about the collection of data
    dict_header=_header[mac]
        
    # Acessing the header variables 
    sampling_rate=dict_header['sampling rate']
    resolution=dict_header['resolution']
    date=dict_header['date']
    time=dict_header['time']
    sensor=dict_header['sensor']
    label=dict_header['label']
    column=dict_header['column']
    sleeve_color=dict_header['sleeve color']
    
    return data, header, sampling_rate, resolution, date, time, sensor, label, column, sleeve_color

   
# In[3]:


def sensor_confirmation(sensor, sleeve_color, label, column):
    
    channel_EMG_MF='Error'; channel_EMG_TR='Error'; channel_EDA='Error'; channel_ECG='Error' 
    # Creating a list with the sensors that are connected
    sensors=[]
    # According to the protocol find the match (sensor name-color) of each sensor used
    for i in range(len(sensor)):
        if (sensor[i]=='EMG') & (sleeve_color[i]=='dark_blue'):
            channel_EMG_MF=label[i]
            sensors.append('EMG_MF')
        elif (sensor[i]=='EMG') & (sleeve_color[i]=='red'):
            channel_EMG_TR=label[i]
            sensors.append('EMG_TR')
        elif (sensor[i]=='EDA') & ((sleeve_color[i]=='red') or (sleeve_color[i]=='yellow')):
            channel_EDA=label[i]
            sensors.append('EDA')
        elif (sensor[i]=='ECG') & ((sleeve_color[i]=='gray') or (sleeve_color[i]=='red')):
            channel_ECG=label[i]
            sensors.append('ECG')
            
    if channel_EMG_MF=='Error'or channel_EMG_TR=='Error'or channel_EDA=='Error'or channel_ECG=='Error':
        print('ATTENTION: not all sensors were connected')
        print('The connected sensors are: ',sensors)
    
    x_nSeq='Nan'; x_trig='Nan'; x_EMG_MF='Nan'; x_EMG_TR='Nan'; x_EDA='Nan'; x_ECG='Nan'
    count=-1;
    # Finding the sensor column where the values are
    for column_ in column:
        count+=1;
        if column_=='nSeq':
            x_nSeq=count
        elif column_=='DI':
            x_trig=count
        elif column_==channel_EMG_MF:
            x_EMG_MF=count
        elif column_==channel_EMG_TR:
            x_EMG_TR=count
        elif column_==channel_EDA:
            x_EDA=count
        elif column_==channel_ECG:
            x_ECG=count    
    
    return x_nSeq, x_trig, x_EMG_MF, x_EMG_TR, x_EDA, x_ECG


# In[4]:


def trigger_function (total_trig, sampling_rate,sequence):
               
    # The triggers column is composed of zeros and ones, with zeros indicating moments when the trigger is not triggered and
    # ones when the trigger is pressed
    
    # Calculating consecutive differences
    diff_trig=[total_trig[i + 1] - total_trig[i] for i in range(0,len(total_trig)-1)] 

    # If the consecutive difference is equal to:
        # 0 - there was no change in the trigger state
        # 1 - the trigger was pressed
        # -1 - the trigger is no longer pressed
    
    # Search the index where the consecutive difference is 1
    index_trig=[] #variable with all the moments were the trigger was pressed
    for i in range(0,len(diff_trig)-1) : 
        if diff_trig[i] == 1 : 
            index_trig.append(i+1)
    
    # Triggers confirmation (if there is more than the supposed triggers)
    for i in range(1,len(index_trig)-1):
        if (index_trig[i]-index_trig[i-1])<(60*sampling_rate):
            index_trig.remove(index_trig[i-1]) # delete the first one
       
    #Mechanical trigger       
    time_df = pd.DataFrame({"Baseline":[300000], 
                            "Fear": [593000],
                            "Happy": [608000],
                            "Neutral": [668000]}
                           )
      
    time_df=time_df.loc[:, [sequence[0],sequence[1],sequence[2],sequence[3]]]
    
    trigger = np.zeros(len(total_trig))
    
    a=time_df.iloc[: , 0].values
    b=time_df.iloc[: , 1].values+a
    c=time_df.iloc[: , 2].values+b
    d=time_df.iloc[: , 3].values+c
    
    for i in range(len(trigger)-1,-1,-1):
        if i<index_trig[0]:
            trigger[i]=0
        elif (i>=index_trig[0]) & (i<(index_trig[0]+300)):
            trigger[i]=1
        elif (i>=index_trig[0]+300) & (i<(index_trig[0]+a+300)):
            trigger[i]=0
        elif (i>=(index_trig[0]+a+300)) & (i<(index_trig[0]+600+a)):
            trigger[i]=1
        elif (i>=(index_trig[0]+a+600)) & (i<(index_trig[0]+b+600)):
            trigger[i]=0
        elif (i>=(index_trig[0])+b+600) & (i<(index_trig[0]+b+900)):
            trigger[i]=1
        elif (i>=(index_trig[0]+b+900)) & (i<index_trig[0]+c+900):
            trigger[i]=0
        elif (i>=index_trig[0]+c+900) & (i<index_trig[0]+c+1200):
            trigger[i]=1
        elif (i>=(index_trig[0]+c+1200)) & (i<(index_trig[0]+d+1200)):
            trigger[i]=0
        elif (i>=index_trig[0]+d+1200) & (i<index_trig[0]+d+1500):
            trigger[i]=1
        elif (i>=(index_trig[0]+d+1500)):
            trigger[i]=0
         
      
    diff_trig_2=[trigger[i + 1] - trigger[i] for i in range(0,len(trigger)-1)] 

    index_trig_2=[] #variable with all the moments were the trigger was pressed
    for i in range(0,len(diff_trig_2)-1) : 
          if diff_trig_2[i] == 1 : 
              index_trig_2.append(i+1)
                
    return index_trig,index_trig_2


# In[7]:


def emotional_data (data_session, emotional_sequence, index_trig):
    
    # Always considering that Fear=1 ; Happy=2 and Neutral=3
    # Creating a list with the emotional sequence in numbers
    set=[]
    for i in emotional_sequence:
        if i=='Fear':
            set.append(1)
        elif i=='Happy':
            set.append(2)
        elif i=='Neutral':
            set.append(3)
    
    # Creating a numpy array with the number of the induced-emotion
    a=np.full((len(data_session[index_trig[1]:index_trig[2],1])),set[0])
    b=np.full((len(data_session[index_trig[2]:index_trig[3],1])),set[1])
    c=np.full((len(data_session[index_trig[3]:index_trig[4],1])),set[2])
    baseline=np.zeros((len(data_session[index_trig[0]:index_trig[1],1])))
    
    # Creating a numpy array vertically with a,b,c
    emotion=np.concatenate((baseline, a, b, c),axis=0)
    emotion=np.reshape(emotion,(-1, 1))
    # Adding the emotion numpy.array to the data_session
    emotion_data=np.append(emotion,data_session[index_trig[0]:index_trig[4],:], axis=1)
    
    # [Indicação da condição, EMG_MF, EMG_TR, EDA, ECG]
    
    return emotion_data 

