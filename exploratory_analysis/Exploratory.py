import numpy as np
import pandas as pd

from scipy.interpolate import pchip_interpolate
from scipy import stats
from scipy.stats import t

import plotly
import plotly.express as px
import plotly.graph_objs as go

import statistics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# times_video={
#   "Fear": [84,210,373],
#   "Happy": [0,384,457],
#   "Neutral": [157,345,591]
# }

# jump_scare=[79, 177, 347,521]


times_video = {
    "Fear": ["01:24:627", "03:31:222", "06:13:774"],
    "Happy": ["06:24:751", "07:37:325"],
    "Neutral": ["02:37:555", "05:45:594", "09:51:499"],
    "Baseline": ["00:00:000", "00:00:000", "00:00:000"]
}

jump_scare = ["01:19:614", "02:57:366", "05:47:317", "08:41:948"]

#pio.renderers.default = "browser"
# In[]


def convert(millis):
    millis = int(millis)
    seconds = (millis/1000) % 60
    seconds = int(seconds)
    minutes = (millis/(1000*60)) % 60
    minutes = int(minutes)
    millis = millis % 1000
    millis = int(millis)

    return "%02d:%02d:%03d" % (minutes, seconds, millis)

# In[]:
def color_line(condition):

    if condition == "Fear":
        line_c= 'rgb(34,154,0)'
        
    elif condition == "Happy":
        line_c= 'rgb(255, 174, 66)'
        
    elif condition == "Neutral":
        line_c= 'rgb(119,136,153)'
        
    elif condition == "Baseline":
        line_c= 'rgb(47,79,79)'
        
    return line_c
    
# In[21.2] NORMALIZADO COM A MEDIA E DESVIO PADRAO DA BASELINE

def graph_profile(dic, dic_baseline, condition, sampling_rate, letter):

    df = pd.DataFrame()
    df2 = pd.DataFrame()
    df_signal = pd.DataFrame()  
    #vector = pd.DataFrame(columns=list(dic.keys()))
    vector = pd.DataFrame()
    vector2 = pd.DataFrame()
    x = {}
    x_new = {}
    r = {}
    b = {}
    norm = {}
    interpol = pd.DataFrame()
    interpol2 = pd.DataFrame()

    for participant in dic.keys():
        
        k = 0

        if participant.startswith(tuple(letter)):

            print(participant)
            
            #Indixes in ecg_Rpeak when occurs a cardiac pulse in ecg_rate. Have the same len has r and b
            x[participant] = np.where(dic[participant]["ecg_Rpeaks"] == 1)
            
            #ECG Rate for the emotion 
            r[participant] = dic[participant]["ecg_rate"]
            
            #Baseline ECG Rate
            b[participant] = dic_baseline[participant]["ecg_rate"]
            
            #Norm of the ECG Rate
            norm[participant] = (
                r[participant]-b[participant].mean(axis=0))/b[participant].std(axis=0)
            
            # index_stat = np.where(r["C001"] == np.amax(r["C001"]))
            # x_center_stat = x["C001"][0][index_stat]

            # index=np.where(r[participant] == np.amax(r[participant][0:times_video[condition][0]]))
            # x_center=x[participant][0][index]
            # x_diff=abs(x_center[0]-x_center_stat[0])

            # if x_center[0] >= x_center_stat[0]:
            #     x_new[participant]=x[participant][0]-x_diff

            # if x_center[0] < x_center_stat[0]:
            #     x_new[participant]=x[participant][0]+x_diff
            # print(max(x[participant][0]))
            
            plt.figure(1)
            plt.plot(x[participant][0],norm[participant])

            #print(norm[participant])
            
            #List of the ECG Rate in the right positions of the participant  
            for i in range(0, len(x[participant][0])):
            #for i,v in enumerate(x[participant][0]):
                #vector[participant].loc[v] = norm[participant][i]
                vector.loc[x[participant][0][i], participant] = norm[participant][k]
                vector2.loc[x[participant][0][i], participant] = r[participant][k]
                k += 1
                
            plt.figure(2)
            plt.plot(vector[participant])    
            
            #DataFrame with the real values of the ecg_rate and the real positions. The empty positions are NaN
            df = pd.concat([df, vector], axis=1)
            #Remove the duplicated columns
            df = df.loc[:, ~df.columns.duplicated()]
            
    #List of all the indexes of df
    #lin = np.linspace(0, len(dic[participant]['ecg_Rpeaks']), num=len(dic[participant]['ecg_Rpeaks']))
    #lin = np.linspace(df.index[0], df.index[-1], num=len(df))
    lin=np.sort(df.index.values)
    # print(lin)
    # print(min(lin))
    # print(max(lin))
    lin_time = []
    for d in lin:
        lin_time.append(convert(d))

    for participant in dic.keys():
        print(participant)
        if participant.startswith(tuple(letter)):

            interpol[participant] = pchip_interpolate(
                x[participant][0], norm[participant], np.linspace(min(lin), max(lin), num=len(lin)))

            interpol2[participant] = pchip_interpolate(
                x[participant][0], r[participant], np.linspace(min(lin), max(lin), num=len(lin)))
            
            # print(len(interpol[participant]))
            # print(len(lin))
            # plt.figure(3)
            # plt.plot(lin, interpol[participant])

        #df2 = pd.concat([df2, pd.DataFrame.from_dict(interpol)], axis=1)
        df2 = pd.concat([df2, interpol], axis=1)
        df2 = df2.loc[:, ~df2.columns.duplicated()]

        df_signal = pd.concat([df_signal, pd.DataFrame.from_dict(interpol2)], axis=1)
        df_signal = df_signal.loc[:, ~df_signal.columns.duplicated()]

    mean = df2.mean(axis=1)
    s = df2.std(axis=1)

    dof = len(df2.columns)-1
    confidence = .95
    t_crit = np.abs(t.ppf((1-confidence)/2, dof))
    conf_neg = mean-s*t_crit/np.sqrt(len(df2.columns))
    conf_pos = mean+s*t_crit/np.sqrt(len(df2.columns))

    

    # M= pd.DataFrame(index=range(len(df)), columns=['id','index'])
    # i=0

    # for index,row in df.iterrows():

    #     #print(index)
    #     #print(i)
    #     data_index=[]
    #     data_id=[]
        
    #     for value in row:
    #         if pd.notnull(value):
    #             data_index.append(index)
    #             data_id=((df == value).idxmax(axis=1)[index])
    #             #print('a')
    #     #data[i]=data_name
        
    #     M['index'].loc[i]=data_index
    #     M['id'].loc[i]=data_id
    #     #M = M.loc[:, ~M.columns.duplicated()]
    #     i=i+1
        
    # plt.figure(4)
    # plt.plot(lin,mean, color = 'blue')
    # plt.fill_between(lin , mean+s, mean-s, alpha = 0.4, color = 'red')
    # plt.axvline(x=times_video[condition][0])
    # plt.axvline(x=times_video[condition][1])
    # plt.axvline(x=times_video[condition][2])
    # plt.show()

    return mean, s, conf_neg, conf_pos,lin, lin_time, df, df_signal,df2


# In[zeros]
def count_zeros(lin, condition, conf_pos, conf_neg):

    rr = []
    rr2=[]
    count = 0
    retlist = {"Index": [], "Count": []}
    index = []

    if condition == "Baseline":
        index=range(len(lin))
        index_2=[i for i,v in enumerate(lin) if v > 60000]
             
    else:
        index = range(len(lin))

    for idx in index:
        stop = conf_pos[idx]
        start = conf_neg[idx]
        step = 0.005

        float_range_array = np.arange(start, stop, step)

        if np.all(float_range_array < 0) | np.all(float_range_array > 0):
            rr.append(float(idx))
        else:
            rr2.append(float(idx))

    for i in range(len(rr)-1):
        if rr[i] + 1 == rr[i+1]:
            count += 1
        else:
            # If it is not append the count and restart counting
            retlist["Count"].append(count)
            retlist["Index"].append(convert(lin[int(rr[i])]))
            count = 1
             
        percent = sum(retlist["Count"])/(len(lin))

    return rr, rr2, retlist, percent, index

# In[]


def fig_mean(condition, lin_time, mean, conf_pos, conf_neg):

    line_c= color_line(condition)
    
    layout = go.Layout(xaxis={'type': 'category', 'dtick': 5000.0})      
        
    fig = go.Figure(data=[

        go.Scatter(
            name='Upper Bound',
            x=lin_time,
            y=conf_pos,
            mode='lines',
            marker=dict(color="rgb(200,200,200)"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='CI 95%',
            x=lin_time,
            y=conf_neg,
            marker=dict(color="rgb(200,200,200)"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(200,200,200,0.7)',
            fill='tonexty',
        ),
        go.Scatter(
            name='Mean',
            x=lin_time,
            y=mean,
            mode='lines',
            line=dict(color=line_c),
        ),
        go.Scatter(
            name='zero line',
            x=lin_time,
            y=[0]*len(lin_time),
            marker=dict(color='rgb(0, 0, 0)'),
            showlegend=False
        )],
        layout=layout
    )

    if (condition == 'Fear') or (condition == 'Neutral'):
        fig.add_trace(go.Scatter(x=[times_video[condition][0], times_video[condition][0]], y=[-2, 2], line_width=3,
                      line_dash="dash", line_color="blue", mode='lines', name='start video 2'))
        fig.add_trace(go.Scatter(x=[times_video[condition][1], times_video[condition][1]], y=[-2, 2],
                      line_width=3, line_dash="dash", line_color="blue", mode='lines', name='start video 3'))
        fig.add_trace(go.Scatter(x=[times_video[condition][2], times_video[condition][2]], y=[-2, 2],
                      line_width=3, line_dash="dash", line_color="blue", mode='lines', name='start video 4'))
    
    elif condition == 'Happy' :
        fig.add_trace(go.Scatter(x=[times_video[condition][0], times_video[condition][0]], y=[-2, 2], line_width=3,
                      line_dash="dash", line_color="blue", mode='lines', name='start video 2'))
        fig.add_trace(go.Scatter(x=[times_video[condition][1], times_video[condition][1]], y=[-2, 2],
                      line_width=3, line_dash="dash", line_color="blue", mode='lines', name='start video 3'))
    
    if condition == "Fear":
        fig.add_trace(go.Scatter(x=[jump_scare[0], jump_scare[0]], y=[-2, 2], line_width=3,
                      line_dash="dash", line_color="red", mode='lines', name='jump scare'))
        fig.add_trace(go.Scatter(x=[jump_scare[1], jump_scare[1]], y=[-2, 2],
                      line_width=3, line_dash="dash", line_color="red", mode='lines', showlegend=False))
        fig.add_trace(go.Scatter(x=[jump_scare[2], jump_scare[2]], y=[-2, 2],
                      line_width=3, line_dash="dash", line_color="red", mode='lines', showlegend=False))
        fig.add_trace(go.Scatter(x=[jump_scare[3], jump_scare[3]], y=[-2, 2],
                      line_width=3, line_dash="dash", line_color="red", mode='lines', showlegend=False))
             
    fig.update_layout(
        yaxis_title='Mean',
        xaxis_title='Time',
        title='Mean '+condition,
        hovermode="x",
        font=dict(family="Times New Roman",
                size=20),
        width=1500,
        height=1000
        )

    fig.update_yaxes(range=[-2, 2])

    plotly.offline.plot(
        fig, filename=r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\graph_"+condition+"_with_Baseline_ci95.html", auto_open=False)

    return(fig)

# In[fig CI width]


def fig_CI(condition, lin_time, conf_pos, conf_neg):
    
    layout = go.Layout(xaxis={'type': 'category', 'dtick': 3000.0})    
    
    line_c= color_line(condition)
    
    fig = go.Figure(data=[

        go.Scatter(
            name='CI95 width',
            x=lin_time,
            y=conf_pos-conf_neg,
            mode='lines',
            line=dict(color=line_c),
        )],
        
        layout=layout      
    )

    if (condition == 'Fear') or (condition == 'Neutral'):
        fig.add_trace(go.Scatter(x=[times_video[condition][0], times_video[condition][0]], y=[0, 4], line_width=3,
                      line_dash="dash", line_color="blue", mode='lines', name='start video 2'))
        fig.add_trace(go.Scatter(x=[times_video[condition][1], times_video[condition][1]], y=[0, 4],
                      line_width=3, line_dash="dash", line_color="blue", mode='lines', name='start video 3'))
        fig.add_trace(go.Scatter(x=[times_video[condition][2], times_video[condition][2]], y=[0, 4],
                      line_width=3, line_dash="dash", line_color="blue", mode='lines', name='start video 4'))
    
    elif condition == 'Happy' :
        fig.add_trace(go.Scatter(x=[times_video[condition][0], times_video[condition][0]], y=[0, 4], line_width=3,
                      line_dash="dash", line_color="blue", mode='lines', name='start video 2'))
        fig.add_trace(go.Scatter(x=[times_video[condition][1], times_video[condition][1]], y=[0, 4],
                      line_width=3, line_dash="dash", line_color="blue", mode='lines', name='start video 3'))
    
    if condition == "Fear":
        fig.add_trace(go.Scatter(x=[jump_scare[0], jump_scare[0]], y=[0, 4], line_width=3,
                      line_dash="dash", line_color="red", mode='lines', name='jump scare'))
        fig.add_trace(go.Scatter(x=[jump_scare[1], jump_scare[1]], y=[0, 4],
                      line_width=3, line_dash="dash", line_color="red", mode='lines', showlegend=False))
        fig.add_trace(go.Scatter(x=[jump_scare[2], jump_scare[2]], y=[0, 4],
                      line_width=3, line_dash="dash", line_color="red", mode='lines', showlegend=False))
        fig.add_trace(go.Scatter(x=[jump_scare[3], jump_scare[3]], y=[0, 4],
                      line_width=3, line_dash="dash", line_color="red", mode='lines', showlegend=False))
             
    fig.update_layout(
        yaxis_title='CI95 width',
        xaxis_title='Time',
        title='CI95 width '+condition,
        hovermode="x",
        font=dict(family="Times New Roman",
                size=20),
        width=1000,
        height=1000
    )

    fig.update_yaxes(range=[0, 4])

    plotly.offline.plot(
        fig, filename=r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\CI95_width_"+condition+".html", auto_open=False)

    return(fig)

# In[fic scatter ]


def fig_scatter(condition, mean, conf_pos, conf_neg, rr):
    
    line_c= color_line(condition)

    fig = go.Figure(data=[
        go.Scatter(
            name='Upper Bound',
            x=mean,
            y=conf_pos-conf_neg,
            mode='markers',
            marker=dict(size=4, color=line_c),
            showlegend=False
            )])
    

    #fig = px.scatter(x=mean_zeros["mean_zeros"], y=conf_pos-conf_neg, color='blue')

    fig.update_layout(
        yaxis_title='CI width',
        xaxis_title='Mean',
        title='Scatter '+condition,
        hovermode="x",
        font=dict(family="Times New Roman",
                size=20),
        width=1000,
        height=1000
    )

    fig.update_yaxes(range=[-1, 4])
    fig.update_xaxes(range=[-2, 2])

   
    plotly.offline.plot(
        fig, filename=r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Scatter_graph_"+condition+".html", auto_open=False)

    return(fig)

# In[histo]


# def fig_histo(condition, mean, rr):

#     mean_zeros = []
#     bin_mean = []

#     group_labels = ['mean', 'mean_zeros']

#     colors = ['slategray', 'blue']

#     for idx in rr:
#         mean_zeros.append(mean[idx])

#     hist, bin_edges = np.histogram(mean_zeros, bins=np.arange(
#         min(mean), max(mean), .01), density=True)

#     for c in range(len(bin_edges)-1):
#         bin_mean.append((bin_edges[c]+bin_edges[c])/2)

#     #mean_zeros_new=np.random.choice(bin_mean, size=len(mean), replace=True, p=hist/100)

#     mean_zeros_new = mean_zeros + \
#         ([min(mean)-10*.01]*(len(mean)-len(mean_zeros)))

#     # Create distplot with curve_type set to 'normal'
#     fig = ff.create_distplot([mean, mean_zeros_new], group_labels,
#                              curve_type='normal',
#                              bin_size=.01,
#                              colors=colors,
#                              histnorm='probability density',
#                              show_rug=False)

#     # Add title
#     fig.update_layout(title_text='Distplot with Normal Distribution')
#     fig.update_xaxes(range=[min(mean), max(mean)])
#     fig.update_yaxes(range=[-1, 4])

#     plotly.offline.plot(
#         fig, filename=r"C:\Users\BeatrizHenriques\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Histogram_"+condition+"_mean.html")

#     return fig

# In[ ]


def fig_histo(condition, mean, rr, rr2):

    line_c= color_line(condition)
        
    mean_zeros = []
    mean_non_zeros =[]
    bin_mean = []
    mean_zeros_new = []

    for idx in rr:
        mean_zeros.append(mean[idx])

    for idx2 in rr2:
        mean_non_zeros.append(mean[idx2])
    # hist, bin_edges = np.histogram(mean_zeros["mean_zeros"], bins=np.arange(min(mean),max(mean),.01), density=True)
    # for c in range(len(bin_edges)-1):
    #     bin_mean.append((bin_edges[c]+bin_edges[c])/2)
    #mean_zeros_new=np.random.choice(bin_mean, size=len(mean), replace=True, p=hist/100)

    mean_zeros_new = mean_zeros+([min(mean)- 5*1]*(len(mean)-len(mean_zeros)))

    normal_curve = stats.norm.pdf(np.linspace(min(mean), max(
        mean)), statistics.mean(mean), statistics.stdev(mean))

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=mean,
        histnorm='probability density',
        xbins=dict(size='.01'),
        name="mean",
        marker_color=line_c
    ))

    fig.add_trace(go.Histogram(
        x=mean_zeros_new,
        histnorm='probability density',
        xbins=dict(size='.01'),
        name='mean_zeros',
        marker_color='rgb(169,169,169)'
    ))

    fig.add_trace(go.Scatter(
        x=np.linspace(min(mean), max(mean)),
        y=normal_curve,
        mode='lines',
        name='normal curve',
        marker_color=line_c
    ))

    fig.update_traces(opacity=1)
    fig.update_yaxes(range=[-1, 4])
    fig.update_xaxes(range=[-1, 1])
    
    fig.update_layout(
        yaxis_title='Probability',
        xaxis_title='Heart rate', 
        title='Distplot with Normal Distribution for '+condition,
        barmode='overlay',
        hovermode="x",
        font=dict(family="Times New Roman",
                size=20),
        width=1000,
        height=1000
    )

    plotly.offline.plot(
        fig, filename=r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Histogram_"+condition+"_mean.html", auto_open=False)

    return fig, mean_zeros, mean_non_zeros

# In[clustering ]

# def clustering(df,mean, mean_zeros,rr, n):
#     mean_clust_neg=[]
#     mean_clust_pos=[]
#     names_clust_neg=[]
#     names_clust_pos=[]
#     df_copy=df.copy()
#     i={}
    
#     for participant in df.keys():
        
#         print(participant)
        
#         a = pd.Series([df[participant].iloc[0]]*n)
#         b = pd.Series([df[participant].iloc[-1]]*n)
#         a.index= a.index + len(df_copy) + len(b)
        
#         a=pd.concat([a,df_copy[participant],b], ignore_index=True)
#         #dic_copy=pd.concat([pd.Series(dic[participant]["ecg_rate"][0]), pd.Series(dic[participant]["ecg_rate"]), pd.Series([dic[participant]["ecg_rate"][-1]]*5)], ignore_index=True)
        
#         #print(participant)
#         for i_rr,value_rr in enumerate(rr):
#             #print(value_rr)
#             if mean[value_rr] < 0:
#                 mean_clust_neg.append(np.array(a[np.arange(value_rr,value_rr+2*n+1)]))
#                 names_clust_neg.append([participant,i_rr])
                
#             elif mean[value_rr] > 0:
#                 mean_clust_pos.append(np.array(a[np.arange(value_rr,value_rr+2*n+1)]))
#                 names_clust_pos.append([participant,i_rr])
                
#     return mean_clust_neg,mean_clust_pos,a,names_clust_neg,names_clust_pos
  

def clustering(dic,df,df_signal, mean, rr, rr2,n):
    m_clust_neg=[]
    m_clust_pos=[]
    m_clust_null=[]
    n_clust_neg=[]
    n_clust_pos=[]
    n_clust_null=[]
    i={}
    x = {}
    c=0
    t=0
    #index of the true values 
    idx=df.index
    
    for participant in df.keys():
                
        d=dic[participant]['ecg_rate']
        d_copy=d.copy()
        
        #Array with the indexes in the df
        x[participant] = np.where(dic[participant]["ecg_Rpeaks"] == 1)
        
        #Array of x for eaxh participant but with one more inicial value 
        x2= np.concatenate(([x[participant][0][0]]*2,x[participant][0],[x[participant][0][-1]]*(n-1)), axis=0)
        print(participant)

        #Array of ecg_rate with one more inicial value and n-1 final values
        #a = pd.Series([d[0]])
        #b = pd.Series([d[-1]]*(n-1))
        #a.index= a.index + len(d_copy) + len(b)
        #a=np.concatenate(([dic[participant]['ecg_rate'][0]],dic[participant]['ecg_rate'],[[dic[participant]['ecg_rate'][-1]]*(n-1)]), axis=0)
        a=[dic[participant]['ecg_rate'][0]]*2+dic[participant]['ecg_rate'].tolist()+[dic[participant]['ecg_rate'][-1]]*(n-1)
        
        for i_rr,value_rr in enumerate(rr):
            t=t+1
            if mean[value_rr] < 0:
            
                  if pd.notnull(df[participant].iloc[int(value_rr)]):
    
                    ecg_indx = np.where(x[participant][0] == idx[int(value_rr)])
                    df_idx_i = np.where(idx == x2[ecg_indx[0][0]])
                    df_idx_f = np.where(idx == x2[ecg_indx[0][0]+n])
                           
                                       
                    array=a[ecg_indx[0][0] : ecg_indx[0][0]+n]
                    lista=[participant,df_idx_i[0][0],df_idx_f[0][0]]
                    
                    #List with indexes of the moment: 0-name of the participant; 1- beggining in the df; 2- finish in the df; 3- begging in the mean (len=20739)
                    if array not in m_clust_neg:
                        m_clust_neg.append(array)
                        n_clust_neg.append(lista +[x2[ecg_indx[0][0]], x2[ecg_indx[0][0]+n]])
                    else:
                        c=c+1
                    #m_clust_neg.append(np.array(a[np.arange(ecg_indx[0][0],ecg_indx[0][0]+5)]))                         
                    #n_clust_neg.append([participant,x2[ecg_indx[0][0]],x2[ecg_indx[0][0]+5], value_rr])    #ecg_indx[0][0] para indice no ecg_Rate
                    
                  else:
                    dif=[]
                    count=0
                    
                    #Verify the closer true heart rate 
                    for i in x[participant][0]:
       
                        dif.append(i-idx[int(value_rr)])
                        count=count+1
                        
                    t1_val, t1_idx = min([(abs(t1_val), t1_idx) for (t1_idx, t1_val) in enumerate(dif)])
                    df_idx_i = np.where(idx == x2[t1_idx])
                    df_idx_f = np.where(idx == x2[t1_idx+n])
                    
                    array=a[t1_idx: t1_idx+n]
                    lista=[participant,df_idx_i[0][0],df_idx_f[0][0]]
                    
                    #print(array)
                    #print(lista)
                    
                    if array not in m_clust_neg:
                        m_clust_neg.append(array)
                        n_clust_neg.append(lista +[x2[t1_idx], x2[t1_idx+n]])
                    else:
                        c=c+1
                #m_clust_neg.append(np.array(a[np.arange(t1_idx,t1_idx+5)])) #[df_signal[participant][value_rr]] if wanted the interpolated value
                #n_clust_neg.append([participant,x2[t1_idx],x2[t1_idx+5], value_rr])   #t1_idx   #idx[int(value_rr)], idx[int(value_rr)+5]
                
    
            elif mean[value_rr] >= 0:
            
                  if pd.notnull(df[participant].iloc[int(value_rr)]):
    
                    ecg_indx = np.where(x[participant][0] == idx[int(value_rr)])
                    df_idx_i = np.where(idx == x2[ecg_indx[0][0]])
                    df_idx_f = np.where(idx == x2[ecg_indx[0][0]+n])
                    
                    array=a[ecg_indx[0][0] : ecg_indx[0][0]+n]
                    lista=[participant,df_idx_i[0][0],df_idx_f[0][0]]
                    
                    #List with indexes of the moment: 0-name of the participant; 1- beggining in the df; 2- finish in the df; 3- begging in the mean (len=20739)
                    if array not in m_clust_pos:
                        m_clust_pos.append(array)
                        n_clust_pos.append(lista + [x2[ecg_indx[0][0]], x2[ecg_indx[0][0]+n]])
                    else:
                        c=c+1
                    #print(df_idx_i[0][0])
                    #print(value_rr)
                    #print(x2[ecg_indx[0][0]])
                    #print(a[ecg_indx[0][0]+5])
                    #print(df_signal[participant][df_idx_f[0][0]])
                    
                    
                  else:
                    dif=[]
                    count=0
                    for i in x[participant][0]:
       
                        dif.append(i-idx[int(value_rr)])
                        count=count+1
                        
                    t1_val, t1_idx = min([(abs(t1_val), t1_idx) for (t1_idx, t1_val) in enumerate(dif)]) 
                    df_idx_i = np.where(idx == x2[t1_idx])
                    df_idx_f = np.where(idx == x2[t1_idx+n])
                    
                    array=a[t1_idx : t1_idx+n]
                    lista=[participant,df_idx_i[0][0],df_idx_f[0][0]]
                
                    if array not in m_clust_pos:
                        m_clust_pos.append(array)
                        n_clust_pos.append(lista + [x2[t1_idx], x2[t1_idx+n]])
                    else:
                        c=c+1      

        for i_rr,value_rr in enumerate(rr2):
            #print(value_rr)
              t=t+1
        #    print(df[participant][int(value_rr)])
              if pd.notnull(df[participant].iloc[int(value_rr)]):

                ecg_indx = np.where(x[participant][0] == idx[int(value_rr)])

                df_idx_i = np.where(idx == x2[ecg_indx[0][0]])
                df_idx_f = np.where(idx == x2[ecg_indx[0][0]+n])
                
                array=a[ecg_indx[0][0] : ecg_indx[0][0]+n]
                lista=[participant,df_idx_i[0][0],df_idx_f[0][0]]
                
                if array not in m_clust_null:
                    m_clust_null.append(array)
                    n_clust_null.append(lista + [x2[ecg_indx[0][0]], x2[ecg_indx[0][0]+n]])   #ecg_indx[0][0] para indice no ecg_Rate
                else:
                    c=c+1    
                
              else:
                dif=[]
                count=0
                
                #Verify the closer true heart rate 
                for i in x[participant][0]:
   
                    dif.append(i-idx[int(value_rr)])
                    count=count+1
                    
                t1_val, t1_idx = min([(abs(t1_val), t1_idx) for (t1_idx, t1_val) in enumerate(dif)])
                df_idx_i = np.where(idx == x2[t1_idx])
                df_idx_f = np.where(idx == x2[t1_idx+n])   
                
                array=a[t1_idx : t1_idx+n]
                lista=[participant,df_idx_i[0][0],df_idx_f[0][0]]
                    
                if array not in m_clust_null:
                    m_clust_null.append(array)
                    n_clust_null.append(lista + [x2[t1_idx], x2[t1_idx+n]])
                else:
                    c=c+1   
     
    print(t)
    print(c)                
    return m_clust_neg,m_clust_pos, m_clust_null, n_clust_neg, n_clust_pos, n_clust_null


# In[corr]
def corrp(dic, mean, mean_clust,df, conf_pos, conf_neg, name_clust,condition):
    
    R=[0 for x in range(len(mean_clust))]
    a=[0 for x in range(len(mean_clust))]
    
    for i_a,val_a in enumerate(mean_clust):
        r = np.corrcoef(np.arange(0,5),val_a)
        R[i_a]=r[0,1]**2
        a[i_a]=np.polyfit(np.arange(0,5),val_a,1)
        
        if (name_clust[i_a][0]) == 'B003' and name_clust[i_a][1]== 13369:

            if R[i_a] < 0.005:

                trendpoly=np.poly1d(a[i_a])
                
                plt.figure()
                plt.title("Linear graph"+str(name_clust[i_a][0])+' in '+str(name_clust[i_a][1]))
                plt.plot(np.arange(0,5),mean_clust[i_a], c='b', label='hr')
                plt.plot(np.arange(0,5),trendpoly(np.arange(0,5)), c='g', label='model')
    
                #fig, axs = plt.subplots(2)
                plt.figure()
                plt.plot(mean[np.arange(int(name_clust[i_a][1]), int(name_clust[i_a][2]))],c='r', label='mean')
                plt.plot(conf_pos[np.arange(int(name_clust[i_a][1]), int(name_clust[i_a][2]))],c='b', label='conf_pos')
                plt.plot(conf_neg[np.arange(int(name_clust[i_a][1]), int(name_clust[i_a][2]))],c='b', label='conf_neg')
                plt.legend(loc='lower right')
    
                
                # fig = go.Figure(data=[

                # go.Scatter(
                #     name='signal',
                #     x=np.array(range(int(name_clust[i_a][1]),int(name_clust[i_a][2]))),
                #     y=df[int(name_clust[i_a][1]):int(name_clust[i_a][2])],
                #     mode='lines',
                #     line=dict(color='rgb(31, 119, 180)'),
                #     )])
                # fig.show()
                
                
                
                plt.figure()
                plt.plot(df[int(name_clust[i_a][1]):int(name_clust[i_a][2])], label='other participants')
                #break
            #axs[1].legend(loc='upper right')
            
            #axs[1].plot(np.arange(0,5), df[name_clust[i_a][0]][np.arange(int(name_clust[i_a][3]), int(name_clust[i_a][3])+5)])
            
                
        fig = go.Figure()
    
        fig.add_trace(go.Histogram(
            x=R,
            #histnorm='probability density',
            xbins=dict(size='.01'),
            name="pearson coef",
            #marker_color="#808080"
        ))
    
    
        #fig.update_xaxes(range=[min(mean), max(mean)])
        #fig.update_yaxes(range=[-1, 4])
        fig.update_layout(
            title='Histogram Pearson correlation coeficients for '+condition,
            hovermode="x",
            font=dict(family="Times New Roman",
                    size=20),
            width=1000,
            height=1000)
        
        plotly.offline.plot(
            fig, filename=r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Histo_corr_"+condition+".html", auto_open=False)


    
    return R
# In[clust]
# def clustering(df,mean, mean_zeros, n):

#     mean_clust=[]
#     a_inds=pd.DataFrame()
#     a = pd.Series([1]*n)
    
    
#     #hist, bin_edges = np.histogram(mean, bins=np.arange(
#     #    min(mean), max(mean), .01))
#     hist_zero, bin_edges_zero = np.histogram(
#         mean_zeros, bins=np.arange(min(mean), max(mean), .01))
    
#     #inds = np.digitize(mean, bin_edges)
#     inds_zero = np.digitize(mean_zeros, bin_edges_zero)

#     #df_copy=df.copy()
#     #a.index=a.index + len(df_copy) + len(a)

#     for participant in df.keys():
#         print(participant)
        
#         #a=pd.concat([a,df_copy[participant],a])

#         for mean_i,mean_value in enumerate(mean_zeros):
#             if mean_value<0:
            
#                 a_inds[mean_value] = np.where(mean == mean_zeros[mean_i])
#                 a_inds[mean_value] = int(a_inds[mean_value])
                
                
#                 for i in a_inds[mean_value]:
#                     #print(np.arange(i,i+k))
#                     #df[participant][np.arange(i,i+k)]
#                     #mean_clust.append(df[participant][[a_inds[mean_value][0]]:[a_inds[mean_value]+k][0]])
#                     mean_clust.append(np.array(df[participant][np.arange(i,i+n)]))



#     # model = KMeans()
#     # visualizer = KElbowVisualizer(model, k=(10,20))

#     # visualizer.fit(np.asarray(mean_clust))      # Fit the data to the visualizer
#     # visualizer.show()        # Finalize and render the figure
#     return mean_clust,a_inds
# In[kmeans_clust]
    
def kmeans_clust(condition,cond,mean_clust,k,n,name_clust):
    clust_names=[]
    count_names={}
    count_time={}
    X={}
    percent={}
    slope={}
    a=pd.DataFrame()
    
    for i,x in enumerate(mean_clust):
        a[i]=np.polyfit(np.arange(0,n),x,1)
        
    a=a.transpose()
    b=a[0].to_numpy()
    
    kmeans = KMeans(n_clusters=k, random_state=101)
    y_kmeans=kmeans.fit_predict(b.reshape(-1,1))         #retirei o mean_clust e experimentei com a ou #b.reshape(-1,1)
           
    clust={key:[] for key in np.unique(y_kmeans)}

    for y_i,y_value in enumerate(y_kmeans):
            clust[y_value].append([y_i,name_clust[y_i]])
    
    for clust_i in clust.keys():
        X[clust_i]=[]
        percent[clust_i]=[]
        slope[clust_i]=[]
        count_names[clust_i] = {}
        count_time[clust_i] = {}
        data=pd.Series([],dtype=pd.StringDtype())
        x=pd.Series([],dtype=pd.StringDtype())
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # fig.suptitle('i='+str(clust_i), fontsize=20)
        # ax.set_xlabel("Data points")
        # ax.set_ylabel("Mean value")
        
        fig = go.Figure()


        for i in clust[clust_i]:

            count_names[clust_i][i[1][0]] = count_names[clust_i].get(i[1][0], 0) + 1
            
            count_time[clust_i][i[1][1]] = count_time[clust_i].get(i[1][1], 0) + 1
            
            #X[clust_i].append(mean_clust[i[0]][int((2*n+1)/2)])
            X[clust_i].append(a[0][i[0]])
            
            data=data.append(pd.Series(mean_clust[i[0]]), ignore_index=True)
            x=x.append(pd.Series(np.arange(0,n)),ignore_index=True )

            trendpoly=np.poly1d(a.loc[i[0]])
            # plt.plot(np.arange(0,n),mean_clust[i[0]])
            #plt.plot(np.arange(0,n),trendpoly(np.arange(0,n)))
            
            fig.add_trace(go.Scatter(
                name='Mean',
                x=np.arange(0,n),
                y=trendpoly(np.arange(0,n)),
                mode='lines')
                )
            
        percent[clust_i]=sum(count_names[clust_i].values())/len(mean_clust)*100
        slope[clust_i]=np.mean(X[clust_i])

        # # df['data']=df['data'].append(data, ignore_index=True)
        # # df['x']=df['x'].append(x, ignore_index=True)
        # #pd.concat([df['data'],data], ignore_index=True)
        # #pd.concatdf['x']=df['x'].append(x, ignore_index=True)
        
        # #if cond == 'neg':
        # #    fig.update_yaxes(range=[-7, 15])
        # #elif cond ==  'pos':
        # #    fig.update_yaxes(range=[-8, 18])
        
        figu = px.density_heatmap(x=x, y=data, nbinsx=n, nbinsy = 100)
        
        fig.update_layout(
            yaxis_title='Heart rate mean',
            xaxis_title='Array',
            title='Density map '+condition,
            hovermode="x",
            font=dict(family="Times New Roman",
                    size=20),
            width=1000,
            height=1000)
        
        figu.update_layout(
            yaxis_title='Heart rate mean',
            xaxis_title='Array',
            title='Density map '+condition,
            hovermode="x",
            font=dict(family="Times New Roman",
                    size=20),
            width=1000,
            height=1000)
        
        figu.update_yaxes(range=[40, 180])
        
        plotly.offline.plot(
              figu, filename=r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Cluster\Cluster_" + str(condition) + str(cond) + str(clust_i) +".html", auto_open=False)
            
        plotly.offline.plot(
               fig, filename=r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Cluster\Cluster_trend_" + str(condition) + str(cond) + str(clust_i) +".html", auto_open=False)
       
        
    return y_kmeans,clust, count_names, count_time, X, percent, slope
    

# In[ ]
# a=pd.DataFrame()
# for i,x in enumerate(clust_pos_b[3]):
# #print(x[0])
#     a[i]=np.polyfit(np.arange(0,2*10+1), x,1)

# a=a.transpose()

# model = KMeans()
# # k is range of number of clusters.
# visualizer = KElbowVisualizer(model, k=(2,30),timings= True, metric='calinski_harabasz')
# # metric='calinski_harabasz',
# # metric='silhouette',
# visualizer.fit(a) # Fit data to visualizer
# visualizer.show()

# k=7

# kmeans = KMeans(n_clusters=k, random_state=101)
# y_means=kmeans.fit_predict(a)

# clust={key:[] for key in np.unique(y_means)}


# for y_i,y_value in enumerate(y_means):
#     clust[y_value].append([y_i,name_clust_pos_b[y_i]])

# n=10

# for clust_i in clust.keys():

#     fig = plt.figure()
# # ax = fig.add_subplot(111)
# # fig.suptitle('i='+str(clust_i), fontsize=20)
# # ax.set_xlabel("Data points")
# # ax.set_ylabel("Mean value")

#     for i in clust[clust_i]:
#         plt.plot(np.arange(0,2*n+1),mean_clust_pos_b[clust_pos_b[3][i[0]][0]])

# x={} 
# pera={}     
# for participant in dic_baseline.keys():
#     #if participant == 'B001':
#         x[participant] = np.where(dic_baseline[participant]["ecg_Rpeaks"] == 1)
#         pera[participant]=[]
#         for indx, val in enumerate(df_interpol_b[participant]):
            
#             #print(dic_baseline[participant]['ecg_Rpeaks'].loc[idx])
#             if dic_baseline[participant]['ecg_Rpeaks'].loc[indx].item()== 1:
#                 #print(indx)
#                 ind=np.where(x[participant][0]==indx)
#                 pera[participant].append( dic_baseline[participant]['ecg_rate'][ind[0][0]])
                
#             else:
#                 dif={}
#                 for i in range(0,len(x[participant][0])):
#                     dif[i] = x[participant][0][i]-indx
#                 t1_val, t1_idx = min([(abs(t1_val), t1_idx) for (t1_idx, t1_val) in enumerate(dif)])
                #pera[participant]=dic_baseline[participant]['ecg_rate'][t1_idx]
        
# #V2
# x={}      

# for participant in dic_baseline.keys():
#      x[participant] = np.where(dic_baseline[participant]["ecg_Rpeaks"] == 1)

# for clust in clust_pos_b.keys():
#     for clust_i in clust_pos_b[clust]:
        
#         participant=clust_pos_b[clust][clust_i][1][0]
#         t=clust_pos_b[clust][clust_i][1][1]
        
#         #print(dic_baseline[participant]['ecg_Rpeaks'].loc[idx])
       
#         if np.where(dic_baseline[participant]['ecg_Rpeak'][t]==1):
#             ind=np.where(x[participant][0]==t)
#             print(dic_baseline[participant]['ecg_rate'][ind])
            
#         else:         
#             for i in range(x[participant][0]):
#                 dif[i] = x[participant][0][i]-t
#             t1_val, t1_idx = min([(abs(t1_val), t1_idx) for (t1_idx, t1_val) in enumerate(dif)])
#             print(dic_baseline[participant]['ecg_rate'][t1_idx])
        
            


# #V3
# M={}
# for column in df_b:
#     print(column)
#     data_name=[]
#     M[i]=[]
#     for i in df_b[column].index:

#         #print(pd.notnull(df_b[column][i]))
#         if pd.notnull(df_b[column][i]):
#             #M[i]['name'].append(column)
#             data_name[i].append(column)

    

  #V4  
# M= pd.DataFrame(index=range(len(df_b)), columns=['id','index'])
# i=0

# for index,row in df_b.iterrows():

#     #print(index)
#     #print(i)
#     data_index=[]
#     data_id=[]
    
#     for value in row:
#         if pd.notnull(value):
#             data_index.append(index)
#             data_id=((df_b == value).idxmax(axis=1)[index])
#             #print('a')
#     #data[i]=data_name
    
#     M['index'].loc[i]=data_index
#     M['id'].loc[i]=data_id
#     #M = M.loc[:, ~M.columns.duplicated()]
#     i=i+1