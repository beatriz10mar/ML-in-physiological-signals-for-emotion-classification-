# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns

from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import auc
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import itertools
from itertools import repeat

import plotly.express as px
import plotly
import plotly.graph_objects as go

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler 

import sys
import pickle
import os
# In[Path]:
sys.path.append(
    "C:\\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques")

path_excel = r'C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Processing the Signals\Try2_poly'

# In[]:

lista_pos_f, lista_neg_f, lista_null_f = pickle.load(
    open("features_f_file", "rb"))
lista_pos_b, lista_neg_b, lista_null_b = pickle.load(
    open("features_b_file", "rb"))
lista_pos_n, lista_neg_n, lista_null_n = pickle.load(
    open("features_n_file", "rb"))
lista_pos_h, lista_neg_h, lista_null_h = pickle.load(
    open("features_h_file", "rb"))
# In[]

# dic = dic_features_pos_b
# p = {}
# p1 = {}
# for clust in dic.keys():
#     count = 0
#     lenght = 0
#     p[clust] = {}

#     for i in dic[clust].keys():
#         p[clust][i] = {}
#         lenght = lenght+len(dic[clust][i])-2

#         for k, j in enumerate(dic[clust][i].keys()):
#             p[clust][i][j] = []

#             if ((j is not 'ecg_Rpeaks') and (j is not 'Participant')):
#                 if any(np.isnan(dic[clust][i][j])) is True:

#                     count = np.isnan(dic[clust][i][j]).sum()
#                     p[clust][i][j] = [len(dic[clust][i][j]), count]

#                     if count == len(dic[clust][i][j]):
#                           print([clust, i, j, dic[clust][i]['Participant']])

#     print('Tamanho:' + str(lenght))
#     print('--------------')

# p1[clust][i].append((np.count_nonzero(np.isnan(dic[clust][i][j]))/len(dic[clust][i][j]))*100)
# In[Label]:


def label(data, label1):
    
    # Define the data as X
    label = data[label1]
    
    # Tranform the label in number
    le = LabelEncoder()
    le.fit(label.unique())
        
    # Define the label as y
    y = le.transform(label)

    labels = np.unique(y)
    labels_name= list(le.classes_)
    
    data.insert(3, 'Label', y)

    X = pd.DataFrame()
    X = X.reset_index()
    X = data.iloc[:,6:]
    
    #del X['index']
    
    return labels, labels_name, data, X, y

# In[Missing]:


def missing_val(X):
    missing_values = ((X.isnull().sum())/len(X))*100

    print('Missing values:' + str(missing_values))
    print('------------------------------')

    # return data
# In[Scaling]:


def scaling(X):
    # X = X.reset_index()
    # del X['index']

    scaler = MinMaxScaler()
    X_s = pd.DataFrame(scaler.fit_transform(X), columns=X.keys())


    #d = pd.concat([data.iloc[:, 0:6], pd.DataFrame(scaler.fit_transform(
    #    data.iloc[:, 6:]), columns=data.iloc[:, 6:].keys())], axis=1)
    #del d['index']

    return X_s

# In[Train&Test]


def train_test_equal_part_diff_emotion(df_, part_list):

    emotion_list = []

    for ele in df_['Emotion']:
        if ele not in emotion_list:
            emotion_list.append(ele)

    number_part = len(part_list)
    print(emotion_list)
    b = list(itertools.combinations(emotion_list, 2))

    emotion_ind = []
    emotion_ind.extend(repeat(b[0], round(number_part/len(b))))
    emotion_ind.extend(repeat(b[1], round(number_part/len(b))))
    emotion_ind.extend(repeat(b[2], round(number_part/len(b))))

    random.shuffle(emotion_ind)

    emotion = [list(elem) for elem in emotion_ind]

    train_set = df_.copy()
    test_set = df_.copy()
    i = 0
    for part in part_list:
        emotion_comb = emotion[i]
        i = i+1

        if emotion_comb[0] == 'Fear' and emotion_comb[1] == 'Happy':
            train_set.drop(train_set[(train_set['ID participant'] == part) & (
                train_set['Emotion'] == 'Neutral')].index, inplace=True)
            test_set.drop(train_set[(train_set['ID participant'] == part) & (
                train_set['Emotion'] == 'Fear')].index, inplace=True)
            test_set.drop(train_set[(train_set['ID participant'] == part) & (
                train_set['Emotion'] == 'Happy')].index, inplace=True)

        elif emotion_comb[0] == 'Fear' and emotion_comb[1] == 'Neutral':
            train_set.drop(train_set[(train_set['ID participant'] == part) & (
                train_set['Emotion'] == 'Happy')].index, inplace=True)
            test_set.drop(train_set[(train_set['ID participant'] == part) & (
                train_set['Emotion'] == 'Fear')].index, inplace=True)
            test_set.drop(train_set[(train_set['ID participant'] == part) & (
                train_set['Emotion'] == 'Neutral')].index, inplace=True)

        elif emotion_comb[0] == 'Happy' and emotion_comb[1] == 'Neutral':
            train_set.drop(train_set[(train_set['ID participant'] == part) & (
                train_set['Emotion'] == 'Fear')].index, inplace=True)
            test_set.drop(train_set[(train_set['ID participant'] == part) & (
                train_set['Emotion'] == 'Happy')].index, inplace=True)
            test_set.drop(train_set[(train_set['ID participant'] == part) & (
                train_set['Emotion'] == 'Neutral')].index, inplace=True)

    return train_set, test_set


# [2]
# Divide the data in participants, meaning the training set has all the data corresponded to some participants and the test is done with all the data of the other participants
def train_test_diff_part_equal_emotion(df_, part_list, test_size):

    random.shuffle(part_list)

    n_train = round((1-test_size) * len(part_list))
    n_test = len(part_list) - n_train

    train_indices = random.sample(part_list, k=n_train)
    test_indices = []

    for part in part_list:
        if part not in train_indices:
            test_indices.append(part)

    train_set = df_[df_['ID participant'].isin(train_indices)].copy()
    test_set = df_[df_['ID participant'].isin(test_indices)].copy()

    return train_set, test_set


# [3]
def train_test_equal_part_emotion(data):

    k = 0
    for i in data["Class"].unique():
        data['Class'] = data['Class'].replace(i, k)
        k = k+1

    e_val = []
    for part in data['ID participant'].unique().tolist():
        for emotion in data['Emotion'].unique().tolist():
            e = int(random.choice(data['Class'].unique().tolist()))
            e_val.append(e)

    # print(e_val)
    train_set = data.copy()
    test_set = pd.DataFrame(columns=data.columns)

    i = 0
    for part in data['ID participant'].unique().tolist():
        test_set = test_set.append(train_set[(train_set['ID participant'] == part) & (
            train_set['Emotion'] == 'Fear') & (train_set['Class'] == e_val[i])])
        train_set.drop(train_set[(train_set['ID participant'] == part) & (
            train_set['Emotion'] == 'Fear') & (train_set['Class'] == e_val[i])].index, inplace=True)
        i = i+1
        test_set = test_set.append(train_set[(train_set['ID participant'] == part) & (
            train_set['Emotion'] == 'Happy') & (train_set['Class'] == e_val[i])])
        train_set.drop(train_set[(train_set['ID participant'] == part) & (
            train_set['Emotion'] == 'Happy') & (train_set['Class'] == e_val[i])].index, inplace=True)
        i = i+1
        test_set = test_set.append(train_set[(train_set['ID participant'] == part) & (
            train_set['Emotion'] == 'Neutral') & (train_set['Class'] == e_val[i])])
        train_set.drop(train_set[(train_set['ID participant'] == part) & (
            train_set['Emotion'] == 'Neutral') & (train_set['Class'] == e_val[i])].index, inplace=True)
        i = i+1

    return train_set, test_set


def train_test_class(data, n_class, labels, test_size):
    
    train_set=pd.DataFrame()
    test_set=pd.DataFrame()
       
    for i in range(0, n_class):
        
        part_list = data.index[data['Class'] == labels[i]].tolist()
        
        n_train = round((1-test_size) * len(part_list))
        n_test = len(part_list) - n_train

        train_indices = random.sample(part_list, k=n_train)
        test_indices = []

        for part in part_list:
            if part not in train_indices:
                test_indices.append(part)

        train_set = pd.concat([train_set, data[data.index.isin(train_indices)].copy()], ignore_index=True)
        test_set = pd.concat([test_set, data[data.index.isin(test_indices)].copy()], ignore_index=True)

    return train_set, test_set


def split_Discrete(data, emotion1, emotion2):

    e_val = []
    for part in data['Participant'].unique().tolist():
        for emotion in data['Emotion'].unique().tolist():
            e = int(random.choice(data['Excerpt'].unique().tolist()))
            e_val.append(e)

    train_set = data.copy()
    test_set = pd.DataFrame(columns=data.columns)
    i = 0
    for part in data['Participant'].unique().tolist():
        test_set = test_set.append(train_set[(train_set['Participant'] == part) & (
            train_set['Emotion'] == emotion1) & (train_set['Excerpt'] == e_val[i])])
        train_set.drop(train_set[(train_set['Participant'] == part) & (
            train_set['Emotion'] == emotion1) & (train_set['Excerpt'] == e_val[i])].index, inplace=True)
        i = i+1
        test_set = test_set.append(train_set[(train_set['Participant'] == part) & (
            train_set['Emotion'] == emotion2) & (train_set['Excerpt'] == e_val[i])])
        train_set.drop(train_set[(train_set['Participant'] == part) & (
            train_set['Emotion'] == emotion2) & (train_set['Excerpt'] == e_val[i])].index, inplace=True)
        i = i+1

    return train_set, test_set

# In[Metrics]:


def metrics(clf, y_test, y_proba, y_predict, labels, labels_name, c,g,k):

    # print('Classification Report')
    # # Accuracy, f1, recall, precision, support, avg
    # test_rep = classification_report(y_test, y_predict, digits=3)
    # dic = classification_report(y_test, y_predict, digits=3, output_dict=True)
    # print(test_rep)
    # print('---------------------------------')
    
    #Accuracy
    print('Balanced Accuracy')
    ba=balanced_accuracy_score(y_test, y_predict)
    print(ba)
    print('---------------------------------')

    # Cohen Kappa
    print('Cohen Kappa')
    ck=cohen_kappa_score(y_test, y_predict, labels=labels)
    print(ck)    
    print('---------------------------------')

    # Matthews Correlation Coeficient
    print('Matthews Correlation Coeficient')
    mcc=matthews_corrcoef(y_test, y_predict)
    print(mcc)
    print('---------------------------------')

    # ROC score
    print('ROC-AUC score')
    roc_auc=roc_auc_score(y_test, y_predict, labels=labels, multi_class='ovr')          #y_proba
    print(roc_auc)
    print('-----------------------------------')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_predict, labels=labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels_name)
    disp.plot()
    plt.title("C="+str(c)+", g="+str(g)+", K-fold="+str(k)+", BA="+str(np.round(ba*100,2)))
    plt.show()
    
    
    # import plotly.figure_factory as ff

    # # invert z idx values
    # z = cm[::-1]

    # x = labels_name
    # y =  x[::-1].copy() # invert idx values of x

    # # change each element of z to type string for annotations
    # z_text = [[str(y) for y in x] for x in z]

    # # set up figure 
    # fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # # add title
    # fig.update_layout(title_text="C="+str(c)+", g="+str(g)+", K-fold="+str(k)+", BA="+str(np.round(ba*100,2)),
    #               #xaxis = dict(title='x'),
    #               #yaxis = dict(title='x')
    #               )

    # # add custom xaxis title
    # fig.add_annotation(dict(font=dict(color="black",size=14),
    #                         x=0.5,
    #                         y=-0.15,
    #                         showarrow=False,
    #                         text="Predicted value",
    #                         xref="paper",
    #                         yref="paper"))
    
    # # add custom yaxis title
    # fig.add_annotation(dict(font=dict(color="black",size=14),
    #                         x=-0.35,
    #                         y=0.5,
    #                         showarrow=False,
    #                         text="Real value",
    #                         textangle=-90,
    #                         xref="paper",
    #                         yref="paper"))
    
    # # adjust margins to make room for yaxis title
    # fig.update_layout(margin=dict(t=50, l=200))
    
    # # add colorbar
    # fig['data'][0]['showscale'] = True
    # fig.show()
    # plotly.offline.plot( 
    #       fig, filename = r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Try2\CM\CM_"+str(c)+"_"+str(g)+"_"+str(k)+".html", auto_open=False)
    

    
    # ROC Curve
    # y_bi = label_binarize(y_test, classes=labels)

    # n_classes = len(labels)
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()

    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_bi[:, i], y_test[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])

    # for i in range(n_classes):
    #     plt.figure()
    #     plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    #     plt.plot([0, 1], [0, 1], 'k--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver operating characteristic example')
    #     plt.legend(loc="lower right")
    #     plt.show()
    
    metrics = pd.DataFrame(
            {'Balanced Accuracy': [np.round(ba*100,2)], 
              'Cohen Kappa Score': [np.round(ck*100,2)],
              'Matthews Correlation Coeficient': [np.round(mcc*100,2)],
              'ROC-AUC score': [np.round(roc_auc*100,2)]
              }) 
    print('---------------------------------')
    
    return metrics

# In[model]:

def emotional_model(c, gamma, kernel, x_res, y_res, x_test, y_test, labels, labels_name, k):
    
    clf_svm = svm.SVC(C=c, kernel=kernel, degree=gamma, probability=True)

    # clf = BaggingClassifier(base_estimator = clf_svm,
    #                         n_estimators = 20,
    #                         max_samples = len(X),
    #                         max_features = len(X.columns),
    #                         random_state = 0)

    clf = AdaBoostClassifier(base_estimator = clf_svm,
                             random_state=42,
                             )

    clf.fit(x_res, y_res)

    y_predict_train = clf.predict_proba(x_res)
    y_predict_test = clf.predict_proba(x_test)

    predictions_test = np.argmax(y_predict_test,  axis=1)
    predictions_train = np.argmax(y_predict_train, axis=1)

    results_train = pd.DataFrame()
    results_test = pd.DataFrame()
    results_train['Truth'] = y_res
    results_test['Truth'] = y_test
    results_train['Predicted Emotion Train'] = predictions_train
    results_test['Predicted Emotion Test'] = predictions_test

    print('SVM parameters: kernel- ' + str(kernel)+', C- '+str(c)+', gamma='+str(gamma)+
          '\nAdaBoost: n_estimators=25')
    
    # Classification report
    print('\nClassification Report Train:')
    clf_metrics_train = metrics(clf, y_res,  y_predict_train, predictions_train, labels, labels_name, c,gamma,k)
     
    print('*****************************************')
    
    print('\nClassification Report Test:')
    clf_metrics_test = metrics(clf, y_test, y_predict_test, predictions_test, labels, labels_name, c,gamma,k)

    return results_train, results_test, clf_metrics_train, clf_metrics_test

# In[]:
def svm_model(c, gamma, kernel, x_res, y_res, x_test, y_test, labels, labels_name, k):
    
    clf = svm.SVC(C=c, kernel=kernel, degree=gamma, probability=True)
    
    clf.fit(x_res, y_res)

    y_predict_train = clf.predict_proba(x_res)
    y_predict_test = clf.predict_proba(x_test)

    predictions_test = np.argmax(y_predict_test,  axis=1)
    predictions_train = np.argmax(y_predict_train, axis=1)

    results_train = pd.DataFrame()
    results_test = pd.DataFrame()
    results_train['Truth'] = y_res
    results_test['Truth'] = y_test
    results_train['Predicted Emotion Train'] = predictions_train
    results_test['Predicted Emotion Test'] = predictions_test

    print('SVM parameters: kernel- ' + str(kernel)+', C- '+str(c)+', gamma='+str(gamma)+
          '\nAdaBoost: n_estimators=25')
    
    # Classification report
    print('\nClassification Report Train:')
    clf_metrics_train = metrics(clf, y_res,  y_predict_train, predictions_train, labels, labels_name, c,gamma,k)
     
    print('*****************************************')
    print('\nClassification Report Test:')
    clf_metrics_test = metrics(clf, y_test, y_predict_test, predictions_test, labels, labels_name, c,gamma,k)

    return results_train, results_test, clf_metrics_train, clf_metrics_test

# In[Cross_validation]:


def cross_validation_smote(cond,frame, test_size, run):

    ext=[]
    metrics_train={}
    metrics_test={}
    
    test_metrics= pd.DataFrame()
    train_metrics = pd.DataFrame()
    
    #Dected nulls and fill with mean value 
    for j in frame: 
        ext.append(j['Subclass'].unique()[-1])
         
        for i in j.columns:
            if j[i].isnull().values.any():
                j[i] = j[i].fillna(value=j[i].mean())
                
    data = pd.concat(frame, ignore_index=True)
    
    #Randomize positions of the observations
    data = data.sample(frac = 1)
    
    #Dataset only with wave interval
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
    for index, row in data_4.iterrows():
        if row['Subclass'] == ext[0]:
            lst.append(row)
        if row['Subclass'] == ext[1]:
            lst.append(row)

    data_5 = pd.DataFrame(lst, columns=data_4.columns)
      
    #Data labeling
    labels, labels_name, data_5, X, y = label(data_5, 'Subclass')
    
    #Verify missing values
    missing_val(X)
    
    #Data Scaling
    X = scaling(X)
    
    # Paramters for sensability anbalyzis
    parameters = {'gamma':[10,5,1,0.1,0.01,0.001], 'C':[1,10, 25, 35, 50, 75, 85,100, 125, 150]}
    #parameters = {'gamma':[1,2,3,4,5,6,7], 'C':[1,10, 25, 35, 50, 75, 85,100, 125, 150]}  
    kernel= 'rbf'
    #p = pd.DataFrame(columns=['C', 'Gamma'])
    
    
    for c in parameters['C']:
         metrics_train[c] = {}
         metrics_test[c] = {}
         
         for g in parameters['gamma']:
             
            metrics_train[c][g] = pd.DataFrame()
            metrics_test[c][g] = pd.DataFrame()
            
            p = pd.DataFrame(
                    {'C': [c], 
                     'Gamma': [g]})              
            print(p)
            
            #Train and test split
            #x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)
            
            #Cross validations
            kf = StratifiedKFold(n_splits=5)
            kfold = kf.split(X, y)
            
            for k,(train_index, test_index) in enumerate(kfold):
            
                print('Fold: %2d, Training/Test Split Distribution: %s' % (k+1, np.bincount(y[train_index])))
                
                x_train , x_test = X.iloc[train_index,:],X.iloc[test_index,:]
                y_train , y_test = y[train_index] , y[test_index]
                           
                # Emotional Model
                results_train, results_test, clf_metrics_train, clf_metrics_test = emotional_model(
                    c, g, kernel, x_train, y_train, x_test, y_test, labels,labels_name, k)
       
                #Classification Report Train:
                results_train_m = pd.concat([p, clf_metrics_train], axis=1 )
                metrics_train[c][g] = pd.concat([metrics_train[c][g], results_train_m])
                
                #Classification Report Test:
                results_test_m = pd.concat([p,clf_metrics_test], axis=1)
                metrics_test[c][g] = pd.concat([metrics_test[c][g], results_test_m])
         
           #Mean of CV' metrics 
            AC_mean_train= metrics_train[c][g]['Balanced Accuracy'].mean()
            CKS_mean_train= metrics_train[c][g]['Cohen Kappa Score'].mean()
            MCC_mean_train= metrics_train[c][g]['Matthews Correlation Coeficient'].mean()
            ROC_AUC_mean_train= metrics_train[c][g]['ROC-AUC score'].mean()
          
            AC_mean_test= metrics_test[c][g]['Balanced Accuracy'].mean()
            CKS_mean_test= metrics_test[c][g]['Cohen Kappa Score'].mean()
            MCC_mean_test= metrics_test[c][g]['Matthews Correlation Coeficient'].mean()
            ROC_AUC_mean_test= metrics_test[c][g]['ROC-AUC score'].mean()
          
            AC_std_train= metrics_train[c][g]['Balanced Accuracy'].std()
            CKS_std_train= metrics_train[c][g]['Cohen Kappa Score'].std()
            MCC_std_train= metrics_train[c][g]['Matthews Correlation Coeficient'].std()
            ROC_AUC_std_train= metrics_train[c][g]['ROC-AUC score'].std()
          
            AC_std_test= metrics_test[c][g]['Balanced Accuracy'].std()
            CKS_std_test= metrics_test[c][g]['Cohen Kappa Score'].std()
            MCC_std_test= metrics_test[c][g]['Matthews Correlation Coeficient'].std()
            ROC_AUC_std_test= metrics_test[c][g]['ROC-AUC score'].std()
          
            Train_metrics = pd.DataFrame(
                      {'C': [c],
                       'Gamma': [g], 
                       'Balanced Accuracy': [AC_mean_train], 
                       'Cohen Kappa Score': [CKS_mean_train],
                       'Matthews Correlation Coeficient': [MCC_mean_train],
                       'ROC-AUC score': [ROC_AUC_mean_train],
                       'BA_std': [AC_std_train], 
                       'CKS_std': [CKS_std_train],
                       'MCC_std': [MCC_std_train],
                       'ROC-AUC_std': [ROC_AUC_std_train]
                       })
           
            Test_metrics = pd.DataFrame(
                      {'C': [c],
                       'Gamma': [g], 
                       'Balanced Accuracy': [AC_mean_test], 
                       'Cohen Kappa Score': [CKS_mean_test],
                       'Matthews Correlation Coeficient': [MCC_mean_test],
                       'ROC-AUC score': [ROC_AUC_mean_test],
                       'BA_std': [AC_std_test], 
                       'CKS_std': [CKS_std_test],
                       'MCC_std': [MCC_std_test],
                       'ROC-AUC_std': [ROC_AUC_std_test]
                       }) 
             
            train_metrics= pd.concat([train_metrics,Train_metrics])
            test_metrics= pd.concat([test_metrics,Test_metrics])
          
          
    for keyo in train_metrics[['Balanced Accuracy','Cohen Kappa Score','Matthews Correlation Coeficient','ROC-AUC score']]:
    #3D plot C, gamma and metrcis
        # fig = plotly.express.scatter_3d(metrics_test, x='C', y='Gamma', z='Balanced Accuracy',
        #                 color='species')
        fig = go.Figure(data = go.Contour(
                    z= train_metrics[keyo],
                    x=test_metrics['C'],
                    y=test_metrics['Gamma'],
                    contours=dict(
                        start=0,
                        end=100,
                        size=5),
                    ))
            
        fig.update_layout(
                title='Sensitive analysis',
                xaxis_title='C',
                yaxis_title='Gamma',
                font=dict(family="Times New Roman",
                        size=30),
                width=1000,
                height=1000,
                )

                
        fig2 = go.Figure(data = go.Contour(
                    z= test_metrics[keyo],
                    x=test_metrics['C'],
                    y=test_metrics['Gamma'],                 
                    contours=dict(
                        start=0,
                        end=100,
                        size=5),
                    ))
            
        fig2.update_layout(
                title='Sensitive analysis',
                xaxis_title='C',
                yaxis_title='Gamma',
                font=dict(family="Times New Roman",
                        size=30),
                width=1000,
                height=1000,
                )
         
        
        
        plotly.offline.plot( 
              fig, filename = r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Try2\3D_"+str(cond)+"_"+str(keyo)+"_SMOTE_Train.html")
        plotly.offline.plot( 
              fig2, filename = r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Try2\3D_"+str(cond)+"_"+str(keyo)+"_SMOTE_Test.html")


    return X, results_train, results_test, metrics_train, metrics_test, train_metrics, test_metrics

# In[Cross_validation]:

def cross_validation_norm(cond,frame, test_size, run):
    ext=[]
    metrics_train={}
    metrics_test={}
    
    test_metrics= pd.DataFrame()
    train_metrics = pd.DataFrame()
    
    #Dected nulls and fill with mean value 
    for j in frame:  
        ext.append(j['Subclass'].unique()[-1])
        
        for i in j.columns:
            if j[i].isnull().values.any():
                j[i] = j[i].fillna(value=j[i].mean())
                
    data = pd.concat(frame, ignore_index=True)
    
    #Randomize positions of the observations
    data = data.sample(frac = 1)
    
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
        if row['Subclass'] == ext[0]:
            lst.append(row)
        if row['Subclass'] == ext[1]:
            lst.append(row)

    data_5 = pd.DataFrame(lst, columns=data_2.columns)
      
    #Data labeling
    labels, labels_name, data_5, X, y = label(data_5, 'Subclass')
    
    #Verify missing values
    missing_val(X)
    
    #Data Scaling
    X = scaling(X)
    
    # Paramters for sensability anbalyzis
    
    #parameters = {'gamma':[10,5,1,0.1,0.01,0.001], 'C':[1,10, 25, 35, 50, 75, 85,100, 125, 150]}
    parameters = {'gamma':[1,2,3,4,5,6,7], 'C':[1,10, 25, 35, 50, 75, 85,100, 125, 150]}  
    kernel= 'poly'
    #p = pd.DataFrame(columns=['C', 'Gamma'])
    
    
    for c in parameters['C']:
         metrics_train[c] = {}
         metrics_test[c] = {}
         
         for g in parameters['gamma']:
             
            metrics_train[c][g] = pd.DataFrame()
            metrics_test[c][g] = pd.DataFrame()
            
            p = pd.DataFrame(
                    {'C': [c], 
                     'Gamma': [g]})              
            print(p)
            
            #Train and test split
            #x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)
    
            sm = SMOTE(random_state=42)         
            
            #Cross validations
            kf = StratifiedKFold(n_splits=5)
            kfold = kf.split(X, y)
            
            for k,(train_index, test_index) in enumerate(kfold):
            
                print('Fold: %2d, Training/Test Split Distribution: %s' % (k+1, np.bincount(y[train_index])))
                
                x_train , x_test = X.iloc[train_index,:],X.iloc[test_index,:]
                y_train , y_test = y[train_index] , y[test_index]
                
                #Resampling data
                x_res, y_res = sm.fit_resample(x_train, y_train)
                
                # Emotional Model
                results_train, results_test, clf_metrics_train, clf_metrics_test = emotional_model(
                    c, g, kernel, x_res, y_res, x_test, y_test, labels, labels_name, k)
       
                #Classification Report Train:
                results_train_m = pd.concat([p, clf_metrics_train], axis=1 )
                metrics_train[c][g] = pd.concat([metrics_train[c][g], results_train_m])
                
                #Classification Report Test:
                results_test_m = pd.concat([p,clf_metrics_test], axis=1)
                metrics_test[c][g] = pd.concat([metrics_test[c][g], results_test_m])
         
           #Mean of CV' metrics 
            AC_mean_train= metrics_train[c][g]['Balanced Accuracy'].mean()
            CKS_mean_train= metrics_train[c][g]['Cohen Kappa Score'].mean()
            MCC_mean_train= metrics_train[c][g]['Matthews Correlation Coeficient'].mean()
            ROC_AUC_mean_train= metrics_train[c][g]['ROC-AUC score'].mean()
          
            AC_mean_test= metrics_test[c][g]['Balanced Accuracy'].mean()
            CKS_mean_test= metrics_test[c][g]['Cohen Kappa Score'].mean()
            MCC_mean_test= metrics_test[c][g]['Matthews Correlation Coeficient'].mean()
            ROC_AUC_mean_test= metrics_test[c][g]['ROC-AUC score'].mean()
          
            AC_std_train= metrics_train[c][g]['Balanced Accuracy'].std()
            CKS_std_train= metrics_train[c][g]['Cohen Kappa Score'].std()
            MCC_std_train= metrics_train[c][g]['Matthews Correlation Coeficient'].std()
            ROC_AUC_std_train= metrics_train[c][g]['ROC-AUC score'].std()
          
            AC_std_test= metrics_test[c][g]['Balanced Accuracy'].std()
            CKS_std_test= metrics_test[c][g]['Cohen Kappa Score'].std()
            MCC_std_test= metrics_test[c][g]['Matthews Correlation Coeficient'].std()
            ROC_AUC_std_test= metrics_test[c][g]['ROC-AUC score'].std()
          
            Train_metrics = pd.DataFrame(
                      {'C': [c],
                       'Gamma': [g], 
                       'Balanced Accuracy': [AC_mean_train], 
                       'Cohen Kappa Score': [CKS_mean_train],
                       'Matthews Correlation Coeficient': [MCC_mean_train],
                       'ROC-AUC score': [ROC_AUC_mean_train],
                       'BA_std': [AC_std_train], 
                       'CKS_std': [CKS_std_train],
                       'MCC_std': [MCC_std_train],
                       'ROC-AUC_std': [ROC_AUC_std_train]
                       })
           
            Test_metrics = pd.DataFrame(
                      {'C': [c],
                       'Gamma': [g], 
                       'Balanced Accuracy': [AC_mean_test], 
                       'Cohen Kappa Score': [CKS_mean_test],
                       'Matthews Correlation Coeficient': [MCC_mean_test],
                       'ROC-AUC score': [ROC_AUC_mean_test],
                       'BA_std': [AC_std_test], 
                       'CKS_std': [CKS_std_test],
                       'MCC_std': [MCC_std_test],
                       'ROC-AUC_std': [ROC_AUC_std_test]
                       }) 
             
            train_metrics= pd.concat([train_metrics,Train_metrics])
            test_metrics= pd.concat([test_metrics,Test_metrics])
          
          
    for keyo in train_metrics[['Balanced Accuracy','Cohen Kappa Score','Matthews Correlation Coeficient','ROC-AUC score']]:
    #3D plot C, gamma and metrcis
        # fig = plotly.express.scatter_3d(metrics_test, x='C', y='Gamma', z='Balanced Accuracy',
        #                 color='species')
      
        fig = go.Figure(data = go.Contour(
                    z= train_metrics[keyo],
                    x=test_metrics['C'],
                    y=test_metrics['Gamma'],
                    contours=dict(
                        start=0,
                        end=100,
                        size=5),
                    ))
            
        fig.update_layout(title='Sensitive analysis',
                    xaxis_title='C',
                    yaxis_title='Degree',
                    font=dict(family="Times New Roman",
                            size=18),
                    width=1000,
                    height=1000,
                    )

                
        fig2 = go.Figure(data = go.Contour(
                    z= test_metrics[keyo],
                    x=test_metrics['C'],
                    y=test_metrics['Gamma'],   
                    contours=dict(
                        start=0,
                        end=100,
                        size=5),
                    ))
            
        fig2.update_layout(title='Sensitive analysis',
                    xaxis_title='C',
                    yaxis_title='Degree',
                    font=dict(family="Times New Roman",
                            size=18),
                    width=1000,
                    height=1000,
                    )
        
                    
        plotly.offline.plot( 
              fig, filename = r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Try2_poly\3D_"+str(cond)+"_"+str(keyo)+"_norm_Train.html")
        plotly.offline.plot( 
              fig2, filename = r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Try2_poly\3D_"+str(cond)+"_"+str(keyo)+"_norm_Test.html")
    

    return X, results_train, results_test, metrics_train, metrics_test, train_metrics, test_metrics

# In[normalized]:


def cross_validation(cond,frame, test_size, run):
    ext=[]
    metrics_train={}
    metrics_test={}
    
    test_metrics= pd.DataFrame()
    train_metrics = pd.DataFrame()
    
    #Dected nulls and fill with mean value 
    for j in frame:  
        ext.append(j['Subclass'].unique()[-1])
        
        for i in j.columns:
            if j[i].isnull().values.any():
                j[i] = j[i].fillna(value=j[i].mean())
                
    data = pd.concat(frame, ignore_index=True)
    
    #Randomize positions of the observations
    data = data.sample(frac = 1)
    
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
    for index, row in data_4.iterrows():
        # if row['Subclass'] == '+++++++':
        #     lst.append(row)
        # if row['Subclass'] == '++++++':
        #     lst.append(row)
        # if row['Subclass'] == '-------':
        #     lst.append(row)
        if row['Subclass'] == ext[0]:
                lst.append(row)
        if row['Subclass'] == ext[1]:
                lst.append(row)

    data_5 = pd.DataFrame(lst, columns=data_4.columns)
      
    #Data labeling
    labels, labels_name, data_5, X, y = label(data_5,'Subclass')
    
    #Verify missing values
    missing_val(X)
    
    #Data Scaling
    X = scaling(X)
    
    # Paramters for sensability anbalyzis
    
    #parameters = {'gamma':[10,5,1,0.1,0.01,0.001], 'C':[1,10, 25, 35, 50, 75, 85,100, 125, 150]}
    parameters = {'gamma':[1,2,3,4,5,6,7], 'C':[1,10, 25, 35, 50, 75, 85,100, 125, 150]}               
    kernel= 'poly' 
    #kernel = 'rbf'
    
    for c in parameters['C']:
         metrics_train[c] = {}
         metrics_test[c] = {}
         
         for g in parameters['gamma']:
             
            metrics_train[c][g] = pd.DataFrame()
            metrics_test[c][g] = pd.DataFrame()
            
            p = pd.DataFrame(
                    {'C': [c], 
                     'Gamma': [g]})              
            print(p)
            
            #Train and test split
            #x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)
    
            sm = SMOTE(random_state=42)         
            
            #Cross validations
            kf = StratifiedKFold(n_splits=5)
            kfold = kf.split(X, y)
            
            for k,(train_index, test_index) in enumerate(kfold):
            
                print('Fold: %2d, Training/Test Split Distribution: %s' % (k+1, np.bincount(y[train_index])))
                
                x_train , x_test = X.iloc[train_index,:],X.iloc[test_index,:]
                y_train , y_test = y[train_index] , y[test_index]
                
                #Resampling data
                x_res, y_res = sm.fit_resample(x_train, y_train)
                
                # Emotional Model
                results_train, results_test, clf_metrics_train, clf_metrics_test = emotional_model(
                    c, g, kernel, x_res, y_res, x_test, y_test, labels, labels_name, k)
       
                #Classification Report Train:
                results_train_m = pd.concat([p, clf_metrics_train], axis=1 )
                metrics_train[c][g] = pd.concat([metrics_train[c][g], results_train_m])
                
                #Classification Report Test:
                results_test_m = pd.concat([p,clf_metrics_test], axis=1)
                metrics_test[c][g] = pd.concat([metrics_test[c][g], results_test_m])
         
             #Mean of CV' metrics 
            AC_mean_train= metrics_train[c][g]['Balanced Accuracy'].mean()
            CKS_mean_train= metrics_train[c][g]['Cohen Kappa Score'].mean()
            MCC_mean_train= metrics_train[c][g]['Matthews Correlation Coeficient'].mean()
            ROC_AUC_mean_train= metrics_train[c][g]['ROC-AUC score'].mean()
            
            AC_mean_test= metrics_test[c][g]['Balanced Accuracy'].mean()
            CKS_mean_test= metrics_test[c][g]['Cohen Kappa Score'].mean()
            MCC_mean_test= metrics_test[c][g]['Matthews Correlation Coeficient'].mean()
            ROC_AUC_mean_test= metrics_test[c][g]['ROC-AUC score'].mean()
            
            AC_std_train= metrics_train[c][g]['Balanced Accuracy'].std()
            CKS_std_train= metrics_train[c][g]['Cohen Kappa Score'].std()
            MCC_std_train= metrics_train[c][g]['Matthews Correlation Coeficient'].std()
            ROC_AUC_std_train= metrics_train[c][g]['ROC-AUC score'].std()
            
            AC_std_test= metrics_test[c][g]['Balanced Accuracy'].std()
            CKS_std_test= metrics_test[c][g]['Cohen Kappa Score'].std()
            MCC_std_test= metrics_test[c][g]['Matthews Correlation Coeficient'].std()
            ROC_AUC_std_test= metrics_test[c][g]['ROC-AUC score'].std()
            
            Train_metrics = pd.DataFrame(
                        {'C': [c],
                         'Gamma': [g], 
                         'Balanced Accuracy': [AC_mean_train], 
                         'Cohen Kappa Score': [CKS_mean_train],
                         'Matthews Correlation Coeficient': [MCC_mean_train],
                         'ROC-AUC score': [ROC_AUC_mean_train],
                         'BA_std': [AC_std_train], 
                         'CKS_std': [CKS_std_train],
                         'MCC_std': [MCC_std_train],
                         'ROC-AUC_std': [ROC_AUC_std_train]
                         })
             
            Test_metrics = pd.DataFrame(
                        {'C': [c],
                         'Gamma': [g], 
                         'Balanced Accuracy': [AC_mean_test], 
                         'Cohen Kappa Score': [CKS_mean_test],
                         'Matthews Correlation Coeficient': [MCC_mean_test],
                         'ROC-AUC score': [ROC_AUC_mean_test],
                         'BA_std': [AC_std_test], 
                         'CKS_std': [CKS_std_test],
                         'MCC_std': [MCC_std_test],
                         'ROC-AUC_std': [ROC_AUC_std_test]
                         }) 
               
            train_metrics= pd.concat([train_metrics,Train_metrics])
            test_metrics= pd.concat([test_metrics,Test_metrics])
            
            
    for keyo in train_metrics[['Balanced Accuracy','Cohen Kappa Score','Matthews Correlation Coeficient','ROC-AUC score']]:
    #3D plot C, gamma and metrcis
        # fig = plotly.express.scatter_3d(metrics_test, x='C', y='Gamma', z='Balanced Accuracy',
        #                 color='species')
        # fig = go.Figure(data=[go.Mesh3d(x=test_metrics['C'],
        #                y=test_metrics['Gamma'],
        #                z =test_metrics[keyo],
        #                opacity=0.5,
        #                color="rgb(34,154,0)"
        #               )])
        
        fig = go.Figure(data = go.Contour(
                    z= train_metrics[keyo],
                    x=test_metrics['C'],
                    y=test_metrics['Gamma'],
                    contours=dict(
                        start=0,
                        end=100,
                        size=5),
                    ))
            
        fig.update_layout(
                title='Sensitive analysis',
                xaxis_title='C',
                yaxis_title='Degree',
                font=dict(family="Times New Roman",
                        size=30),
                width=1000,
                height=1000,
                )
        
                
        fig2 = go.Figure(data = go.Contour(
                    z= test_metrics[keyo],
                    x=test_metrics['C'],
                    y=test_metrics['Gamma'],
                    contours=dict(
                        start=0,
                        end=100,
                        size=5),
                    ))
            
        fig2.update_layout(
                title='Sensitive analysis',
                xaxis_title='C',
                yaxis_title='Degree',
                font=dict(family="Times New Roman",
                        size=30),
                width=1000,
                height=1000,
                )
        
                    
        plotly.offline.plot( 
              fig, filename = r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Try2_poly\3D_"+str(cond)+"_"+str(keyo)+"_Train.html")            
        plotly.offline.plot( 
              fig2, filename = r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Try2_poly\3D_"+str(cond)+"_"+str(keyo)+"_Test.html")

    return X, results_train, results_test, metrics_train, metrics_test, train_metrics, test_metrics

# In[with Neutral]:

def cross_validation_N(cond,frame, test_size, run):
    ext=[]
    metrics_train={}
    metrics_test={}
    
    test_metrics= pd.DataFrame()
    train_metrics = pd.DataFrame()
    
    #Dected nulls and fill with mean value 
    for j in frame:  
        ext.append(j['Subclass'].unique()[-1])
        
        for i in j.columns:
            if j[i].isnull().values.any():
                j[i] = j[i].fillna(value=j[i].mean())
                
    data = pd.concat(frame, ignore_index=True)
    
    #Randomize positions of the observations
    data = data.sample(frac = 1)
    
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
    for index, row in data_4.iterrows():
        if row['Subclass'] == ext[0]:
            lst.append(row)
        if row['Subclass'] == ext[1]:
            lst.append(row)
        if row['Subclass'] == ext[2]:
            lst.append(row)

    data_5 = pd.DataFrame(lst, columns=data_4.columns)
       
    #Data labeling
    labels, labels_name, data_5, X, y = label(data_5, 'Subclass')
    
    #Verify missing values
    missing_val(X)
    
    #Data Scaling
    X = scaling(X)
    
    #Undersmapling Neutral class
    ru = RandomUnderSampler(sampling_strategy='majority', random_state=42)
    x_rus, y_rus = ru.fit_resample(X, y)
    # Paramters for sensability anbalyzis
    
    #parameters = {'gamma':[10,5,1,0.1,0.01,0.001], 'C':[1,10, 25, 35, 50, 75, 85,100, 125, 150]}
    parameters = {'gamma':[1,2,3,4,5,6,7], 'C':[1,10, 25, 35, 50, 75, 85,100, 125, 150]}  
    kernel= 'poly'
    #p = pd.DataFrame(columns=['C', 'Gamma'])
    
    
    for c in parameters['C']:
         metrics_train[c] = {}
         metrics_test[c] = {}
         
         for g in parameters['gamma']:
             
            metrics_train[c][g] = pd.DataFrame()
            metrics_test[c][g] = pd.DataFrame()
            
            p = pd.DataFrame(
                    {'C': [c], 
                     'Gamma': [g]})              
            print(p)
            
            #Train and test split
            #x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)
    
            sm = SMOTE(random_state=42)         

            #Cross validations
            kf = StratifiedKFold(n_splits=5)
            kfold = kf.split(x_rus, y_rus)
            
            for k,(train_index, test_index) in enumerate(kfold):
            
                print('Fold: %2d, Training/Test Split Distribution: %s' % (k+1, np.bincount(y[train_index])))
                
                x_train , x_test = x_rus.iloc[train_index,:], x_rus.iloc[test_index,:]
                y_train , y_test = y_rus[train_index] , y_rus[test_index]
                
                #Resampling data
                x_res, y_res = sm.fit_resample(x_train, y_train)
                           
                # Emotional Model
                results_train, results_test, clf_metrics_train, clf_metrics_test = emotional_model(
                    c, g, kernel, x_res, y_res, x_test, y_test, labels, labels_name, k)
       
                #Classification Report Train:
                results_train_m = pd.concat([p, clf_metrics_train], axis=1 )
                metrics_train[c][g] = pd.concat([metrics_train[c][g], results_train_m])
                
                #Classification Report Test:
                results_test_m = pd.concat([p,clf_metrics_test], axis=1)
                metrics_test[c][g] = pd.concat([metrics_test[c][g], results_test_m])
         
             #Mean of CV' metrics 
            AC_mean_train= metrics_train[c][g]['Balanced Accuracy'].mean()
            CKS_mean_train= metrics_train[c][g]['Cohen Kappa Score'].mean()
            MCC_mean_train= metrics_train[c][g]['Matthews Correlation Coeficient'].mean()
            ROC_AUC_mean_train= metrics_train[c][g]['ROC-AUC score'].mean()
            
            AC_mean_test= metrics_test[c][g]['Balanced Accuracy'].mean()
            CKS_mean_test= metrics_test[c][g]['Cohen Kappa Score'].mean()
            MCC_mean_test= metrics_test[c][g]['Matthews Correlation Coeficient'].mean()
            ROC_AUC_mean_test= metrics_test[c][g]['ROC-AUC score'].mean()
            
            AC_std_train= metrics_train[c][g]['Balanced Accuracy'].std()
            CKS_std_train= metrics_train[c][g]['Cohen Kappa Score'].std()
            MCC_std_train= metrics_train[c][g]['Matthews Correlation Coeficient'].std()
            ROC_AUC_std_train= metrics_train[c][g]['ROC-AUC score'].std()
            
            AC_std_test= metrics_test[c][g]['Balanced Accuracy'].std()
            CKS_std_test= metrics_test[c][g]['Cohen Kappa Score'].std()
            MCC_std_test= metrics_test[c][g]['Matthews Correlation Coeficient'].std()
            ROC_AUC_std_test= metrics_test[c][g]['ROC-AUC score'].std()
            
            Train_metrics = pd.DataFrame(
                        {'C': [c],
                         'Gamma': [g], 
                         'Balanced Accuracy': [AC_mean_train], 
                         'Cohen Kappa Score': [CKS_mean_train],
                         'Matthews Correlation Coeficient': [MCC_mean_train],
                         'ROC-AUC score': [ROC_AUC_mean_train],
                         'BA_std': [AC_std_train], 
                         'CKS_std': [CKS_std_train],
                         'MCC_std': [MCC_std_train],
                         'ROC-AUC_std': [ROC_AUC_std_train]
                         })
             
            Test_metrics = pd.DataFrame(
                        {'C': [c],
                         'Gamma': [g], 
                         'Balanced Accuracy': [AC_mean_test], 
                         'Cohen Kappa Score': [CKS_mean_test],
                         'Matthews Correlation Coeficient': [MCC_mean_test],
                         'ROC-AUC score': [ROC_AUC_mean_test],
                         'BA_std': [AC_std_test], 
                         'CKS_std': [CKS_std_test],
                         'MCC_std': [MCC_std_test],
                         'ROC-AUC_std': [ROC_AUC_std_test]
                         }) 
               
            train_metrics= pd.concat([train_metrics,Train_metrics])
            test_metrics= pd.concat([test_metrics,Test_metrics])
            
            
    for keyo in train_metrics[['Balanced Accuracy','Cohen Kappa Score','Matthews Correlation Coeficient','ROC-AUC score']]:
    #3D plot C, gamma and metrcis
        # fig = plotly.express.scatter_3d(metrics_test, x='C', y='Gamma', z='Balanced Accuracy',
        #                 color='species')
        fig = go.Figure(data = go.Contour(
                    z= train_metrics[keyo],
                    x=test_metrics['C'],
                    y=test_metrics['Gamma'],
                    contours=dict(
                        start=0,
                        end=100,
                        size=5),
                    ))
            
        fig.update_layout(title='Sensitive analysis',
                    xaxis_title='C',
                    yaxis_title='Degree',
                    font=dict(family="Times New Roman",
                            size=30),
                    width=1000,
                    height=1000,
                    )

                
        fig2 = go.Figure(data = go.Contour(
                    z= test_metrics[keyo],
                    x=test_metrics['C'],
                    y=test_metrics['Gamma'],
                    contours=dict(
                        start=0,
                        end=100,
                        size=5),
                    ))
            
        fig2.update_layout(title='Sensitive analysis',
                    xaxis_title='C',
                    yaxis_title='Degree',
                    font=dict(family="Times New Roman",
                            size=30),
                    width=1000,
                    height=1000,
                    )
                    
        plotly.offline.plot( 
              fig, filename = r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Try2_poly\3D_"+str(cond)+"_"+str(keyo)+"_N_Train.html")
        plotly.offline.plot( 
              fig2, filename = r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Try2_poly\3D_"+str(cond)+"_"+str(keyo)+"_N_Test.html")
    
    return X, results_train, results_test, metrics_train, metrics_test, train_metrics, test_metrics

# In[Bag]:

def cross_validation_boost(cond,frame, test_size, run, N):
    ext=[]
    metrics_train={}
    metrics_test={}
    
    test_metrics= pd.DataFrame()
    train_metrics = pd.DataFrame()
    
    #Dected nulls and fill with mean value 
    for j in frame:  
        ext.append(j['Subclass'].unique()[-1])
        
        for i in j.columns:
            if j[i].isnull().values.any():
                j[i] = j[i].fillna(value=j[i].mean())
                
    data = pd.concat(frame, ignore_index=True)
    
    #Randomize positions of the observations
    data = data.sample(frac = 1)
    
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
    for index, row in data_4.iterrows():
        if row['Subclass'] == ext[0]:
            lst.append(row)
        if row['Subclass'] == ext[1]:
            lst.append(row)

    data_5 = pd.DataFrame(lst, columns=data_4.columns)
      
    #Data labeling
    labels, labels_name, data_5, X, y = label(data_5, 'Subclass')
    
    #Verify missing values
    missing_val(X)
    
    #Data Scaling
    X = scaling(X)
    
    # Paramters for sensability anbalyzis
    
    #parameters = {'gamma':[10,5,1,0.1,0.01,0.001], 'C':[1,10, 25, 35, 50, 75, 85,100, 125, 150]}
    parameters = {'gamma':[1,2,3,4,5,6,7], 'C':[1,10, 25, 35, 50, 75, 85,100, 125, 150]}  
    kernel= 'poly'
    #p = pd.DataFrame(columns=['C', 'Gamma'])
    
    
    for c in parameters['C']:
         metrics_train[c] = {}
         metrics_test[c] = {}
         
         for g in parameters['gamma']:
             
            metrics_train[c][g] = pd.DataFrame()
            metrics_test[c][g] = pd.DataFrame()
            
            p = pd.DataFrame(
                    {'C': [c], 
                     'Gamma': [g]})              
            print(p)
            
            #Train and test split
            #x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)
    
            sm = SMOTE(random_state=42)         
            
            #Cross validations
            kf = StratifiedKFold(n_splits=5)
            kfold = kf.split(X, y)
            
            for k,(train_index, test_index) in enumerate(kfold):
            
                print('Fold: %2d, Training/Test Split Distribution: %s' % (k+1, np.bincount(y[train_index])))
                
                x_train , x_test = X.iloc[train_index,:],X.iloc[test_index,:]
                y_train , y_test = y[train_index] , y[test_index]
                
                #Resampling data
                x_res, y_res = sm.fit_resample(x_train, y_train)
                
                # Emotional Model
                results_train, results_test, clf_metrics_train, clf_metrics_test = svm_model(
                    c, g, kernel, x_res, y_res, x_test, y_test, labels, labels_name,k)
       
                #Classification Report Train:
                results_train_m = pd.concat([p, clf_metrics_train], axis=1 )
                metrics_train[c][g] = pd.concat([metrics_train[c][g], results_train_m])
                
                #Classification Report Test:
                results_test_m = pd.concat([p,clf_metrics_test], axis=1)
                metrics_test[c][g] = pd.concat([metrics_test[c][g], results_test_m])
         
             #Mean of CV' metrics 
            AC_mean_train= metrics_train[c][g]['Balanced Accuracy'].mean()
            CKS_mean_train= metrics_train[c][g]['Cohen Kappa Score'].mean()
            MCC_mean_train= metrics_train[c][g]['Matthews Correlation Coeficient'].mean()
            ROC_AUC_mean_train= metrics_train[c][g]['ROC-AUC score'].mean()
            
            AC_mean_test= metrics_test[c][g]['Balanced Accuracy'].mean()
            CKS_mean_test= metrics_test[c][g]['Cohen Kappa Score'].mean()
            MCC_mean_test= metrics_test[c][g]['Matthews Correlation Coeficient'].mean()
            ROC_AUC_mean_test= metrics_test[c][g]['ROC-AUC score'].mean()
            
            AC_std_train= metrics_train[c][g]['Balanced Accuracy'].std()
            CKS_std_train= metrics_train[c][g]['Cohen Kappa Score'].std()
            MCC_std_train= metrics_train[c][g]['Matthews Correlation Coeficient'].std()
            ROC_AUC_std_train= metrics_train[c][g]['ROC-AUC score'].std()
            
            AC_std_test= metrics_test[c][g]['Balanced Accuracy'].std()
            CKS_std_test= metrics_test[c][g]['Cohen Kappa Score'].std()
            MCC_std_test= metrics_test[c][g]['Matthews Correlation Coeficient'].std()
            ROC_AUC_std_test= metrics_test[c][g]['ROC-AUC score'].std()
            
            Train_metrics = pd.DataFrame(
                        {'C': [c],
                         'Gamma': [g], 
                         'Balanced Accuracy': [AC_mean_train], 
                         'Cohen Kappa Score': [CKS_mean_train],
                         'Matthews Correlation Coeficient': [MCC_mean_train],
                         'ROC-AUC score': [ROC_AUC_mean_train],
                         'BA_std': [AC_std_train], 
                         'CKS_std': [CKS_std_train],
                         'MCC_std': [MCC_std_train],
                         'ROC-AUC_std': [ROC_AUC_std_train]
                         })
             
            Test_metrics = pd.DataFrame(
                        {'C': [c],
                         'Gamma': [g], 
                         'Balanced Accuracy': [AC_mean_test], 
                         'Cohen Kappa Score': [CKS_mean_test],
                         'Matthews Correlation Coeficient': [MCC_mean_test],
                         'ROC-AUC score': [ROC_AUC_mean_test],
                         'BA_std': [AC_std_test], 
                         'CKS_std': [CKS_std_test],
                         'MCC_std': [MCC_std_test],
                         'ROC-AUC_std': [ROC_AUC_std_test]
                         }) 
               
            train_metrics= pd.concat([train_metrics,Train_metrics])
            test_metrics= pd.concat([test_metrics,Test_metrics])
            
            
    for keyo in train_metrics[['Balanced Accuracy','Cohen Kappa Score','Matthews Correlation Coeficient','ROC-AUC score']]:
    #3D plot C, gamma and metrcis
        # fig = plotly.express.scatter_3d(metrics_test, x='C', y='Gamma', z='Balanced Accuracy',
        #                 color='species')
        fig = go.Figure(data = go.Contour(
                    z= train_metrics[keyo],
                    x=test_metrics['C'],
                    y=test_metrics['Gamma'],
                    contours=dict(
                        start=0,
                        end=100,
                        size=5),
                    ))
            
        fig.update_layout(title='Sensitive analysis',
                    xaxis_title='C',
                    yaxis_title='Degree',
                    font=dict(family="Times New Roman",
                            size=18),
                    width=1000,
                    height=1000,
                    )

                
        fig2 = go.Figure(data = go.Contour(
                    z= test_metrics[keyo],
                    x=test_metrics['C'],
                    y=test_metrics['Gamma'],
                    contours=dict(
                        start=0,
                        end=100,
                        size=5),
                    ))
            
        fig2.update_layout(title='Sensitive analysis',
                    xaxis_title='C',
                    yaxis_title='Degree',
                    font=dict(family="Times New Roman",
                        size=18),
                    width=1000,
                    height=1000,
                    )
                    
        plotly.offline.plot( 
              fig, filename = r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Try2_poly\3D_"+str(cond)+"_"+str(keyo)+"_Boost_Train.html")
        plotly.offline.plot( 
              fig2, filename = r"C:\Users\Bia10\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\Try2_poly\3D_"+str(cond)+"_"+str(keyo)+"_Boost_Test.html")
    
    
    return X, results_train, results_test, metrics_train, metrics_test, train_metrics, test_metrics
# In[]:
# Classifyfor Fear
frame = [lista_neg_n, lista_pos_n]

# In[]: 
run = 5 
results = cross_validation('Neutral',frame, test_size=0.25, run=run)

filename = 'neutral_results_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(results, file)
file.close()

# In[]:
run = 5 
results_smote = cross_validation_smote('Fear',frame, test_size=0.25, run=run)

filename = 'results_smote_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(results_smote, file)
file.close()

# In[]:
run = 5 
results_norm = cross_validation_norm('Fear',frame, test_size=0.25, run=run)

filename = 'results_norm_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(results_norm, file)
file.close()
# In[]:
run = 5 
results_boost = cross_validation_boost('Fear',frame, test_size=0.25, run=run, N=5)

filename = 'results_bag_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(results_boost, file)
file.close()

# In[]:
frame = [lista_neg_f, lista_null_f, lista_pos_f]
run = 5 
results_N = cross_validation_N('Fear',frame, test_size=0.25, run=run)

filename = 'results_N_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(results_N, file)
file.close()

# In[]:
    
results = pickle.load(open("results_file", "rb"))
results_smote = pickle.load(open("results_smote_file", "rb"))
results_norm = pickle.load(open("results_norm_file", "rb"))
results_boost = pickle.load(open("results_bag_file", "rb"))
results_N = pickle.load(open("results_N_file", "rb"))


# In[Neutral]:
    
frame = [lista_neg_n, lista_pos_n]

run = 5 
neutral_poly_results = cross_validation('Neutral',frame, test_size=0.25, run=run)

filename = 'neutral_results_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(neutral_poly_results, file)
file.close()
# In[]:
#Classify in Happy
frame = [lista_neg_h, lista_pos_h]

run = 5 
happy_poly_results = cross_validation('Happy',frame, test_size=0.25, run=run)

filename = 'happy_poly_results_file'
file = open(path_excel+'/'+filename, 'wb')
pickle.dump(happy_poly_results, file)
file.close()
# In[]:

 # In[sc]:

# run=1 
# metrics_train = pd.DataFrame()
# metrics_test = pd.DataFrame()

# for i in data.columns:
#     if data[i].isnull().values.any():
#         data[i] = data[i].fillna(value=data[i].mean())

# l = ['QRS_duration', 'P_duration', 'T_duration', 'ST_interval', 'TP_interval', 'PR_interval']

# data_2= pd.DataFrame()
# data_2=data[['Emotion', 'Class', 'Subclass','ID participant', 'Idx_i']].copy()
# for keyo in l:
#     data_2[str('Result_'+keyo)] = data[keyo]/data[str(keyo+'_Mean')]

# data_3= pd.DataFrame()
# data_3=data[['Emotion', 'Class', 'Subclass','ID participant', 'Idx_i','QRS_duration', 'P_duration', 'T_duration', 'ST_interval', 'TP_interval', 'PR_interval']].copy()

# j = ['QRS_duration', 'P_duration', 'T_duration', 'ST_interval', 'TP_interval', 'PR_interval']
# data_4= pd.DataFrame()
# data_4=data[['Emotion', 'Class', 'Subclass','ID participant', 'Idx_i']].copy()
# for keyo in j:
#     data_4[str('Result_'+keyo)] = data[keyo] * data['ECG_Rate_Mean']

# data_5 = pd.DataFrame()

# for index, row in data_3.iterrows():
#     if (row['Subclass'] == '++++++') or row['Subclass'] == '-------':
#         data_5 = pd.concat([data_5 ,row], ignore_index=True)
        
# labels, df1, X, y = label(data_5, 'Emotion','Class')
# missing_val(X)
# X = scaling(X)

#Randomize positions of the observations
# randomlist = random.sample(range(0, len(X.columns)), len(X.columns))
# X=X.iloc[:, randomlist]
# In[]
# from sklearn.decomposition import PCA

# pca = PCA(n_components=4)
# X_pca = pd.DataFrame(pca.fit_transform(X), columns=['PC1', 'PC2', 'PC3', 'PC4'])

#*******************************************************
# t = time.time()
# c = [*range(len(data.columns))]
# l = [*range(len(data))]
# combinations = {(i,j) for i in c for j in l}
# random_pos = random.sample(combinations, len(combinations))
# X2=pd.DataFrame()
# i=0
# for column in data:
#     for index, row in data.iterrows():
#         X2.loc[index,column] =  data.iloc[random_pos[i][1],random_pos[i][0]]
#         i=i+1
# elapsed = time.time() - t
# print(elapsed)
#*******************************************************
# c = len(X.columns)
# r = len(X)

# list_c = []
# list_r = []

# for i in range(c):
#     list_c.append(X.columns[i])

# for j in range(r):
#     list_r.append(j)

# original_permutation = list(itertools.product(list_c, list_r)) 

# list_c_shuffle = random.sample(list_c,len(list_c))
# list_r_shuffle = random.sample(list_r,len(list_r))

# permutations = list(itertools.product(list_c_shuffle, list_r_shuffle)) 
# permutations_random = random.sample(permutations,len(permutations))


# X2 = X.copy()

# for i in range(len(permutations_random)):
#         X2.loc[permutations_random[i][1], permutations_random[i][0]] = X[original_permutation[i][0]].iloc[original_permutation[i][1]]
  
# In[]:  
kernel= 'rbf'  
c=100
g=1

p = pd.DataFrame(
        {'C': [c], 
         'Gamma': [g]})              
print(p)

# x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)
# print('Split done')

sm = SMOTE(random_state=42)
x_res, y_res = sm.fit_resample(X, y)  #X_pca

kf = StratifiedKFold(n_splits=5)
kfold = kf.split(x_res, y_res)

for k,(train_index, test_index) in enumerate(kfold):

    print('Fold: %2d, Training/Test Split Distribution: %s' % (k+1, np.bincount(y_res[train_index])))
    
    x_train2 , x_test = x_res.iloc[train_index,:],x_res.iloc[test_index,:]
    y_train2 , y_test = y_res[train_index] , y_res[test_index]

    # Emotional Model
    results_train, results_test, clf_metrics_train, clf_metrics_test = emotional_model (
        c, g, kernel, x_train2, y_train2, x_test, y_test, labels)
    # best_params, results_train, results_test, clf_metrics_train, clf_metrics_tes = grid_emotional_model(
    #     x_res, y_res, x_train, y_train, x_test, y_test)
    
    results_train_m = pd.concat([p, clf_metrics_train], axis=1 )
    results_test_m = pd.concat([p,clf_metrics_test], axis=1)
    
    #Classification Report Train:
    metrics_train = pd.concat([metrics_train, results_train_m])
    
    #Classification Report Test:
    metrics_test = pd.concat([metrics_test, results_test_m])

  
# In[voting]:
# C = [100, 50, 50, 100]
# kernel = ['rbf', 'rbf', 'poly', 'poly']
# gamma = [10, 5, 10, 5]
# degree = [4, 5, 6, 7]
# clf = {}

# for i in range(0, 4):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.33, random_state=42)
#     clf[i] = svm.SVC(C=C[i], kernel=kernel[i], gamma=gamma[i],
#                      degree=degree[i], class_weight='balanced', probability=True)

# eclf = VotingClassifier(estimators=[(
#     'svm1', clf[1]), ('svm2', clf[2]), ('svm3', clf[3]), ('svm4', clf[0])], voting='soft')
# eclf.fit(X_train, y_train)

# y_predict_train = eclf.predict_proba(X_train)
# y_predict_test = eclf.predict_proba(X_test)

# predictions_test = np.argmax(y_predict_test, axis=1)
# predictions_train = np.argmax(y_predict_train, axis=1)

# results_train = pd.DataFrame()
# results_test = pd.DataFrame()
# results_train['Truth'] = y_train
# results_test['Truth'] = y_test
# results_train['Predicted Emotion Train'] = predictions_train
# results_test['Predicted Emotion Test'] = predictions_test

# # Classification report
# print('\nClassification Report Train:')
# dic_train = metrics(eclf, y_train,  y_predict_train, predictions_train, labels)

# print('\nClassification Report Test:')
# dic_test = metrics(eclf, y_test, y_predict_test, predictions_test, labels)

# In[]:
    
import seaborn as sns
#g = sns.PairGrid(data_5[['Class', 'Result_ECG_Rate', 'Result_QRS_duration', 'Result_P_duration', 'Result_T_duration','Result_ST_interval', 'Result_TP_segment', 'Result_PR_interval']], hue='Class', palette='pastel', diag_sharey=False)
#g = sns.PairGrid(data_5[['Subclass', 'ECG_Rate', 'QRS_duration', 'P_duration', 'T_duration','ST_interval', 'TP_segment', 'PR_interval']], hue='Subclass', palette='pastel', diag_sharey=False)
g = sns.PairGrid(data_2[['Class', 'ECG_Rate', 'QRS_duration', 'P_duration', 'T_duration','ST_interval', 'TP_segment', 'PR_interval']], hue='Class', palette='pastel', diag_sharey=False)

g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot)
g.add_legend()
# In[]:
#Creating dataset
# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
for index, row in x_res.iterrows():
    az = row['Result_T_duration']
    ax = row['Result_P_duration']
    ay = row['Result_QRS_duration']
    
    if y_res[index] == 1:
        ax.scatter3D(ax, ay, az, color='r')
    else:
        ax.scatter3D(ax,ay,az,color='b')
    
plt.title("simple 3D scatter plot")
ax.set_xlabel('P')
ax.set_ylabel('QRS')
ax.set_zlabel('T')

# show plot
plt.show()
 
# import plotly
# import plotly.express as px

# df = data
# fig = plotly.express.scatter_3d(data, x='QRS_duration', y='T_duration', z='P_duration',
#               color='species')
# fig.show()  

# plotly.offline.plot( 
#       fig, filename=r"C:\Users\BeatrizHenriques\OneDrive - Universidade de Aveiro\5 ano\Tese_Beatriz_Henriques\Imagens\3D_Fear.html", auto_open=False)

# In[]:
    
from sklearn.decomposition import PCA

pca = PCA()
x_train = pca.fit_transform(x_train)
explained_variance = pca.explained_variance_ratio_

# Plot explained variance
fig, ax = pca.plot()

# Scatter first 2 PCs
fig, ax = pca.scatter()

# Make biplot with the number of features
fig, ax = pca.biplot(n_feat=4)

# In[]

from statsmodels.multivariate.pca import PCA

pca = PCA(x_res)


def eigen_scaling(pca, scaling = 0):
    # pca is a PCA object obtained from statsmodels.multivariate.pca
    # scaling is one of [0, 1, 2, 3]
    # the eigenvalues of the pca object are n times the ones computed with R
    # we thus need to divide their sum by the number of rows
    const = ((pca.scores.shape[0]-1) * pca.eigenvals.sum()/ pca.scores.shape[0])**0.25
    if scaling == 0:
        scores = pca.scores
        loadings = pca.loadings
    elif scaling == 1:
        scaling_fac = (pca.eigenvals / pca.eigenvals.sum())**0.5
        scaling_fac.index = pca.scores.columns
        scores = pca.scores * scaling_fac * const
        loadings = pca.loadings * const
    elif scaling == 2:
        scaling_fac = (pca.eigenvals / pca.eigenvals.sum())**0.5
        scaling_fac.index = pca.scores.columns
        scores = pca.scores * const
        loadings = pca.loadings * scaling_fac * const
    elif scaling == 3:
        scaling_fac = (pca.eigenvals / pca.eigenvals.sum())**0.25
        scaling_fac.index = pca.scores.columns
        scores = pca.scores * scaling_fac * const
        loadings = pca.loadings * scaling_fac * const
    else:
        sys.exit("Scaling should either be 0, 1, 2 or 3")
    return([scores, loadings])


def biplot(pca, scaling = 0, plot_loading_labels = True, color = None, alpha_scores = 1):
    scores, loadings = eigen_scaling(pca, scaling=scaling)
    # Plot scores
    if color is None:
        scores_ = scores.copy()
        sns.relplot(
            x = "comp_0",
            y = "comp_1",
            palette = "muted",
            alpha = alpha_scores,
            data = scores_,
        )
    else:
        scores_ = scores.copy()
        scores_["group"] = color
        sns.relplot(
            x = "comp_0",
            y = "comp_1",
            hue = "group",
            palette = "muted",
            alpha = alpha_scores,
            data = scores_,
        )

    # Plot loadings
    if plot_loading_labels:
        loading_labels = pca.loadings.index

    for i in range(loadings.shape[0]):
        plt.arrow(
            0, 0,
            loadings.iloc[i, 0],
            loadings.iloc[i, 1],
            color = 'black',
            alpha = 0.7,
            linestyle = '-',
            head_width = loadings.values.max() / 50,
            width = loadings.values.max() / 2000,
            length_includes_head = True
        )
        if plot_loading_labels:
            plt.text(
                loadings.iloc[i, 0]*1.05,
                loadings.iloc[i, 1]*1.05,
                loading_labels[i],
                color = 'black',
                ha = 'center',
                va = 'center',
                fontsize = 10
            );

    # range of the plot
    scores_loadings = np.vstack([scores.values[:, :2], loadings.values[:, :2]])
    xymin = scores_loadings.min(axis=0) * 1.2
    xymax = scores_loadings.max(axis=0) * 1.2

    plt.axhline(y = 0, color = 'k', linestyle = 'dotted', linewidth=0.75)
    plt.axvline(x = 0, color = 'k', linestyle = 'dotted', linewidth=0.75)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.xlim(xymin[0], xymax[0])
    plt.ylim(xymin[1], xymax[1]);
    

# In[[]]

from plotly.offline import plot
import plotly.graph_objs as go
from sklearn.decomposition import PCA


pca = PCA(n_components=6).fit(x_res)
X_reduced = pca.transform(x_res)
trace1 = go.Scatter3d(
    x=X_reduced[:,0],
    y = X_reduced[:,1],
    z = X_reduced[:,2],
    mode='markers',
    marker=dict(
        size=5,
        color= y_res,                
        opacity=1
)

)

dc_1 = go.Scatter3d( x = [0,pca.components_.T[0][0]],
                     y = [0,pca.components_.T[0][1]],
                     z = [0,pca.components_.T[0][2]],
                     marker = dict( size = 0.1,
                                    color = "rgb(84,48,5)"),
                     line = dict( color = "red",
                                width = 6),
                     name = "Var1"
                     )
dc_2 = go.Scatter3d( x = [0,pca.components_.T[1][0]],
                   y = [0,pca.components_.T[1][1]],
                   z = [0,pca.components_.T[1][2]],
                   marker = dict( size = 0.1,
                                  color = "rgb(84,48,5)"),
                   line = dict( color = "green",
                                width = 6),
                   name = "Var2"
                 )
dc_3 = go.Scatter3d( x = [0,pca.components_.T[2][0]],
                     y = [0,pca.components_.T[2][1]],
                     z = [0,pca.components_.T[2][2]],
                     marker = dict( size = .1,
                                  color = "rgb(84,48,5)"),
                     line = dict( color = "blue",
                                width = 6),
                     name = "Var3"
                 ) 
dc_4 = go.Scatter3d( x = [0,pca.components_.T[3][0]],
                     y = [0,pca.components_.T[3][1]],
                     z = [0,pca.components_.T[3][2]],
                     marker = dict( size = .1,
                                  color = "rgb(84,48,5)"),
                     line = dict( color = "yellow",
                                width = 6),
                     name = "Var4"
                   )
dc_5 = go.Scatter3d( x = [0,pca.components_.T[4][0]],
                     y = [0,pca.components_.T[4][1]],
                     z = [0,pca.components_.T[4][2]],
                     marker = dict( size = .1,
                                  color = "rgb(84,48,5)"),
                     line = dict( color = "black",
                                width = 6),
                     name = "Va5"
                   )
dc_6 = go.Scatter3d( x = [0,pca.components_.T[5][0]],
                     y = [0,pca.components_.T[5][1]],
                     z = [0,pca.components_.T[5][2]],
                     marker = dict( size = .1,
                                  color = "rgb(84,48,5)"),
                     line = dict( color = "grey",
                                width = 6),
                     name = "Var6"
                   )
data = [trace1,dc_1,dc_2,dc_3,dc_4,dc_5,dc_6]
layout = go.Layout(
    xaxis=dict(
        title='PC1',
        titlefont=dict(
           family='Courier New, monospace',
           size=18,
           color='#7f7f7f'
       )
   )
)
fig = go.Figure(data=data, layout=layout)
plot(fig, filename='3d-scatter-tupac-with-mac')

# In[]:

df1 = pd.DataFrame()

for index, row in data_3.iterrows():
    if row['Subclass'] == '++++++':
        df1 =df1.append(row)
    if row['Subclass'] == '+++++':
        df1 =df1.append(row)
    if row['Subclass'] == '-------':
        df1 =df1.append(row)
    if row['Subclass'] == '------': 
        df1 =df1.append(row)
    
#    -------


# In[]: 
ext=[]
metrics_train={}
metrics_test={}

test_metrics= pd.DataFrame()
train_metrics = pd.DataFrame()

#Dected nulls and fill with mean value 
for j in frame:  
    ext.append(j['Subclass'].unique()[-1])
    
    for i in j.columns:
        if j[i].isnull().values.any():
            j[i] = j[i].fillna(value=j[i].mean())
            
data = pd.concat(frame, ignore_index=True)

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
j = ['ECG_Rate','QRS_duration', 'P_duration', 'T_duration', 'ST_interval', 'TP_segment', 'PR_interval']
data_4= pd.DataFrame()
data_4=data[['Emotion', 'Class', 'Subclass','ID participant', 'Idx_i']].copy()
for keyo in j:
    data_4[str('Result_'+keyo)] = data[keyo] * data['ECG_Rate']

#Dataset with extremes 
lst=[]
for index, row in data_4.iterrows():
    if row['Subclass'] == ext[0]:
        lst.append(row)
    if row['Subclass'] == ext[1]:
        lst.append(row)

data_5 = pd.DataFrame(lst, columns=data_4.columns)



    
#Data labeling
labels, labels_name, data_5, X, y = label(data_5,'Class')