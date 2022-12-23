This section aims the analysis and exploration of the EMOTE database to prepare the data for an emotional model that is capable of classifying the emotions given information contained in the physiological signals. 

The pipeline is divided in the following stages:

- **Reading**: import the scrip Reading.py to read physiological signals and segment the data in conditions
- **Features Extraction**: preprocess the data and extraction of the features (Neurokit_Function.py) of interest in the ECG signal, which were organized in a dataframe. 
- **Stationarity Analysis**: importat the scrip Stationarity.py to scan of the previous dataframe into the ADF and KPSS stationarity test and analyse the results concordance.
- **HR Exploratory Analysis**: import the scrip Exploratory.py to conduct an exhaustive exploratory analysis on the HR feature of the ECG to evaluate the HR progression during the emotional procedure (by the average group response) and to extract the individual responses. 
- **Feature Selection**: selection of the features with higher discriminatio to feed the model.


