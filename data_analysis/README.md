# Data Analysis

This section contains all the code made for the analysis of the data. The aim is to analyse the physiological data collected in order to extrcat the information more relevant and that best describes the emotional states stimulated to introduce in the emotional model.

##  Organization:

- Preprocessing: read the physiological signals and verification of the data quality;
- Features Extraction: extract the features of interest in the ECG signal, which were organized in a dataframe;
- Stationarity Analysis: scan of the previous dataset into the ADF and KPSS stationarity test;
- HR Exploratory Analysis: exploratory analysis on the heart rate (HR) feature of the ECG to evaluate the HR progression during the emotional procedure;
- Feature Selection: select the features with higher discriminationto feed the model in the section.



## **Packages**:

It is necessary to install the packages and libraries:
- pandas (1.4.2);
- numpy (1.20.3);
- statistics;
- seaborn (0.11.2);
- statsmodels (0.12.2);
- scikit-learn (1.0.2);
- scipy (1.7.3);
- matplotlib (3.5.1);
- neurokit2 (0.2.1)<sup>1</sup> ; 
- plotly (5.6.0)<sup>2</sup> ; 
- imbalanced-learn (0.7.0)<sup>3</sup>. 

1. https://neuropsychology.github.io/NeuroKit/installation.ht
2. https://plotly.com/python/getting-started/#installation
3. https://imbalanced-learn.org/stable/




