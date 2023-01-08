# Machine-learning-in-physiological-signals-for-emotion-classification-

This repository was created in the context of the thesis "Machine Learning in physiological signals for emotional classification" and it aimed the development of a machine learning algorithm, more specirficaly a SVM model, that identify emotions (Fear, Happy, Neutral) and their level of intensity, through the information of the physiological signals ECG of a subject. It was developed at the University of Aveiro (UA), within the scope of the project Institute of Electronic Engineering and Informatics of Aveiro (UIDB/00127/2020).

## **Repository Organization:**
This repository contains several Python scripts containing the analysis of these time series. The Python scripts are divided by the sections of the construction of the model in accordance with the dissertation. The different **sections** carried out throughout the work are:
- Preprocessing
- HR Exploratory analysis
- Classification by SVM

## **Algorithm performance**
The machine learning model was developed based on Support Vector Machine combination of data resampling techniques, an ensemble of models and data normalization. The emotional model was aoolied separately to each emotion to classify the inner classes and subclasses. The following performance for the Balanced Accuracy were obtained:
- Fear: 96.20% (training) and 85.05% (test);
- Happy: 99.66% (training) and 84.59% (test);
- Neutral: 95.23% (training) and 78.20% (test).

## **More information:**
The **master's thesis** can be consulted at xxxxxx.

## **Contacts:**
For more information about the database and protocol please contact the supervisores of the dissertation:
- **Sónia Gouveia**, PhD Researcher, Institute of Electronics and Informatics Engineering of Aveiro, University of Aveiro, sonia.gouveia@ua.pt.
- **Susana Brás**, PhD Researcher, Institute of Electronics and Informatics Engineering of Aveiro, University of Aveiro, susana.bras@ua.pt.

For more information about the algoritm and dissertation please contact the author of the dissertation:
- **Beatriz Henriques**, Research Fellow, Institute of Electronics and Informatics Engineering of Aveiro, University of Aveiro, beatriz.henriques@ua.pt

## **Version:**
Version 1.0 (december 2022)
