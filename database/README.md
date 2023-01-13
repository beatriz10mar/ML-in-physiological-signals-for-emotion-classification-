# EMOTE Database

The EMOTE is the acronym for Emotion MultimOdal daTabasE where “emote” stands for “give emotion to, in a stage or movie role”. This dataset is composed of files with simultaneous physiological signals (EMG MF, EMG TR, EDA, and ECG, 1000 Hz of sampling rate) collected from 57 emotionally stimulated participants in two sessions using movie excerpts.

This repository contains three folders and a .csv file. The .csv presents the list of participants and the name of the corresponding files. Each folder corresponds to one participant, and it is named with the participant code. In each folder there is two .txt files, one for each experimental session. The .txt file contains the experimental data of the corresponding participant in a given session.

All the files are organized with the same structure, as illustrated in the figure below. The header is presented in the first three lines, where it shows the general information about the experimental session, such as the name of the device, date of the collection, sampling rate, number of channels, sensors, and colour sleeve used. The remaining file is divided into six columns. The second column corresponds to the trigger and is composed of “1” and “0”. The 1 indicates that the trigger is activated and the 0 is deactivated. The third, fourth, fifth, and sixth column presents the data respectively for the physiological signals EMG MF, EMG TR, EDA, and ECG.

![image](https://user-images.githubusercontent.com/95349173/211217788-a88cee84-b64f-4325-9a57-f84141226022.png)

## **Important nome**:
The presented data do not consist of the true experimental data collected for this work, only random values with the same structure as the real data so that the code is reproducible.
