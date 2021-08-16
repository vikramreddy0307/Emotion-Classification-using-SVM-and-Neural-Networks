# Emotion-Classification-using-SVM-and-Neural-Networks

This repository contains two different approaches
1. Facial Emotion classification using Linear SVM and Landmark detector
2. Facial Emotion classification using CNN


## Linear SVM and Landmark detector

1. Data Used: Cohn-Kanade Images
2. Landmark Extraction : CV2 Landmark Detector
3. Model Used for training : Support Vector Machine


First Images are passed to CV2 landmark detector to extract the landmarks from the face. 
Then these landmarks extracted from the images are passed to SUpport vector machine. 
The image is classified in to  one of the following labels ['anger' 'contempt' 'disgust' 'fear' 'happy' 'sadness' 'surprise']

Grid Search Parameters used:
C, Gamma, Kernel

Optimal Parameters obtained are:




## CNN

1. Data Used: Cohn-Kanade Images
2. Frontal Face Extraction : CV2 Face Detector
3. Model Used for training : CNN


Data size : 3017, Input size : (60, 60, 3)

First Images are passed to CV2 Face detector to extract the frontal face 
Then these  cropped facial images are passed to Support vector machine and the small sized cropped faces are reshaped/interpolated. 
The image is classified in to  one of the following labels ['anger' 'contempt' 'disgust' 'fear' 'happy' 'sadness' 'surprise']


1. Weight Initalizer : Xavier/Glorot
2. Loss: Categorical class entropy since it is Multi Class Classification
3. Optimizer : Adam

Trained for 10 Epochs
![Alt text](download.png)



