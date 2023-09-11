# Breast_density_estimation_in_MRI
This repository contained the code to reproduce trained classification models reported in "Automated Breast Density Assessment in MRI using Deep Learning and Radiomics: Strategies for Reducing Inter-observer Variability"

## Image preprocessing
Deep learning and Radiomics models were developmed with segmented breasts. For breast segmentation in T1w MRI please refer to "J. Zhang, A. Saha, Z. Zhu and M. A. Mazurowski, "Hierarchical Convolutional Neural Networks for Segmentation of 
Breast Tumors in MRI With Application to Radiogenomics," in IEEE Transactions on Medical Imaging, vol. 38, no. 2, pp. 435-447, Feb. 2019, doi: 10.1109/TMI.2018.2865671."
## Radiomics feature
Radiomics feature extraction was performed with pyradiomics. The extraction pipe line could be found at https://github.com/Astarakee/Radiomics_pipeline.

## Deep learning models
The deep learning models could be found at https://drive.google.com/drive/folders/1TZ4iygIOGw-gSCM3Qi8QQlD0qEwlAG0y?usp=share_link.
Please dowonlad the deep learning models and place under ./models/
