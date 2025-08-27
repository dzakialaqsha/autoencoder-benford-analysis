# Combined Autoencoder and Benford's First Digit Analysis Approach for Anomaly Detection in Accounting Journal Entry Data

## Project Participants
| Student ID     | Name                             | University                         | Department | Role                      |
|----------------|--------------------------------- |------------------------------------|------------| ------------------------- |
| 12030121120034 | Muhammad Dzaki Al Aqsha          | Universitas Diponegoro             | Accounting | First Author              |
|                | Adi Firman Ramadhan, S.E., M.Ak. | Universitas Diponegoro             | Accounting | Second Author, Supervisor |

## Project Overview
This project was developed as part of an **undergraduate thesis (skripsi)** to fulfill the graduation requirements for a bachelor's degree. The research aims to develop and introduce a novel anomaly detection procedure on accounting journal entry data by combining autoencoder and Benford's first digit analysis.

This project is greatly inspired by the [work](https://github.com/GitiHubi/deepAI) of Marco Schreyer and Timur Sattarov, as well as the works of Mark Nigrini.

## Approach Overview
### Baseline Autoencoder Implementation
![Baseline Implementation](images/Skripsi_Viz_1.png)

### Baseline Benford's First Digit Analysis for Anomaly Detection
![Baseline Implementation](images/Skripsi_Viz_2.png)

### Proposed Combined Method
![Proposed Method](images/Skripsi_Viz_3.png)

## Result Overview
| Anomaly Detection Method | Sub-Sample Method | Sub-Sample Size | Accuracy | Precision | Recall | F1-Score |
| ------------------------ | ----------------- | --------------- | ----------- | ----------- | ----------- | ----------- |
| Autoencoder | Distribution Preserving Stratified Random Sampling | 100,042 entries | 99.96% | 93.33% | 80.46% | 86.42% |
| Benford's Analysis | Full Data | 533,009 entries | 99.99% | 100% | 70% | 82.35% |
| Combined Approach | Distribution Preserving Stratified Random Sampling | 100,042 entries | 99.96% | 92.72% | 80.46% | 86.15% |
