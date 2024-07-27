

# Healthcare Fraud Detection using Big Data and Deep Anomaly Detection


This project focuses on detecting healthcare fraud using big data analysis techniques. Leveraging the power of CMS big data, the project employs Deep Anomaly Detection methods, along with techniques like SMOTE and PCA for sampling and dimensionality reduction. The ultimate goal is to achieve an accuracy score greater than 85% while minimizing the occurrence of False Positives. 


## Table of Contents

1. [Dataset](#dataset)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)



## Dataset

The dataset can be accessed from the CMS data portal: [CMS Data](https://data.cms.gov)

## Features

- Utilizes CMS big data for healthcare fraud detection.
- Applies Deep Anomaly Detection methods for improved accuracy.
- Uses SMOTE for balancing class distribution.
- Aims for an accuracy score greater than 85% while minimizing False Positives.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/sujaykumarmag/medical-fraud-detection.git
cd medical-fraud-detection
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

- On Windows:

```bash
venv\\Scripts\\activate
```

- On macOS/Linux:

```bash
source venv/bin/activate
```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the following notebook to start the analysis:
- Download the Part-B, Part-D, DMEPOS datasets with full datadescription in a folder of your choice.
- Run the each filename.ipynb in the root folder of the project to analyze the data.
- Make Sure that the input-file-name and directory is matching.
- Usage of Imputation techniques is in the folder `preprocessing/imputation`.
- Run all the Experiments in the model folders `models/` and get the results.

## Project Structure

```
medical-fraud-detection/
├── data/
│   ├── raw/                           # Raw data files and data sources
│   └── processed/                     # Processed data files
│
│  
├── models/
│   ├── Bi-LSTM/                       # Bi-directional LSTM (DL)
│   │   ├── partB.ipynb
│   │   ├── partD.ipynb
│   │   ├── Combined.ipynb         
│   │   └── DMEPOS.ipynb
│   │ 
│   ├── Logistic Regression/           # Logistic Regression (ML)
│   │   ├── partB.ipynb
│   │   ├── partD.ipynb
│   │   ├── Combined.ipynb         
│   │   └── DMEPOS.ipynb
│   │ 
│   ├── LSTM/                          # LSTM model (DL)
│   │   ├── partB.ipynb
│   │   ├── partD.ipynb
│   │   ├── Combined.ipynb         
│   │   └── DMEPOS.ipynb
│   │ 
│   ├── Random Forest/                 # Random Forest (Bagging)
│   │   ├── partB.ipynb
│   │   ├── partD.ipynb
│   │   ├── Combined.ipynb         
│   │   └── DMEPOS.ipynb
│   │ 
│   ├── XGBoost/                       # XGBoost model (Boosting)
│   │   ├── partB.ipynb
│   │   ├── partD.ipynb
│   │   ├── Combined.ipynb         
│   │   └── DMEPOS.ipynb
│   │           
│   └── CatBoost/                       # CatBoost model (Boosting)
│       ├── partB.ipynb
│       ├── partD.ipynb
│       ├── Combined.ipynb         
│       └── DMEPOS.ipynb
│
│   
├── preprocessing/
│   ├── Combined_sampling.ipynb
│   ├── DMEPOS_sampling.ipynb
│   ├── Part_B_sampling.ipynb
│   ├── Part_D_sampling.ipynb
│   │
│   ├── imputation/                    # Imputation for each dataset
│   │   ├── PART-B Full Imputation.ipynb
│   │   ├── PART-D Full Imputation.ipynb         
│   │   └── DMEPOS Full Imputation.ipynb
│   │           
│   └── features/                       # Feature Sampling from the previous research papers
│       ├── Combined_features_sampling.ipynb
│       ├── DMEPOS_features_sampling.ipynb
│       ├── Part_B_features_sampling.ipynb 
│       └── Part_D_features_sampling.ipynb 
│
│
├── Combined FULL dataset.ipynb
│
├── Combined_Feature_Dataset.ipynb
│
├── Fraud_Labelling_DMEPOS.ipynb
│
├── Fraud_Labelling_Part-B.ipynb
│
├── Fraud_Labelling_Part-D.ipynb 
│
├── requirements.txt                    # Python dependencies
│   
├── README.md
│  
└── Book1.xlsx                 
```
