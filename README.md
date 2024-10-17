# Heart Disease Prediction

This project aims to develop a machine learning model to predict the likelihood of heart disease in patients using the Scikit-Learn library. The model utilizes various health metrics and patient data to provide insights and predictions.

## GUI

Below is a screenshot of the graphical user interface (GUI):

![GUI Screenshot](images/gui.png)

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)

## Project Overview

Heart disease is one of the leading causes of death worldwide. Early detection and prediction of heart disease can significantly reduce mortality rates. This project uses a dataset to train a machine learning model that predicts the likelihood of heart disease based on various features such as age, cholesterol levels, and blood pressure.

## Technologies Used

- Python
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Dataset

The dataset used for this project is the **UCI Heart Disease Dataset**, which can be found [here](https://archive.ics.uci.edu/ml/datasets/heart+Disease). The dataset consists of various attributes, including:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- Resting Electrocardiographic Results
- Maximum Heart Rate Achieved
- Exercise Induced Angina
- Oldpeak
- Slope of the Peak Exercise ST Segment
- Number of Major Vessels
- Thalassemia
- Target (Heart Disease Presence)

## Installation

To run this project, ensure you have Python installed on your machine. You can create a virtual environment and install the required libraries as follows:

```bash
# Clone the repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
