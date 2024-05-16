#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import linear_model, tree, ensemble


# In[2]:


dataframe=pd.read_csv("heart_disease_data.csv")


# In[3]:


dataframe.head(10)


# In[4]:


dataset = pd.get_dummies(dataframe, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


# In[5]:


standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


# In[6]:


y = dataframe['target']
X = dataframe.drop(['target'], axis = 1)


# In[7]:


X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=40)


# In[8]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=500,criterion='entropy',max_depth=8,min_samples_split=5)
RF_Model = rfc.fit(X_train, y_train)
prediction3 = RF_Model.predict(X_test)


# In[9]:


from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=1.0, class_weight='balanced', dual=False,fit_intercept=True, intercept_scaling=1, l1_ratio=None,max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',random_state=1234, solver='lbfgs', tol=0.0001, verbose=0,warm_start=False)
LR_Model=lr.fit(X_train,y_train)
prediction1=LR_Model.predict(X_test)


# In[11]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(max_depth=5,criterion='entropy')
cv_scores = cross_val_score(tree_model, X, y, cv=10, scoring='accuracy')
DT_Model=tree_model.fit(X, y)
prediction4=DT_Model.predict(X_test)


# In[36]:


# import tkinter as tk
# from tkinter import messagebox
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier  # Assuming you have RF_Model
# models = {
#     "Logistic Regression": LR_Model,
#     "Decision Tree": DT_Model,
#     "Random Forest": RF_Model,
# }

# def predict_heart_disease():
#     try:
#         # Get user input
#         age = int(age_entry.get())
#         sex = int(sex_entry.get())
#         cp = int(cp_entry.get())
#         trestbps = int(trestbps_entry.get())
#         chol = int(chol_entry.get())
#         fbs = int(fbs_entry.get())
#         restecg = int(restecg_entry.get())
#         thalach = int(thalach_entry.get())
#         exang = int(exang_entry.get())
#         oldpeak = float(oldpeak_entry.get())
#         slope = int(slope_entry.get())
#         ca = int(ca_entry.get())
#         thal = int(thal_entry.get())

#         # Create and reshape input array
#         input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
#         input_reshaped = input_data.reshape(1, -1)

#         # Make prediction
#         prediction = RF_Model.predict(input_reshaped)[0]

#         if prediction == 1:
#             message = "The patient seems to be at risk of heart disease. Please consult a doctor for further evaluation."
#         else:
#             message = "The prediction suggests a lower risk of heart disease. However, a healthy lifestyle is always recommended."

#         messagebox.showinfo("Prediction Result", message)

#     except ValueError:
#         messagebox.showerror("Error", "Please enter valid numerical values.")

# # Create the GUI window
# root = tk.Tk()
# root.title("Heart Disease Prediction")

# # Labels and entry fields
# age_label = tk.Label(root, text="Age:")
# age_label.grid(row=0, column=0, pady=5)
# age_entry = tk.Entry(root, width=10)
# age_entry.grid(row=0, column=1, pady=5)

# sex_label = tk.Label(root, text="Sex (1-Male, 0-Female):")
# sex_label.grid(row=1, column=0, pady=5)
# sex_entry = tk.Entry(root, width=10)
# sex_entry.grid(row=1, column=1, pady=5)

# cp_label = tk.Label(root, text="Chest pain type (refer to model documentation):")
# cp_label.grid(row=2, column=0, pady=5)
# cp_entry = tk.Entry(root, width=10)
# cp_entry.grid(row=2, column=1, pady=5)
# # ... continue from previous code ...

# trestbps_label = tk.Label(root, text="Resting blood pressure (in mm Hg):")
# trestbps_label.grid(row=3, column=0, pady=5)
# trestbps_entry = tk.Entry(root, width=10)
# trestbps_entry.grid(row=3, column=1, pady=5)

# chol_label = tk.Label(root, text="Serum cholesterol in mg/dl:")
# chol_label.grid(row=4, column=0, pady=5)
# chol_entry = tk.Entry(root, width=10)
# chol_entry.grid(row=4, column=1, pady=5)

# fbs_label = tk.Label(root, text="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false):")
# fbs_label.grid(row=5, column=0, pady=5)
# fbs_entry = tk.Entry(root, width=10)
# fbs_entry.grid(row=5, column=1, pady=5)

# restecg_label = tk.Label(root, text="Resting electrocardiographic results (refer to model documentation):")
# restecg_label.grid(row=6, column=0, pady=5)
# restecg_entry = tk.Entry(root, width=10)
# restecg_entry.grid(row=6, column=1, pady=5)

# thalach_label = tk.Label(root, text="Maximum heart rate achieved:")
# thalach_label.grid(row=7, column=0, pady=5)
# thalach_entry = tk.Entry(root, width=10)
# thalach_entry.grid(row=7, column=1, pady=5)

# exang_label = tk.Label(root, text="Exercise induced angina (1 = yes, 0 = no):")
# exang_label.grid(row=8, column=0, pady=5)
# exang_entry = tk.Entry(root, width=10)
# exang_entry.grid(row=8, column=1, pady=5)

# oldpeak_label = tk.Label(root, text="ST depression induced by exercise relative to rest:")
# oldpeak_label.grid(row=9, column=0, pady=5)
# oldpeak_entry = tk.Entry(root, width=10)
# oldpeak_entry.grid(row=9, column=1, pady=5)

# slope_label = tk.Label(root, text="The slope of the peak exercise ST segment (refer to model documentation):")
# slope_label.grid(row=10, column=0, pady=5)
# slope_entry = tk.Entry(root, width=10)
# slope_entry.grid(row=10, column=1, pady=5)

# ca_label = tk.Label(root, text="Number of major vessels (0-3) colored by fluoroscope:")
# ca_label.grid(row=11, column=0, pady=5)
# ca_entry = tk.Entry(root, width=10)
# ca_entry.grid(row=11, column=1, pady=5)

# thal_label = tk.Label(root, text="Thalium stress test results (refer to model documentation):")
# thal_label.grid(row=12, column=0, pady=5)
# thal_entry = tk.Entry(root, width=10)
# thal_entry.grid(row=12, column=1, pady=5)

# # ... rest of the code (predict button and root.mainloop())

# # ... Add labels and entry fields for all remaining features (trestbps, chol, fbs, etc.)

# # Predict button
# predict_button = tk.Button(root, text="Predict", command=predict_heart_disease)
# predict_button.grid(row=13, columnspan=2, pady=10)

# # Run the GUI
# root.mainloop()


# In[37]:


import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from tkinter import ttk  # for combobox (dropdown list)

# Dummy training data - Replace with your actual data
X_train, y_train = np.random.rand(100, 13), np.random.randint(0, 2, 100)

# Define the models
models = {
    "Logistic Regression": LogisticRegression(C=1.0, class_weight='balanced', dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='auto', n_jobs=None, penalty='l2', random_state=1234, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, criterion='entropy'),
    "Random Forest": RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=8, min_samples_split=5)
}

# Train the models (assuming you have training data)
LR_Model = models["Logistic Regression"].fit(X_train, y_train)
DT_Model = models["Decision Tree"].fit(X_train, y_train)
RF_Model = models["Random Forest"].fit(X_train, y_train)

# Global variable to store selected model
selected_model = RF_Model  # Default model

def select_model(event):
    global selected_model
    selected_model = models[model_combo.get()]

def predict_heart_disease():
    try:
        # Get user input
        age = int(age_entry.get())
        sex = int(sex_entry.get())
        cp = int(cp_entry.get())
        trestbps = int(trestbps_entry.get())
        chol = int(chol_entry.get())
        fbs = int(fbs_entry.get())
        restecg = int(restecg_entry.get())
        thalach = int(thalach_entry.get())
        exang = int(exang_entry.get())
        oldpeak = float(oldpeak_entry.get())
        slope = int(slope_entry.get())
        ca = int(ca_entry.get())
        thal = int(thal_entry.get())

        # Create and reshape input array
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        input_reshaped = input_data.reshape(1, -1)

        # Make prediction using the selected model
        prediction = selected_model.predict(input_reshaped)[0]

        if prediction == 1:
            message = "The patient seems to be at risk of heart disease. Please consult a doctor for further evaluation."
        else:
            message = "The prediction suggests a lower risk of heart disease. However, a healthy lifestyle is always recommended."

        messagebox.showinfo("Prediction Result", message)

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values.")

# Create the GUI window
root = tk.Tk()
root.title("Heart Disease Prediction")

# ComboBox for selecting the model
model_label = ttk.Label(root, text="Select Model:")
model_label.grid(row=0, column=2, padx=10, pady=5)
model_combo = ttk.Combobox(root, values=list(models.keys()), state="readonly")
model_combo.grid(row=0, column=3, padx=10, pady=5)
model_combo.current(2)  # Default to Random Forest
model_combo.bind("<<ComboboxSelected>>", select_model)

# Labels and entry fields
age_label = tk.Label(root, text="Age:")
age_label.grid(row=1, column=0, pady=5)
age_entry = tk.Entry(root, width=10)
age_entry.grid(row=1, column=1, pady=5)

sex_label = tk.Label(root, text="Sex (1-Male, 0-Female):")
sex_label.grid(row=2, column=0, pady=5)
sex_entry = tk.Entry(root, width=10)
sex_entry.grid(row=2, column=1, pady=5)

cp_label = tk.Label(root, text="Chest pain type (refer to model documentation):")
cp_label.grid(row=3, column=0, pady=5)
cp_entry = tk.Entry(root, width=10)
cp_entry.grid(row=3, column=1, pady=5)

trestbps_label = tk.Label(root, text="Resting blood pressure (in mm Hg):")
trestbps_label.grid(row=4, column=0, pady=5)
trestbps_entry = tk.Entry(root, width=10)
trestbps_entry.grid(row=4, column=1, pady=5)

chol_label = tk.Label(root, text="Serum cholesterol in mg/dl:")
chol_label.grid(row=5, column=0, pady=5)
chol_entry = tk.Entry(root, width=10)
chol_entry.grid(row=5, column=1, pady=5)

fbs_label = tk.Label(root, text="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false):")
fbs_label.grid(row=6, column=0, pady=5)
fbs_entry = tk.Entry(root, width=10)
fbs_entry.grid(row=6, column=1, pady=5)

restecg_label = tk.Label(root, text="Resting electrocardiographic results (refer to model documentation):")
restecg_label.grid(row=7, column=0, pady=5)
restecg_entry = tk.Entry(root, width=10)
restecg_entry.grid(row=7, column=1, pady=5)

thalach_label = tk.Label(root, text="Maximum heart rate achieved:")
thalach_label.grid(row=8, column=0, pady=5)
thalach_entry = tk.Entry(root, width=10)
thalach_entry.grid(row=8, column=1, pady=5)

exang_label = tk.Label(root, text="Exercise induced angina (1 = yes, 0 = no):")
exang_label.grid(row=9, column=0, pady=5)
exang_entry = tk.Entry(root, width=10)
exang_entry.grid(row=9, column=1, pady=5)

oldpeak_label = tk.Label(root, text="ST depression induced by exercise relative to rest:")
oldpeak_label.grid(row=10, column=0, pady=5)
oldpeak_entry = tk.Entry(root, width=10)
oldpeak_entry.grid(row=10, column=1, pady=5)

slope_label = tk.Label(root, text="The slope of the peak exercise ST segment (refer to model documentation):")
slope_label.grid(row=11, column=0, pady=5)
slope_entry = tk.Entry(root, width=10)
slope_entry.grid(row=11, column=1, pady=5)

ca_label = tk.Label(root, text="Number of major vessels (0-3) colored by fluoroscope:")
ca_label.grid(row=12, column=0, pady=5)
ca_entry = tk.Entry(root, width=10)
ca_entry.grid(row=12, column=1, pady=5)

thal_label = tk.Label(root, text="Thalium stress test results (refer to model documentation):")
thal_label.grid(row=13, column=0, pady=5)
thal_entry = tk.Entry(root, width=10)
thal_entry.grid(row=13, column=1, pady=5)


# Continue adding other labels and entries...

# Predict button
predict_button = tk.Button(root, text="Predict", command=predict_heart_disease)
predict_button.grid(row=14, columnspan=4, pady=10)

# Run the GUI
root.mainloop()


# In[ ]:




