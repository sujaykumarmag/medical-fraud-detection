#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


df = pd.read_csv("../../labelled_data/part_B/features/partb_features_data.csv")


# In[3]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Rndrng_Prvdr_Type'] = label_encoder.fit_transform(df['Rndrng_Prvdr_Type'])


# In[4]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Rndrng_Prvdr_Gndr'] = label_encoder.fit_transform(df['Rndrng_Prvdr_Gndr'])


# In[5]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['FraudType'] = label_encoder.fit_transform(df['FraudType'])


# In[6]:


df = df.drop(["Unnamed: 0","Rndrng_NPI"],axis=1)


# In[7]:


df


# In[8]:


X = df.drop(['Fraud', 'FraudType'], axis=1)
y = df['Fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CatBoost Classifier
model = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.05, loss_function='MultiClass', random_seed=42)

# Train the model
model.fit(X_train, y_train, cat_features=None)  # You can specify categorical features if needed

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\n----------------------------------- PART B Features ------------------------------\n")
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)


# In[9]:


df = pd.read_csv("../../labelled_data/part_B/features/partb_rus_features.csv")


# In[10]:


df = df.drop(["Unnamed: 0","Rndrng_NPI","Unnamed: 0.1"],axis=1)


# In[11]:


df


# In[12]:


X = df.drop(['Fraud'], axis=1)
y = df['Fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CatBoost Classifier
model = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.05, loss_function='MultiClass', random_seed=42)

# Train the model
model.fit(X_train, y_train, cat_features=None)  # You can specify categorical features if needed

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\n----------------------------------- PART B RUS Features ------------------------------\n")

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)


# In[13]:


df = pd.read_csv("../../labelled_data/part_B/features/partb_ros_features.csv")


# In[14]:


df = df.drop(["Unnamed: 0","Rndrng_NPI","Unnamed: 0.1"],axis=1)


# In[15]:


df


# In[16]:


X = df.drop(['Fraud'], axis=1)
y = df['Fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CatBoost Classifier
model = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.05, loss_function='MultiClass', random_seed=42)

# Train the model
model.fit(X_train, y_train, cat_features=None)  # You can specify categorical features if needed

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\n----------------------------------- PART B ROS Features ------------------------------\n")

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)


# In[17]:


df = pd.read_csv("../../labelled_data/part_B/features/partb_smote_features.csv")


# In[18]:


df = df.drop(["Unnamed: 0","Rndrng_NPI","Unnamed: 0.1"],axis=1)


# In[19]:


df


# In[20]:


X = df.drop(['Fraud'], axis=1)
y = df['Fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CatBoost Classifier
model = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.05, loss_function='MultiClass', random_seed=42)

# Train the model
model.fit(X_train, y_train, cat_features=None)  # You can specify categorical features if needed

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\n----------------------------------- PART B SMOTE Features ------------------------------\n")
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)


# In[21]:


df = pd.read_csv("../../labelled_data/part_B/features/partb_rus_type_features.csv")


# In[22]:


df = df.drop(["Unnamed: 0","Rndrng_NPI","Unnamed: 0.1"],axis=1)


# In[23]:


df


# In[24]:


X = df.drop(['FraudType'], axis=1)
y = df['FraudType']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CatBoost Classifier
model = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.05, loss_function='MultiClass', random_seed=42)

# Train the model
model.fit(X_train, y_train, cat_features=None)  # You can specify categorical features if needed

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\n----------------------------------- PART B RUS Type Features ------------------------------\n")

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)


# In[25]:


df = pd.read_csv("../../labelled_data/part_B/features/partb_ros_type_features.csv")


# In[26]:


df = df.drop(["Unnamed: 0","Rndrng_NPI","Unnamed: 0.1"],axis=1)


# In[27]:


df


# In[ ]:


X = df.drop(['FraudType'], axis=1)
y = df['FraudType']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CatBoost Classifier
model = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.05, loss_function='MultiClass', random_seed=42)

# Train the model
model.fit(X_train, y_train, cat_features=None)  # You can specify categorical features if needed

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print("\n----------------------------------- PART B ROS Type Features ------------------------------\n")
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)


# In[ ]:




