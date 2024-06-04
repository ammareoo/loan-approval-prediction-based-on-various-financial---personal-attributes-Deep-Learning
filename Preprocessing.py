#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_json('/content/loan_approval_dataset.json')
df = df.drop(['Id'],axis=1)
df.head()


# In[ ]:


print("df.shape:" , df.shape, '\n')

print("df.info():" , df.info(), '\n')

print("df.describe():" , df.describe(), '\n')

print("df.isna().sum():" , df.isna().sum(), '\n')


# # Categorial

# In[ ]:


categorical_columns = ['Risk_Flag', 'Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', 'STATE']

for column in categorical_columns:
    unique_values = df[column].unique()
    print(f"{len(unique_values)} Unique values for {column}: {'' if len(unique_values) > 5 else unique_values}")


# In[ ]:


loan3 = df['Risk_Flag'].replace({0: 'Yes', 1: 'No'}).value_counts()
plt.figure()
plt.pie(loan3.values, labels=loan3.index,
        startangle=90, autopct="%1.2f%%",
        labeldistance=None, textprops={'fontsize': 12}, shadow=True, explode=[0, 0.12])
plt.legend()
plt.title('Loan Approval through Behviour', fontsize=14)

plt.tight_layout()
plt.show()


# In[ ]:


# Set up the figure layout
plt.figure(figsize=(18, 12))

# Plot pie chart for Married/Single
plt.subplot(3, 1, 1)
df['Married/Single'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Marital Status')

# Plot pie chart for House_Ownership
plt.subplot(3, 1, 2)
df['House_Ownership'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of House Ownership')

# Plot pie chart for Car_Ownership
plt.subplot(3, 1, 3)
df['Car_Ownership'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Car Ownership')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()


# In[ ]:


# categorical_columns = ['Risk_Flag', 'Married/Single', 'House_Ownership', 'Car_Ownership']

# df = pd.get_dummies(df, columns=categorical_columns)

for column in ['Married/Single', 'House_Ownership', 'Car_Ownership']:
    # Calculate the frequency of each category
    frequency_map = df[column].value_counts(normalize=True)
    # Replace categories with their corresponding frequencies
    df[f'{column}'] = df[column].map(frequency_map)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure layout
plt.figure(figsize=(18, 12))

# Plot distribution for Profession
plt.subplot(3, 1, 1)
sns.countplot(x='Profession', data=df, order = df['Profession'].value_counts().index)
plt.title('Distribution of Profession')
plt.xlabel('Profession')
plt.ylabel('Count')
plt.xticks(rotation=90)

# Plot distribution for CITY
plt.subplot(3, 1, 2)
sns.countplot(x='CITY', data=df, order = df['CITY'].value_counts().index)
plt.title('Distribution of CITY')
plt.xlabel('CITY')
plt.ylabel('Count')
plt.xticks(rotation=90)

# Plot distribution for STATE
plt.subplot(3, 1, 3)
sns.countplot(x='STATE', data=df, order = df['STATE'].value_counts().index)
plt.title('Distribution of STATE')
plt.xlabel('STATE')
plt.ylabel('Count')
plt.xticks(rotation=90)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()


# In[ ]:


for column in ['Profession', 'CITY', 'STATE']:
    # Calculate the frequency of each category
    frequency_map = df[column].value_counts(normalize=True)
    # Replace categories with their corresponding frequencies
    df[f'{column}'] = df[column].map(frequency_map)


# In[ ]:


df.head()


# # Numeric

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

#(Loan Approval through Behaviour)
correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap (Loan Approval through Behaviour)')
plt.show()


# In[ ]:


numeric_columns = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure layout
plt.figure(figsize=(18, 12))

# Define numeric columns
numeric_columns = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']

# Plot histograms for each numeric column
for i, column in enumerate(numeric_columns, start=1):
    plt.subplot(3, 2, i)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
numeric_columns = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']

scaler = MinMaxScaler()

df[numeric_columns] = scaler.fit_transform(df[numeric_columns])


# 

# In[ ]:


df.head()


# In[ ]:


df.to_csv('loan_approval_dataset_preprocessed.csv', index=False)


# In[ ]:




