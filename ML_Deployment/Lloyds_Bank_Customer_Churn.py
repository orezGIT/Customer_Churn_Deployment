
#%%
import pandas as pd
import matplotlib.pyplot as plt
from summarytools import dfSummary
import numpy as np 
import seaborn as sns


#%%
# load the first sheet 
file = r"C:\Users\Orezime Isaac\Downloads\Customer_Churn_Data_Large.xlsx"
df1 = pd.read_excel(file, sheet_name='Transaction_History') 
df1

#%%
# load the second sheet 
df2 = pd.read_excel(file, sheet_name='Customer_Service')
df2

#%%
# load the third sheet 
df3 = pd.read_excel(file, sheet_name='Online_Activity')
df3

#%%
# load the third sheet 
df4 = pd.read_excel(file, sheet_name='Churn_Status')
df4 


# Merged all four dataframes
merged_df = df1.merge(df2, on='CustomerID', how='left')\
                .merge(df3, on='CustomerID', how='left')\
                .merge(df4, on='CustomerID', how='left')
                
# Display final dataframe
merged_df


#%%
dfSummary(merged_df)

# In[ ]:

# NOTE: Visualising Numerical Variables

# create a copy from the original data frame 
merged_copy = merged_df.copy()

# implement feature engineering by creating a new column for 'Days_Since_last_Transaction'
merged_copy['Days_Since_Last_Transaction'] = (pd.to_datetime('today') - pd.to_datetime(merged_copy['TransactionDate'])).dt.days

# implement feature engineering by creating a new column for 'Days_Since_last_Interaction'
merged_copy['Days_Since_last_Interaction'] = (pd.to_datetime('today') - pd.to_datetime(merged_copy['InteractionDate'])).dt.days

# implement feature engineering by creating a new column for 'Days_Since_last_Login'
merged_copy['Days_Since_Last_Login'] = (pd.to_datetime('today') - pd.to_datetime(merged_copy['LastLoginDate'])).dt.days

merged_copy


# In[29]:

# Extract relevant numeric variables  
num_df = merged_copy.select_dtypes(include=['int64', 'float64'])

# Drop specific columns that is not required
num_df = num_df.drop(columns=['TransactionDate', 'InteractionDate', 'LastLoginDate'], errors='ignore')

num_df.columns


# In[30]:

# calculate the correlation matrix
corr = num_df.corr()

# Sort the correlated values 
sort_corr = corr.sort_values(by='ChurnStatus', ascending=False)

# show the correlation heatmap of the numerical variable to the target variable (ChurnStatus)
plt.figure(figsize=(10, 4))
sns.heatmap(sort_corr, annot=True, cmap='coolwarm', linewidth=0.5)
plt.show()


# In[31]:

fig, axs = plt.subplots(3, 3, figsize=(8,8))

# Flatten the 2D array to 1D array for easy iteration
axs = axs.flatten()

#loop through the subpolots
for i, col in enumerate(num_df.columns): 
    axs[i].boxplot(num_df[col])
    axs[i].set_title(f'{col}')
    
plt.tight_layout()      
plt.show()
    

# In[ ]:
# Note: Visualising Categorical Variables

# convert the ChurnStatus into object data type 
merged_copy['ChurnStatus'] = merged_copy['ChurnStatus'].astype('object')

# Extract only categorical variables  
cat_df = merged_copy.select_dtypes(include='object').columns

# loop through the categorical varables to determine 'ChurnStatus' distribution
for col in cat_df: 
    plt.figure(figsize=(8,4))
    sns.countplot(x=col, hue='ChurnStatus', data=merged_df) 
    plt.title(f'Churn distribution by {col}') 
    plt.xticks(rotation=45) 
    plt.show()


# In[ ]:

# NOTE: Hypotheses for Chi-Square Test
# 
# Null Hypothesis (H0):
# There is no association between the predictor and ChurnStatus (i.e., they are independent)

# Alternative Hypothesis (H1):  
# There is an association between the predictor and ChurnStatus (i.e., they are dependent)


# carrying out test to determine the association between predictors and the response variables 
from scipy.stats import chi2_contingency

for col in cat_df: 
    if col != 'ChurnStatus': 
        #create a contingency table 
        contigency_table = pd.crosstab(merged_copy[col], merged_copy['ChurnStatus'])

        # Apply Chi-Square test 
        chi2, p, dof, ex = chi2_contingency(contigency_table)

        # Display result 
        print(f"Chi-Square Test for '{col}' vs ChurnStatus") 
        print(f'Chi2 Result: {chi2:.2f}, p-value: {p:.4f}') 
        if p<0.05: 
            print(f"There is a significant association (Reject H0)")

        else: 
             print(f"No significant association (Fail to Reject H0)\n")



# In[34]:

# overwrite 'merged_df' with 'merged_copy' 
merged_df = merged_copy.copy()

merged_df = merged_df.drop(['CustomerID', 'TransactionID', 'TransactionDate', 'InteractionID',
                            'InteractionDate', 'InteractionType', 'ResolutionStatus', 'LastLoginDate'], axis=1) 

merged_df                              


# In[ ]:

# NOTE: Data Preparatation / Cleaning 


dfSummary(merged_df)


# In[36]:

# Checking rows with Nan values
merged_df.head(40)


# In[37]:


# convert target column to numerical data type
merged_df['ChurnStatus']= merged_df['ChurnStatus'].astype('int')
merged_df.dtypes


# In[38]:


# import the relevant library
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# In[39]:
# Note: Create the Processing Pipeline

# select numerical columns with missing values 
num_cols = ['AmountSpent', 'LoginFrequency', 'Days_Since_Last_Transaction', 'Days_Since_last_Interaction', 'Days_Since_Last_Login']

# select categorical columns with missing values
cat_cols = ['ProductCategory', 'ServiceUsage']

# create numerical imputer pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), 
    ('scaler', StandardScaler())
])

# create categorical imputer pipeline
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

# combine pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols), 
    ('cat', cat_pipeline, cat_cols)
])


# In[46]:

# NOTE: Implementing Stacking Machine Learning Method 

# define base estimator for stacking
base_learners = [ 
    ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)), 
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]

# In[152]:

# Define final estimator with scaling inside the pipelne 
final_estimator=Pipeline(steps=[
        ('standardscaler', StandardScaler()),
        ('logisticregression', LogisticRegression(class_weight='balanced', random_state=42))
    ])

# In[153]:

# Define the stacking model
stack_model = Pipeline(steps=[
    ('preprocessing', preprocessor), 
    ('model', StackingClassifier(
        estimators=base_learners, 
        final_estimator=final_estimator
        )
    )
])
    
# In[154]:

# Filter the dataset for training and testing 
x = merged_df.drop('ChurnStatus', axis=1) 
y = merged_df['ChurnStatus'] 

#%%

# split the data to train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)


#%%

# fit stack_model
stack_model.fit(x_train, y_train)

#%%

import pickle

# save the trained model using pickle 
with open('stacking_model.pkl', 'wb') as f: 
    pickle.dump(stack_model, f)
    
# In[155]:

# load the trained model (for prediction)
with open('stacking_model.pkl', 'rb') as f: 
    loaded_model = pickle.load(f)

# make prediction on the raw test data  
y_pred = loaded_model.predict(x_test)
