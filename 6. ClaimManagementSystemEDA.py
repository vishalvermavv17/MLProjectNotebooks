#!/usr/bin/env python
# coding: utf-8

# <h1 align='center'> Claim Management System </h1>

# ![image.png](attachment:64c348c8-d0a0-41d8-b940-9f0778852934.png)

# 1. [Introduction](#1) <a id=18></a>
#     - 1.1 [Data Dictionary](#2)
#     - 1.2 [Data Source Links](#3)
#     - 1.3 [Task](#4)
# 2. [Preparation](#5)
#     - 2.1 [Packages](#6)
#     - 2.2 [Data](#7)
#     - 2.3 [Understanding Data](#8)
# 3. [Exploratory Data Analysis](#9)
#     - 3.1 [Univariate Analysis](#10)
#     - 3.2 [Bivariate Analysis](#11)
# 4. [Conclusions from EDA](#12)

# ## 1. Introduction <a id=1></a>
# 
# In a world shaped by the emergence of new uses and lifestyles, everything is going faster and faster. When facing unexpected events, customers expect their insurer to support them as soon as possible. However, claims management may require different levels of check before a claim can be approved and a payment can be made. With the new practices and behaviors generated by the digital economy, this process needs adaptation thanks to data science to meet the new needs and expectations of customers.
# 
# ![image.png](attachment:7bce8865-4307-4334-bb70-7e60a698af0a.png)
# 
# In this challenge, BNP Paribas Cardif is providing an anonymized database with two categories of claims:
# 1. Suitable for approval.
# 2. Not suitable for approval.
# 
# We need to predict the category of a claim based on features available early in the process, helping BNP Paribas Cardif accelerate its claims process and therefore provide a better service to its customers.

# ### 1.1 Data Dictionary <a id=2></a>

# ### 1.2 Data Source Links <a id=3></a>
# 
# https://www.kaggle.com/competitions/bnp-paribas-cardif-claims-management/data

# ### 1.3 Task <a id=4></a>
# 
# To predict the category of a claim based on features available early in the process. We'll follow the below data science project lifecycle:
# 1. Understanding of Data (EDA)
# 2. Pre-processing of data.
# 3. Selection of ML algorithm.
# 4. Hyper-parameter tuning.
# 

# ## 2. Preparation <a id=5></a>

# ### 2.1 Packages <a id=6></a>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# ### 2.2 Data <a id=7></a>

# In[2]:


df = pd.read_csv('../input/bnp-paribas-cardif-claims-management/train.csv.zip')


# ### 2.3 Understanding Data <a id=8></a>

# #### 2.3.1 The shape of the data 

# In[3]:


df.shape


# #### 2.3.2 Preview of the first 5 rows of dataset 

# In[4]:


df.head(10)


# #### 2.3.3 Checking the number of unique values  in each column 

# In[5]:


dict = {}
for col in df.columns:
    dict[col] = df[col].value_counts().shape[0]
pd.set_option("display.max_columns", None)
pd.DataFrame(dict, index=['unique value count'])


# #### 2.3.4 Checking data type of each column

# In[6]:


df.dtypes.reset_index().transpose()


# #### 2.3.5 Separating Categorical and Continuous variables

# In[7]:


target_col = ['target']
cat_cols = ['v3', 'v22', 'v24', 'v30', 'v31', 'v38', 'v47', 'v52', 'v56', 'v62', 'v66', 'v71', 'v72', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112', 'v113', 'v125', 'v129'] 
con_cols = list(df.columns.drop(target_col + cat_cols))

print("There are {} Categorical cols : {}".format(len(cat_cols), cat_cols))
print("There are {} Continuous cols : {}".format(len(con_cols), con_cols))
print("There are {} Target cols : {}".format(len(target_col), target_col))


# #### 2.3.6 Checking the number of unique values for cat columns

# In[8]:


dict = {}
for col in cat_cols:
    dict[col] = df[col].value_counts().shape[0]
pd.set_option("display.max_columns", None)
pd.DataFrame(dict, index=['unique value count'])


# Category columns v22 and v56 has a lot of categorical values. We might need to address these variables in data preprocessing.
# 1. We can use bayesian target encoder.
# 2. We can use frequency based ordinal encoder to give importance to each value relatice to its frequeny.

# #### 2.3.7 Summary statistics for cols

# In[9]:


df.describe()


# #### 2.3.8 Missing Values
# ##### Missing values (column wise)
# 
# There are multiple columns with a high degree of missing values.

# In[10]:


pd.DataFrame(df.isnull().sum()).transpose()


# ##### Missing values in terms of percentage of total data
# 
# **33%** of the data is has missing values.

# In[11]:


df.isnull().sum().sum()/(df.shape[0]*df.shape[1])


# ## 3. Exploratory Data Analysis <a id=9></a>

# ### 3.1 Univariate Analysis <a id=10></a>
# 
# #### 3.1.1 Count plot for target variable

# In[12]:


def set_spines_visibility(ax, is_visible):
    for s in ['left', 'right', 'top', 'bottom']:
        ax.spines[s].set_visible(is_visible)


# In[13]:


fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(1, 2)

ax0 = fig.add_subplot(gs[0,0])
ax0.text(x=0.5, y=0.5, s="Count plot for \n Target variable",
        verticalalignment='center', horizontalalignment='center',
        fontsize='24', fontweight='bold')
set_spines_visibility(ax0, False)
ax0.tick_params(bottom=False, left=False)
ax0.set_xticklabels([])
ax0.set_yticklabels([])

ax1 = fig.add_subplot(gs[0,1])
sns.countplot(ax=ax1, data=df, x=target_col[0])
set_spines_visibility(ax1, False)


# #### 3.1.1 Count plot for categorical variables

# In[14]:


fig = plt.figure(figsize=(15,40))
gs = fig.add_gridspec(12, 2)

ax0 = fig.add_subplot(gs[0,0])
ax0.text(x=0.5, y=0.5, s="Count plot for \n Categorical variables",
        verticalalignment='center', horizontalalignment='center',
        fontsize='24', fontweight='bold')
set_spines_visibility(ax0, False)
ax0.tick_params(bottom=False, left=False)
ax0.set_xticklabels([])
ax0.set_yticklabels([])

for i in range(len(cat_cols)):
    ax = fig.add_subplot(gs[(i+1)//2, 1-i%2])
    if df[cat_cols[i]].unique().shape[0] > 15:
        df_top_feature_values = df[cat_cols[i]].value_counts().reset_index()
        df_top_feature_values.columns = ['col_values', 'value_count']
        top_values = df_top_feature_values['col_values'].head(15)
        sns.countplot(ax=ax, data=df[df[cat_cols[i]].isin(top_values)], x=cat_cols[i])
    else:
        sns.countplot(ax=ax, data=df, x=cat_cols[i])
    set_spines_visibility(ax, False)

# plt.xticks(rotation=0)
plt.show()


# There are lot of imbalance in data within single feature column. Some feature columns are almost dominated by single value itself. We can drop these columns in next step after getting actual numbers, as it wouldn't provide any information to the model.

# #### 3.1.2 Identification of Single Value dominated Feature columns
# 
# Based on above univariate count plots, features <b> v3, v30, v31, v38, v62, v74 & v129 </b> seemed single valued feature, let's further analysing these features.

# In[15]:


percent_v3 = df[df['v3'] == 'C'].shape[0]/df.shape[0]
print('Feature value V3 has value = "C" for {}% out of total samples'.format(percent_v3*100))

percent_v30 = df[df['v30'] == 'C'].shape[0]/df.shape[0]
print('Feature value V30 has value = "C" for {}% out of total samples'.format(percent_v30*100))

percent_v31 = df[df['v31'] == 'A'].shape[0]/df.shape[0]
print('Feature value V31 has value = "A" for {}% out of total samples'.format(percent_v31*100))

percent = df[df['v38'] == 0].shape[0]/df.shape[0]
print('Feature value V38 has value = "0" for {}% out of total samples'.format(percent*100))

percent = df[df['v62'] == 1].shape[0]/df.shape[0]
print('Feature value V62 has value = "1" for {}% out of total samples'.format(percent*100))

percent = df[df['v74'] == 'B'].shape[0]/df.shape[0]
print('Feature value V74 has value = "B" for {}% out of total samples'.format(percent*100))

percent = df[df['v129'] == 0].shape[0]/df.shape[0]
print('Feature value V129 has value = "0" for {}% out of total samples'.format(percent*100))


# Feature Columns <b>[v3, v38, v74] </b> is not adding useful information for model building. These columns have same value in more than <b>95%</b> of the samples.

# #### 3.1.3 Boxen plot for continuous feature variables

# In[16]:


fig = plt.figure(figsize=(16, 250))
gs = fig.add_gridspec(55, 2)

for i in range(len(con_cols) + 1):
    if i == 0:
        ax = fig.add_subplot(gs[0,0])
        ax.text(x=0.5, y=0.5, s="Boxen Plot for \nContinuous Variables",
               horizontalalignment='center', verticalalignment='center',
               fontweight='bold', fontsize='24')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)
        set_spines_visibility(ax, False)
    else:
        ax = fig.add_subplot(gs[(i)//2, i%2])
        sns.boxenplot(ax=ax, data=df, x=con_cols[i-1])
        set_spines_visibility(ax, False)
plt.show()


# <b>Outliers</b> are present in a `majority of continuous feature variables` that we need to address during data pre-processing.

# #### 3.1.4 Principal component analysis on continuous features

# In[17]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(pd.notnull(df[con_cols])))
scaled_df.head()

pca = PCA(.99)
pca_df = pca.fit_transform(scaled_df)
print("original data shape {}", scaled_df.shape)
print("pca data shape", pca_df.shape)
pca.explained_variance_ratio_


# PCA shows that we can keep <b>99% data variance</b> with 5 columns only. Hence, we can reduce the dimensionality of continuous features from <b>109 --> 5</b>.

# ## 3.2 Bivariate Analysis <a id=11></a>
# ### 3.2.1 Correlation Matrix for continuous features
# Given data has a large number of continuous features. To better visualise the heatmap, we have divided the heatmap into 4 views (top-left, top-right, bottom-left & bottom-right).

# In[18]:


df_corr = df[con_cols].corr().transpose()

fig = plt.figure(figsize=(14, 50))
gs = fig.add_gridspec(4,1)

ax = fig.add_subplot(gs[0,0])
mask = np.triu(np.ones_like(df_corr.iloc[:55, :55], dtype=bool))
sns.heatmap(ax=ax, data=df_corr.iloc[:55, :55], mask=mask)
_ = ax.set_title('top-left view of the heatmap',fontsize=22, fontweight='bold', fontfamily='serif')

ax = fig.add_subplot(gs[1,0])
mask = np.triu(np.ones_like(df_corr.iloc[:55, 55:], dtype=bool))
sns.heatmap(ax=ax, data=df_corr.iloc[:55, 55:], mask=mask)
_ = ax.set_title('top-right view of the heatmap',fontsize=22, fontweight='bold', fontfamily='serif')

ax = fig.add_subplot(gs[2,0])
mask = np.triu(np.ones_like(df_corr.iloc[55:, :55], dtype=bool))
sns.heatmap(ax=ax, data=df_corr.iloc[55:, :55], mask=mask)
_ = ax.set_title('bottom-left view of the heatmap',fontsize=22, fontweight='bold', fontfamily='serif')

ax = fig.add_subplot(gs[3,0])
mask = np.triu(np.ones_like(df_corr.iloc[55:, 55:], dtype=bool))
sns.heatmap(ax=ax, data=df_corr.iloc[55:, 55:], mask=mask)
_ = ax.set_title('bottom-right view of the heatmap',fontsize=22, fontweight='bold', fontfamily='serif')


# ### 3.2.2 Scatterplot heatmap of dataframe

# In[19]:


fig = plt.figure(figsize=(100, 100))

corr_mat = df.corr().stack().reset_index(name="correlation")

# Setting correlation value=0, for less correlated features to better visualise highly correlated feature.
mask = [(corr_mat['correlation'] < .75) & (corr_mat['correlation'] >-.75)]
corr_mat['correlation'].iloc[mask] = 0

grid = sns.relplot(data=corr_mat, x='level_0', y='level_1', hue='correlation', hue_norm=(-1, 1),
           height=30, sizes=(50, 350), size_norm=(-1, 1), size='correlation')

grid.set(xlabel="Features on X", ylabel="Features on Y", aspect="equal")
grid.despine(left=True, bottom=True)
grid.fig.suptitle('Scatterplot heatmap',fontsize=28, fontweight='bold', fontfamily='serif')
grid.ax.margins(.02)

for label in grid.ax.get_xticklabels():
    label.set_rotation(90)
    label.set_fontsize(15)
for label in grid.ax.get_yticklabels():
    label.set_fontsize(15)

plt.show()


# ### 3.2.3 Distribution of categorical features against target

# In[20]:


def set_spines_visibility(ax, is_visible):
    for s in ['left', 'right', 'top', 'bottom']:
        ax.spines[s].set_visible(False)


# In[21]:


fig = plt.figure(figsize=(12, 90))
gs = fig.add_gridspec(12, 2)

for i in range(len(cat_cols)+1):
    ax = fig.add_subplot(gs[i//2, i%2])
    if i == 0:
        ax.text(x=.5, y=.5, s="Distribution of \nCategorical features \nagainst target",
               horizontalalignment='center', verticalalignment='center',
               fontsize='24', fontweight='bold')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(left=False, bottom=False)
    else:
        sns.countplot(ax=ax, data=df, x=cat_cols[i-1], fill=True, hue=target_col[0])
    set_spines_visibility(ax, False)


# #### 3.2.4 Boxen plot of continuous variables vs target

# In[22]:


fig = plt.figure(figsize=(16, 250))
gs = fig.add_gridspec(55, 2)

for i in range(len(con_cols) + 1):
    if i == 0:
        ax = fig.add_subplot(gs[0,0])
        ax.text(x=0.5, y=0.5, s="Boxen Plot for \nContinuous Variables",
               horizontalalignment='center', verticalalignment='center',
               fontweight='bold', fontsize='24')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)
        set_spines_visibility(ax, False)
    else:
        ax = fig.add_subplot(gs[(i)//2, i%2])
        sns.boxenplot(ax=ax, data=df, y=con_cols[i-1], x=target_col[0])
        set_spines_visibility(ax, False)
plt.show()


# ## 4 Conclusions from EDA <a id=12></a>
# 1. There are both categorical and continuous features present in data and target feature has binary value.
# 2. Feature values are mixed of int, float64 & object data types.
# 3. There are multiple columns with a high degree of missing values.
# 4. There is a class imbalance in target variable ('label 0':23.1%, 'label 1':76.9%).
# 5. Based on univariate count plots of categorical variables, features <b> v3, v30, v31, v38, v62, v74 & v129 </b> seemed single valued feature i.e. zero-vector predictor. These columns have same value in more than <b>95%</b> of the samples and not adding useful information for model building.
# 6. <b>Outliers</b> are present in a `majority of continuous feature variables` that we need to address during data pre-processing.
# 7. There are continuous variables with skewed distributions.
# 8. PCA shows that we can keep <b>99% data variance</b> with 5 columns only. Hence, there are dependent variables in data.
# 9. There are highly correlated dependent features to other dependent features based on the [3.2.1] heatmap and [3.2.2] scatter plot heatmap.

# ### If you like the notebook, consider giving an upvote.
# Check my other notebooks 
# 
# 1. https://www.kaggle.com/code/crashoverdrive/data-science-salary-complete-eda
# 2. https://www.kaggle.com/code/crashoverdrive/studentsperformance-data-visualization-beginners
# 3. https://www.kaggle.com/code/crashoverdrive/heart-attack-analysis-prediction-90-accuracy
# 4. https://www.kaggle.com/code/crashoverdrive/song-popularity-prediction-visualizations
