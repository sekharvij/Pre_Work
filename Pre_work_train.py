#!/usr/bin/env python
# coding: utf-8

# ## **Pre -Work**

# ### **Data definition**
# 
# **Air Quality Index Data** 
# 
# - **Date**: 
# - **CBSA Code**:                     
# - **Defining Parameter**:             
# - **Number of Sites Reporting**:     
# - **lat**:                        
# - **lng**:                         
# - **population**:                   
# - **density**:                      
# - **timezone**:                      
# - **city**:                           
# - **state**:     
#     
#     
# ### **Problem Statement**
# 
# - To predict the air quality index (AQI) from the us AQI data and the air quality scores can be categorised in the following ways:
# 
# Good:                           (0, 50)
# Hazardous:                      (303, 1250)
# Moderate:                       (51, 100)
# Unhealthy:                      (151, 200)
# Unhealthy for Sensitive Groups: (101, 150)
# Very Unhealthy:                 (201, 298)
# 
# 

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import zipfile
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import pickle 


# In[2]:


zi = zipfile.ZipFile('C:/Users/SEKHAR/Desktop/Pre_Work/recruitment-air-quality-2/data/train.zip') 
data = pd.read_csv(zi.open('train.csv'))


# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# In[6]:


data['Date'] = pd.to_datetime(data['Date'])


# In[7]:


data.head()


# In[8]:


data['Defining Parameter'].unique()


# In[9]:


sum(data.duplicated())


# - No null values and duplicated data found

# In[10]:


pd.Categorical(data['Defining Parameter']).describe()


# ### EDA

# In[11]:


def PropByvar(df, var):
    dataframe_pie = df[var].value_counts()
    ax = dataframe_pie.plot.pie(figsize =(10,10), autopct='%1.2f%%',fontsize =10)
    ax.set_title (var+ '\n',fontsize=12)
    return np.round(dataframe_pie/df.shape[0]*100,2)


# In[12]:


PropByvar(data,'Defining Parameter')


# In[13]:


plt.tight_layout()
plt.savefig("Defining Parameter Pie Chart.png",dpi=120)
plt.close()


# In[14]:


sns.pairplot(data)


# In[15]:


df = data.copy()


# In[16]:


bins = [0, 50, 100, 150, 200, 302, 1250]
names = ['Good', 'Moderate', 'Unhealthy_for_Sensitive_Groups', 'Unhealthy', 'Very Unhealthy','Hazardous']

df['AQ'] = pd.cut(df['AQI'], bins, labels=names)


# In[ ]:





# In[17]:


#(df['AQ']==0).value_counts()
#False    295405
#True       4595
#Name: AQ, dtype: int64


# In[18]:


(df['AQ']==0).value_counts()


# In[19]:


df.head()


# In[20]:


df.corr()


# In[21]:


corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[22]:


plt.tight_layout()
plt.savefig("correlation Heat Map.png",dpi=120)
plt.close()


# In[23]:


df.dtypes


# ### Label Encoding

# In[24]:


cat_mod = ['Defining Parameter','AQ']
le = LabelEncoder()
for i in cat_mod:
    df[i] = le.fit_transform(df[i]).astype(str)


# In[25]:


df.head()


# In[ ]:





# In[26]:


X = df.drop(columns=['AQI', 'Date'])
y = df['AQI']


# In[27]:


X['AQ'] = X['AQ'].astype(str).astype(int)
X['Defining Parameter'] = X['Defining Parameter'].astype(str).astype(int)


# ### Feature Importance

# In[28]:


model = ExtraTreesRegressor()
model.fit(X,y)


# In[29]:


feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[30]:


plt.tight_layout()
plt.savefig("Feature Importance.png",dpi=120)
plt.close()


# ### Test Train Split

# In[31]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ### XG Boost Regression

# In[32]:


xgb_regressor=xgb.XGBRegressor()


# In[33]:


xgb_regressor.fit(X_train,y_train)


# In[34]:


xgb_score=cross_val_score(xgb_regressor,X,y,cv=5)


# In[35]:


xgb_score


# In[36]:


xgb_prediction=xgb_regressor.predict(X_test)
xgb_prediction


# ### XG - Hyperparameter Tuning

# In[37]:


# Number of trees in random forest
xg_n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 2)]
# Various learning rate parameters
xg_learning_rate = ['0.05','0.1', '0.5']
# Maximum number of levels in tree
xg_max_depth = [int(x) for x in np.linspace(5, 10, num = 2)]
# max_depth.append(None)
#Subssample parameter values
xg_subsample=[0.7,0.6,0.8]
# Minimum child weight parameters
xg_min_child_weight=[3,5,7]


# In[38]:


xgb_random_grid = {'n_estimators': xg_n_estimators,
               'learning_rate': xg_learning_rate,
               'max_depth': xg_max_depth,
               'subsample': xg_subsample,
               'min_child_weight': xg_min_child_weight}


# In[39]:


xg_random = RandomizedSearchCV(estimator = xgb_regressor, param_distributions = xgb_random_grid,scoring='neg_mean_squared_error', n_iter = 4, cv = 5, verbose=2,random_state=42)


# In[40]:


xg_random.fit(X_train,y_train)


# In[41]:


xgb_predictions=xg_random.predict(X_test)


# In[42]:


print('MAE:', metrics.mean_absolute_error(y_test, xgb_predictions))
print('MSE:', metrics.mean_squared_error(y_test, xgb_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xgb_predictions)))


# In[43]:


with open("xg_metric.txt", "w") as outfile:
 outfile.write("XG Boost Regressor Root Mean Sqaure Error:%2.1f%%\n"% np.sqrt(metrics.mean_squared_error(y_test, xgb_predictions)))


# In[44]:


xg_y_pred = xg_random.predict(X_test) + np.random.normal(0,0.2,len(y_test))
xg_y_actual = y_test + np.random.normal(0,0.2,len(y_test))
xg_res_df = pd.DataFrame(list(zip(xg_y_actual,xg_y_pred)),columns=["XG_Actual", "XG_Pred"])


# In[45]:


ax = sns.scatterplot(x="XG_Actual", y= "XG_Pred", data = xg_res_df)
ax.set_aspect("equal")
ax.set_xlabel("XG_Actual", fontsize = 10)
ax.set_ylabel("XG_Pred", fontsize = 10)
ax.set_title("XG Residual", fontsize = 12)
ax.plot([1,10],[1,10],"red", linewidth =1)
plt.xlim((2.5,8.5))
plt.ylim((2.5,8.5))


# In[46]:


plt.tight_layout()
plt.savefig("XG boost Residual.png",dpi=120)
plt.close()


# - from the Three models Liner, Random Forest and XG Boost , The RSME value of **XG Boost model (9.497117381810169)** found less compared to other models **RF model (9.581215579324233)** and **LR model (16.058382899819687)**
#  
#                        

# In[47]:


file = open('Pre_work_regression_model.pkl', 'wb')
pickle.dump(xg_random, file)


# In[ ]:




