
# coding: utf-8

# In[84]:

#Importing Libraries
import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# In[85]:

# Reading the dataset
data = pd.read_csv(r'F:/cocoran/nyc-rolling-sales.csv')


# In[86]:

data.head(5)


# In[87]:

data = data.replace(' -  ', np.nan)
data = data.replace('', np.nan)


# # Converting following object type variables to Category

# In[88]:

data['BUILDING CLASS AT PRESENT'] = data['BUILDING CLASS AT PRESENT'].astype('category')
data['NEIGHBORHOOD'] = data['NEIGHBORHOOD'].astype('category')
data['BUILDING CLASS CATEGORY'] = data['BUILDING CLASS CATEGORY'].astype('category')
data['TAX CLASS AT PRESENT'] = data['TAX CLASS AT PRESENT'].astype('category')
data['ADDRESS'] = data['ADDRESS'].astype('category')
data['BUILDING CLASS AT TIME OF SALE'] = data['BUILDING CLASS AT TIME OF SALE'].astype('category')


# In[89]:

data['BUILDING CLASS AT PRESENT_new'] = pd.Categorical.from_array(data['BUILDING CLASS AT PRESENT']).codes
data['NEIGHBORHOOD_new'] = pd.Categorical.from_array(data['NEIGHBORHOOD']).codes
data['BUILDING CLASS CATEGORY_new'] = pd.Categorical.from_array(data['BUILDING CLASS CATEGORY']).codes
data['TAX CLASS AT PRESENT_new'] = pd.Categorical.from_array(data['TAX CLASS AT PRESENT']).codes
data['ADDRESS_new'] = pd.Categorical.from_array(data['ADDRESS']).codes
data['BUILDING CLASS AT TIME OF SALE_new'] = pd.Categorical.from_array(data['BUILDING CLASS AT TIME OF SALE']).codes


# # Finding Correlation Matrix 

# In[90]:

corr = data.corr()
corr


# # Dropping columns wich are highly correlated to reduce multi-collinearity

# In[91]:

data = data.drop(['BUILDING CLASS AT PRESENT','BUILDING CLASS CATEGORY','TAX CLASS AT PRESENT','RESIDENTIAL UNITS'],axis=1)


# In[92]:

data['SALE DATE'] = data['SALE DATE'].str[:10]


# In[93]:

data['LAND SQUARE FEET'] = data['LAND SQUARE FEET'].str.replace('$', '').astype(float)
data['GROSS SQUARE FEET'] = data['GROSS SQUARE FEET'].str.replace('$', '').astype(float)
data['SALE PRICE'] = data['SALE PRICE'].str.replace('$', '').astype(float)
data['SALE DATE'] = pd.to_datetime(data['SALE DATE'])


# In[94]:

#Converting Date to numeric type
data['SALE DATE'] = pd.to_datetime(data['SALE DATE'])
data['SALE DATE'].dt.strftime('%Y%m%d')
data['SALE DATE'] = pd.to_numeric(data['SALE DATE'])


# # Correlation of each variable with Target variable

# In[95]:

corr = data.corr()
corr_with_SalePrice = pd.DataFrame(corr['SALE PRICE'].drop('SALE PRICE'))
corr_with_SalePrice.sort_values(by = 'SALE PRICE', ascending = False)


# # Dropping less significant variables 

# In[96]:

#dropping column EASE-MENT that have no value in any row and APARTMENT NUMBER that have very few non-null values
data = data.drop(['EASE-MENT'],axis=1)
data = data.drop(['APARTMENT NUMBER'],axis=1)


# In[97]:

data=data.drop(['NEIGHBORHOOD','ADDRESS','BUILDING CLASS AT TIME OF SALE'],axis=1)


# In[98]:

#dropping features which have less correlation with sale price
data = data.drop(['NEIGHBORHOOD_new','ADDRESS_new','LOT','BUILDING CLASS AT PRESENT_new','BUILDING CLASS AT TIME OF SALE_new','ZIP CODE'],axis=1)


# In[99]:

#dropping rows that have null value in targte variable
data = data[np.isfinite(data['SALE PRICE'])]


# # Creating dummy variables for categorized features

# In[100]:

data['BOROUGH'] = data['BOROUGH'].astype('category')
data['TAX CLASS AT TIME OF SALE'] = data['TAX CLASS AT TIME OF SALE'].astype('category')


# In[102]:

Borough_dummies = pd.get_dummies(data['BOROUGH']).rename(columns=lambda x: 'BOROUGH_' + str(x))


# In[104]:

TAX_CLASS_AT_TIME_OF_SALE_dummies = pd.get_dummies(data['TAX CLASS AT TIME OF SALE']).rename(columns=lambda x: 'TAX CLASS AT TIME OF SALE_' + str(x))


# In[105]:

data = pd.concat([data, Borough_dummies,TAX_CLASS_AT_TIME_OF_SALE_dummies], axis=1)


# In[127]:

data.info()


# # Handling missing values

# In[128]:

#Plotting boxplot to check the distribution of values and decide if the missing values be replaced by median or mean values
data.boxplot(column=['LAND SQUARE FEET','GROSS SQUARE FEET'])


# In[129]:

#Replacing missing NAN values with median value
data['LAND SQUARE FEET'].fillna(data['LAND SQUARE FEET'].median(),inplace=True)
data['GROSS SQUARE FEET'].fillna(data['GROSS SQUARE FEET'].median(),inplace=True)


# # Plotting pair plot

# In[130]:

# Plotting pairplot to check relationship between all variables
sns.set(style="ticks", color_codes=True)
sns.pairplot(data=data,
                  y_vars=['SALE PRICE'],
                  x_vars=['BOROUGH', 'BLOCK', 'TOTAL UNITS','GROSS SQUARE FEET'])


# # Creating Train and Test Data
# 

# In[112]:

label = data['SALE PRICE']
#Dropping less significant variables
ml_data = data.drop(['SALE PRICE'], axis = 1)


# In[113]:

ml_data


# In[114]:

X_train, X_test, y_train, y_test = train_test_split(ml_data, label, test_size = 0.2, random_state = 25)


# # Linear Regression

# In[115]:

model = LinearRegression()


# In[116]:

X_train.info()


# In[117]:

model.fit(X_train,y_train)


# In[118]:

model.score(X_train,y_train)


# # Random Forest Regressor

# In[119]:

lr = RandomForestRegressor(n_estimators=100)


# In[120]:

lr.fit(X_train,y_train)


# In[121]:

lr.score(X_train,y_train)

