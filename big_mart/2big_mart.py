# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:40:23 2016

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#train['source']='train'
#test['source']='test'
data=pd.concat([train,test],ignore_index=True)
print(data.apply(lambda x : sum(pd.isnull(x))))
print(data.apply(lambda x:len(x.unique())))
#print(data.dtypes.index)
categorial_cols=[x for x in data.dtypes.index if data.dtypes[x]=='object']
categorial_Cols=[x for x in categorial_cols if x not in ['Outlet_Identifier','Item_Identifier','Source']]
for i in categorial_cols:
    print(data[i].value_counts())
item_avg_weight=data.pivot_table(values='Item_Weight',index='Item_Identifier')
#print(item_avg_weight)
missing_value_bool=data['Item_Weight'].isnull()
#print(missing_value_bool)
#print(sum(missing_value_bool))
data.loc[missing_value_bool,'Item_Weight']=data.loc[missing_value_bool,'Item_Identifier'].apply(lambda x:item_avg_weight[x])
#print(sum(missing_value_bool))
print(data.info())
from scipy.stats import mode
#outlet_size_mode=data.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x : mode(x).mode[0]) )
data.loc[data[(data['Outlet_Type']=='Grocery_Store') & (data['Outlet_Size'].isnull())].index,'Outlet_Size']='Small'
data.loc[data[(data['Outlet_Type']=='Supermarket Type1') & (data['Outlet_Size'].isnull())].index,'Outlet_Size']='Small'
data.loc[data[(data['Outlet_Type']=='Supermarket Type2') & (data['Outlet_Size'].isnull())].index,'Outlet_Size']='Medium'
data.loc[data[(data['Outlet_Type']=='Supermarket Type3') & (data['Outlet_Size'].isnull())].index,'Outlet_Size']='Medium'

#FEATURE ENGINEERING
outlet_type_mean=data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type')
print(outlet_type_mean)
visibility_mean=data.pivot_table(values='Item_Visibility',index='Item_Identifier')
missing_bool=(data['Item_Visibility']==0)
#print(missing_bool)
data.loc[missing_bool,'Item_Visibility']=data.loc[missing_bool,'Item_Identifier'].apply(lambda x : visibility_mean[x])
data['Item_Visibility_Ratio']=data.apply(lambda x:x['Item_Visibility']/visibility_mean[x['Item_Identifier']],axis=1)
print(data['Item_Visibility_Ratio'].describe())
data['Food_Item_Combined']=data['Item_Identifier'].apply(lambda x: x[0:2])
data['Food_Item_Combined']=data['Food_Item_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
print(data['Food_Item_Combined'].value_counts())
data['Outlet_Age']=2013-data['Outlet_Establishment_Year']
data['Item_Fat_Content']=data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
print(data['Item_Fat_Content'].value_counts())
data.loc[data['Food_Item_Combined']=='Non-Consumable','Item_Fat_Content']='Non_Edible'
print(data['Item_Fat_Content'].value_counts())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Outlet']=le.fit_transform(data['Outlet_Identifier'])
data=pd.get_dummies(data,columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Food_Item_Combined','Outlet'])
print(data.info())
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)
#train=data.loc[data['source']=='train']
#test=data.loc[data['source']=='test']
ID=['Item_Identifier','Outlet_Identifier']
data.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
#test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
#train.drop(['source'],axis=1,inplace=True)
train = data.ix[0:8522]
test = data.ix[8523:]

#MODEL
target=train['Item_Outlet_Sales']
ID=['Item_Identifier','Outlet_Identifier']
from sklearn import cross_validation, metrics
def modelfit(alg,dtrain,dtest,predictors,target,ID,filename):
    alg.fit(dtrain[predictors],target[target])
    dtrain_predictions=alg.predict(dtrain[predictors])
    cv_score=cross_validation.cross_val_score(alg,dtrain[predictors],dtrain[target],cv=20,scoreing='mean_squared_error')
    cv_score=np.sqrt(np.abs(cv_score))
    print("model report")
    print("rmse %.4g"%np.sqrt(metrics.mean_squared_error(dtrain[target].values,dtrain_predictions)))
    print("cv_score %"%np.mean(cv_score))
    dtest[target]=alg.predict(dtrain[predictors])
    ID.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in ID})
    submission.to_csv(filename, index=False)
    
from sklearn.ensemble import RandomForestRegressor
#train.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg, train, test, predictors, target, ID, 'sub1.csv')

    
    
    