#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 10:41:54 2018

@author: priyansu
"""
import pandas as pd
import numpy as np

train=pd.read_csv("Train.csv")
test=pd.read_csv("Test.csv")

train["source"]="train"
test["source"]="test"

data=pd.concat([train,test],ignore_index=True)

avg_weight= data.pivot_table(values="Item_Weight", index="Item_Identifier")
miss_bool=data["Item_Weight"].isnull()

data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x : avg_weight.at[x,"Item_Weight"])
p=pd.DataFrame()
p["Outlet_Size"]=data["Outlet_Size"]
p["Outlet_Type"]=data["Outlet_Type"]
p["Outlet_Size"]=p.dropna(subset=["Outlet_Size"], inplace=True)
p["Outlet_Size"]=data["Outlet_Size"]

 
from scipy.stats import mode
outlet_size_mode=p.pivot_table(values="Outlet_Size",index="Outlet_Type",aggfunc=lambda x: mode(x).mode[0])

miss_bool_2=data["Outlet_Size"].isnull()
data.loc[miss_bool_2,"Outlet_Size"]=data.loc[miss_bool_2,"Outlet_Type"].apply(lambda x: outlet_size_mode.at[x,"Outlet_Size"])
z=(data["Item_Visibility"]==0)

avg_visibility=data.pivot_table(values="Item_Visibility",index="Item_Identifier")
data.loc[z,"Item_Visibility"]=data.loc[z,"Item_Identifier"].apply(lambda x:avg_visibility.at[x,"Item_Visibility"] )

data["Item_Type_Combined"]=data["Item_Identifier"].apply(lambda x:x[0:2])
data["Item_Type_Combined"]=data["Item_Type_Combined"].map({"FD": "Food",
                                         "NC": "Non-consumable",
                                         "DR": "Drinks"})
  
data["years"]=2013-data["Outlet_Establishment_Year"]                                      

data["Item_Fat_Content"]=data["Item_Fat_Content"].replace({"LF": "Low Fat",
                                                           "reg": "Regular",
                                                           "low fat": "Low Fat"})
data.loc[data["Item_Type_Combined"]=="Non-consumable","Item_Fat_Content"]="Non-Edible"

data.drop(["Outlet_Establishment_Year","Item_Type"],axis=1,inplace= True)

train=data.loc[data["source"]=="train"]
test=data.loc[data["source"]=="test"]

train.to_csv("train_modified.csv")
test.to_csv("test_modified.csv")

#model_prediction

x_train=train.iloc[:,[0,1,2,4,5,6,7,8,9,11,12]].values
x_test=test.iloc[:,[0,1,2,4,5,6,7,8,9,11,12]].values
y_train=train.iloc[:,3].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label=LabelEncoder()
x_train[:,0]=label.fit_transform(x_train[:,0])
x_train[:,1]=label.fit_transform(x_train[:,1])
x_train[:,5]=label.fit_transform(x_train[:,5])
x_train[:,6]=label.fit_transform(x_train[:,6])
x_train[:,7]=label.fit_transform(x_train[:,7])
x_train[:,8]=label.fit_transform(x_train[:,8])
x_train[:,9]=label.fit_transform(x_train[:,9])

label2=LabelEncoder()
x_test[:,0]=label2.fit_transform(x_test[:,0])
x_test[:,1]=label2.fit_transform(x_test[:,1])
x_test[:,5]=label2.fit_transform(x_test[:,5])
x_test[:,6]=label2.fit_transform(x_test[:,6])
x_test[:,7]=label2.fit_transform(x_test[:,7])
x_test[:,8]=label2.fit_transform(x_test[:,8])
x_test[:,9]=label2.fit_transform(x_test[:,9])

one=OneHotEncoder(categorical_features=[0])
x_train=one.fit_transform(x_train).toarray()
two=OneHotEncoder(categorical_features=[1])
x_train=two.fit_transform(x_train).toarray()
five=OneHotEncoder(categorical_features=[5])
x_train=five.fit_transform(x_train).toarray()
six=OneHotEncoder(categorical_features=[6])
x_train=six.fit_transform(x_train).toarray()
seven=OneHotEncoder(categorical_features=[7])
x_train=seven.fit_transform(x_train).toarray()
eight=OneHotEncoder(categorical_features=[8])
x_train=eight.fit_transform(x_train).toarray()
nine=OneHotEncoder(categorical_features=[9])
x_train=nine.fit_transform(x_train).toarray()

one=OneHotEncoder(categorical_features=[0])
x_test=one.fit_transform(x_test).toarray()
two=OneHotEncoder(categorical_features=[1])
x_test=two.fit_transform(x_test).toarray()
five=OneHotEncoder(categorical_features=[5])
x_test=five.fit_transform(x_test).toarray()
six=OneHotEncoder(categorical_features=[6])
x_test=six.fit_transform(x_test).toarray()
seven=OneHotEncoder(categorical_features=[7])
x_test=seven.fit_transform(x_test).toarray()
eight=OneHotEncoder(categorical_features=[8])
x_test=eight.fit_transform(x_test).toarray()
nine=OneHotEncoder(categorical_features=[9])
x_test=nine.fit_transform(x_test).toarray()

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)





















