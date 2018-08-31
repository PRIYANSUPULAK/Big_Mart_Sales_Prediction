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

x=data.iloc[:,[0,1,2,4,5,6,7,8,9,11,12]].values
y=data.iloc[:,3].values























