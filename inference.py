import pickle
import numpy as np
import pandas as pd
from joblib import dump,load

model= pickle.load(open('model_ml7.pkl','rb'))
convert=load('poly_convert.joblib')

def predict(df):
    df.drop('Administration',axis=1,inplace=True)
    df["State_Florida"]=df["State"]
    df["State_Newyork"]=df["State"]
    for i in range(len(df.State)):
        if(df.State[i]=='Florida'):
            df.State_Florida[i]=1
            df.State_Newyork[i]=0
        else:
            df.State_Florida[i]=0
            df.State_Newyork[i]=1
    df.drop('State',axis=1,inplace=True)
    a=convert.fit_transform(df)
    return(model.predict(a))

predict(df)
