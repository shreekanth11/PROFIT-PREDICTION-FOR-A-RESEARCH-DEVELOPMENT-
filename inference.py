import pickle
import numpy as np
import pandas as pd
from joblib import dump,load

model= pickle.load(open('model_ml7.pkl','rb'))
convert=load('poly_convert.joblib')

def predict(df):
    df=df[["RD_Spend","Administration","Marketing_Spend","State"]]
    df.drop('Administration',axis=1,inplace=True)
    df["State_Florida"]=df["State"]
    df["State_Newyork"]=df["State"]
    for i in range(len(df.State)):
        if(df.State[i]=='Florida'):
            df.State_Florida[i]=1
            df.State_Newyork[i]=0
        elif(df.State[i]=='Newyork'):
            df.State_Florida[i]=0
            df.State_Newyork[i]=1
        else:
            df.State_Florida[i]=0
            df.State_Newyork[i]=0
    df.drop('State',axis=1,inplace=True)
    a=convert.fit_transform(df)
    numpy_array = a.to_numpy()
    predictions = model.predict(numpy_array)
    output = (np.around(predictions)).tolist()
    sent = "Predicted Profit is: "
    output = [str(i) for i in output]
    output = ["{}{}".format(sent , i) for i in output]
    return output
