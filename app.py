import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
st.set_option('deprecation.showPyplotGlobalUse', False)

model=pickle.load(open("ada.pkl", "rb"))

def main():
    st.title('Heart Disease Prediction')
    age=st.slider("Age",0,100,1)
    sex=st.selectbox("Sex",["Female","Male"])
    cp=st.selectbox("CP",[0,1,2,3])
    trestbps=st.slider("Trestbps",90,200,1)
    chol=st.slider("Chol",126,564,1)
    fbs=st.selectbox("fbs",[0,1])
    restecg=st.selectbox("Restecg",[0,1,2])
    thalach=st.slider("Thalach",71,202,1)
    exang=st.selectbox("Exang",[0,1])
    oldpeak=st.slider("Oldpeak",0,5,1)/10
    slope=st.selectbox("Slope",[0,1,2])
    ca=st.selectbox("Ca",[0,1,2,3,4])
    thal=st.selectbox("Thal",[0,1,2,3])
    
    df2 = pd.DataFrame(data=[[cp,oldpeak,thal,ca,thalach,age,chol,trestbps,exang]],columns=['cp', 'oldpeak', 'thal', 'ca', 'thalach', 'age', 'chol', 'trestbps', 'exang'])
    #df3=scalar.transform(df2)
    df2=df2.fillna(value=0)
    df2=df2.interpolate(method='ffill')
    #df3
    
    #df2 = pd.DataFrame(data=[[cp,oldpeak,thal,ca,thalach,age,chol,trestbps,exang]],columns=['cp', 'oldpeak', 'thal', 'ca', 'thalach', 'age', 'chol', 'trestbps', 'exang'])
    #df3=scalar.transform(df2)
    #df3=df2[['cp','oldpeak','thal','ca','thalach','age','chol','trestbps','exang']]
    #df=load_data()
    #number=[0,1,2]
    #for col in df.itertuples():
        #if col.cp in number:
            #df['cp'].replace(to_replace=col.cp, value=1, inplace=True)
    #df_top8 = df.loc[:, ['cp','oldpeak','thal','ca','thalach','age','chol','trestbps','exang']]
    #df_top8 = df_top8.interpolate(method='ffill')
    #df = df.interpolate(method='ffill')
    #x = df.iloc[:,:-1]
    #x_std = StandardScaler().fit_transform(x)
    #y=df['target']
    

    if st.button("Predict"):
        prediction = model.predict(df2)
        if prediction == 0:
            st.error("Not attacked")
        else:
            st.success("Attack")
            
def load_data():
    df = pd.read_csv("heart_disease.csv")
    return df 

if __name__ == "__main__":
    main()