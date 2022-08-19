# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 20:26:38 2022

@author: 91931
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, render_template

import pickle

app = Flask(__name__)


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
  
    age                 = float(request.args.get('age'))                                              
    sex                 = float(request.args.get('sex'))
    on_thyroxine        = float(request.args.get('on_thyroxine'))
    query_on_thyroxine  = float(request.args.get('query_on_thyroxine'))
    on_antithyroid_meds = float(request.args.get('on_antithyroid_meds'))
    sick                = float(request.args.get('sick'))
    pregnant            = float(request.args.get('pregnant'))
    thyroid_surgery     = float(request.args.get('thyroid_surgery'))
    I131_treatment      = float(request.args.get('I131_treatment'))
    query_hypothyroid   = float(request.args.get('query_hypothyroid'))
    query_hyperthyroid  = float(request.args.get('query_hyperthyroid'))
    lithium             = float(request.args.get('lithium'))
    goitre              = float(request.args.get('goitre'))
    tumor               = float(request.args.get('tumor'))
    hypopituitary       = float(request.args.get('hypopituitary'))
    psych               = float(request.args.get('psych'))
    TSH                 = float(request.args.get('TSH'))
    T3                  = float(request.args.get('T3'))
    TT4                 = float(request.args.get('TT4'))
    T4U                 = float(request.args.get('T4U'))
    FTI                 = float(request.args.get('FTI'))
    TBG                 = float(request.args.get('TBG'))
    model1              = int(request.args.get('model1'))
    
    #  Checking Missing Values

    dataset= pd.read_csv('thyroidDF.csv')
    dataset ['TSH'] = dataset['TSH'].fillna(dataset['TSH'].median())
    dataset ['T3']  = dataset['T3'].fillna(dataset['T3'].median())
    dataset ['TT4'] = dataset['TT4'].fillna(dataset['TT4'].median())
    dataset ['T4U'] = dataset['T4U'].fillna(dataset['T4U'].median())
    dataset ['FTI'] = dataset['FTI'].fillna(dataset['FTI'].median())
    dataset['sex'] = np.where((dataset.sex.isnull()) & (dataset.pregnant == 't'), 'F', dataset.sex)
    dataset ['TBG'] = dataset['TBG'].fillna(0)
    dataset['sex'] = np.where((dataset.sex.isnull()) , 'M', dataset.sex)

    # dropping redundant attributes from dataset
    dataset.drop(['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 'patient_id', 'referral_source'], axis=1, inplace=True)
    # re-mapping target vaues to diagnostic groups
    diagnoses = {'-': 'negative',
                 'A': 'hyperthyroid', 
                 'B': 'hyperthyroid', 
                 'C': 'hyperthyroid', 
                 'D': 'hyperthyroid',
                 'E': 'hypothyroid', 
                 'F': 'hypothyroid', 
                 'G': 'hypothyroid', 
                 'H': 'hypothyroid'}

    dataset['target'] = dataset['target'].map(diagnoses) # re-mapping
    # dropping observations with 'target' null after re-mapping
    dataset.dropna(subset=['target'], inplace=True) 
    # Extracting independent variable:
    X = dataset.iloc[:, 0:22].values 

    labelencoder_X = LabelEncoder()
    X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
    X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
    X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
    X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
    X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
    X[:, 6] = labelencoder_X.fit_transform(X[:, 6])
    X[:, 7] = labelencoder_X.fit_transform(X[:, 7])
    X[:, 8] = labelencoder_X.fit_transform(X[:, 8])
    X[:, 9] = labelencoder_X.fit_transform(X[:, 9])
    X[:,10] = labelencoder_X.fit_transform(X[:, 10])
    X[:,11] = labelencoder_X.fit_transform(X[:, 11])
    X[:,12] = labelencoder_X.fit_transform(X[:, 12])
    X[:,13] = labelencoder_X.fit_transform(X[:, 13])
    X[:,14] = labelencoder_X.fit_transform(X[:, 14])
    X[:,15] = labelencoder_X.fit_transform(X[:, 15])
    
    sc = StandardScaler()
    X = sc.fit_transform(X)

    if model1==0:
      model=pickle.load(open('thyroid_linearreg.pkl', 'rb'))
      prediction = model.predict(([[age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI, TBG]]))  
      prediction = np.around(prediction)
    elif model1==1:
      model=pickle.load(open('thyroid_Logisticreg.pkl', 'rb'))
      prediction = model.predict(([[age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI, TBG]]))
    elif model1==2:
      model=pickle.load(open('thyroid_Decision_Tree.pkl', 'rb'))
      prediction = model.predict(([[age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI, TBG]]))
    elif model1==3:
      model=pickle.load(open('thyroid_KNN.pkl', 'rb')) 
      prediction = model.predict(([[age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI, TBG]]))
    elif model1==4:
      model=pickle.load(open('thyroid_kernal_svm.pkl', 'rb')) 
      prediction = model.predict(sc.transform([[age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI, TBG]]))
    elif model1==5:
      model=pickle.load(open('thyroid_linear_svm.pkl', 'rb'))
      prediction = model.predict(sc.transform([[age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI, TBG]]))
    elif model1==6:
      model=pickle.load(open('thyroid_randomforest.pkl', 'rb'))
      prediction = model.predict(sc.transform([[age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI, TBG]]))
    elif model1==7:
      model=pickle.load(open('thyroid_nb.pkl', 'rb'))
      prediction = model.predict(([[age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI, TBG]]))
    elif model1==8:
      model=pickle.load(open('thyroid_kmeanscluster.pkl', 'rb'))      
      prediction = model.predict(([[age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI, TBG]]))


    if prediction==0:
      output = "Hyperthyroid"
    if prediction==1:
      output = "Hypothyroid"
    if prediction==2:
      output = "Negative"
      
      
    return render_template('index.html', prediction_text='Model  has predicted, Patient with diagnosis :: {}'.format(output))       

if __name__=="__main__":
  app.run()
