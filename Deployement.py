#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
    st.title("Credit card default app")
    html_temp="""
    <div style="background-color:maroon ;padding:10px">
    <h2 style="color:white;text-align:center;">Hello viewer </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    option=st.write("Created by: Mirthula")
    st.header(option)
    st.sidebar.title("Credit Card Default Web App")
    st.markdown("Chances of account to become default by next month 💳")
    st.markdown("Our client has a credit card company. They have brought us a dataset that includes some demographics and recent financial data (the past six months) for a sample of 30,000 of their account holders. This data is at the credit account level; in other words, there is one row for each account. Rows are labeled by whether in the next month after the six month historical data period, an account owner has defaulted, or in other words, failed to make the minimum payment.")
    st.markdown("Choose your classifier from the given drop down box and you can set your hyperparameters accordingly and obtain the suitable metrics")
    st.sidebar.markdown("Chances of account to get defaulted by next month 💳")


    @st.cache(persist=True)
    def load_data():
        url = "https://raw.githubusercontent.com/mmirthula02/Credit-card/main/cleaned_data.csv"
        data = pd.read_csv(url,sep=",")
        labelencoder=LabelEncoder()
        for col in data.columns:
            data[col] = labelencoder.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        y = df['default payment next month'].values
        x = df.drop(columns=['default payment next month','ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school','others', 'university'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            #fig, ax = matplotlib.pyplot.subplots()
            #ax.plot([0,0.5,1],[0,0.5,1])
            #st.pyplot(fig)

         
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            #fig, ax = matplotlib.pyplot.subplots()
            #ax.plot([0,0.5,1],[0,0.5,1])
            #st.pyplot(fig)

    df = load_data()
    class_names = ['Default', 'Not Default']
    
    x_train, x_test, y_train, y_test = split(df)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression", "Random Forest"))

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Credit card default (Classification)")
        st.write(df)
        #st.markdown("This data set includes descriptions of bank account credit card payment data, we prdict the result analyzing these datas.")

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 50, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    

if __name__ == '__main__':
    main()


# In[ ]:




