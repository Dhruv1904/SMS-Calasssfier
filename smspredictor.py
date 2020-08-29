#!/usr/bin/env python
# coding: utf-8

# In[13]:


# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 02:20:31 2020

@author:
"""

# -*- coding: utf-8 -*-
"""
Created on Fri August 14 12:50:04 2020

@author: Dhruv.Shah
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words='english') 
from PIL import Image

pickle_in = open("naivebayes.pkl","rb")
model=pickle.load(pickle_in)
vect=open("vect.pkl", 'rb')
vectorizer=pickle.load(vect)
#@app.route('/')
def welcome():
    
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_spam(sms):
    sms=str(sms)
    y=model.predict_proba(vectorizer.transform([sms]))
    if y[0,0]>0.5:
        y=1
    else:
        y=0
    return y
    
def main():
    st.title("SMS Spam Predictor")
    html_temp = """
    <div style="background-color:#FF0000;padding:10px">
    <h2 style="color:Black;text-align:center;">SMS SPAM PREDICTOR WEB-APP </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    sms=st.text_input('Enter the message')

    
    result=""    
    if st.button("Check"):
        result=predict_spam(sms)
        if result ==1:
            result= str('The entered message is is not Spam')
        else:
            result= str('The entered message is a Spam')
    st.success(result)
  
    st.text("Follow DataMonk on Youtube to Know how to make such Awesome Webapps")

if __name__=='__main__':
    main()
    
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




