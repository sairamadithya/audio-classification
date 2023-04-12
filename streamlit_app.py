#!/usr/bin/env python
# coding: utf-8

# In[83]:


#%%writefile audio_test.py
import streamlit as st
import librosa, librosa.display
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from audio_recorder_streamlit import audio_recorder
import soundfile as sf
def load_model():
        model=tf.keras.models.load_model(r"urbansound8k-2.9.h5")
        return model
with st.spinner('Model is being loaded..'):
        model=load_model()
html_temp = """ 
  <div style="background-color:pink ;padding:10px">
  <h2 style="color:white;text-align:center;">DEEP LEARNING BASED AUDIO CLASSIFICATION</h2>
  </div>
  """
st.markdown(html_temp,unsafe_allow_html=True)
st.subheader('this website can classify audio samples into 10 different categories:-')
st.write('1.air conditioner')
st.write('2. car horn')
st.write('3. children playing')
st.write('4. dog bark')
st.write('5. drilling')
st.write('6. engine idling')
st.write('7. gun shot')
st.write('8. jackhammer')
st.write('9. siren')
st.write('10. street music')
st.write('Please record or upload the audio sample')
file=st.file_uploader("Choose a file",type=[".wav",".mp3"])
if file is None:
        st.warning('Please upload a valid file!!')
else:
        st.audio(file,format="audio/wav")
        audio, sample_rate = librosa.load(file) 
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
        predicted_label=model.predict(mfccs_scaled_features)
        a=np.argmax(predicted_label,axis=1)
        if st.button('predict'):
            if a==0:
                out='air conditioner'
            elif a==1:
                out='car horn'
            elif a==2:
                out='children laying'
            elif a==3:
                out='dog bark'
            elif a==4:
                out='drilling'
            elif a==5:
                out='engine idling'
            elif a==6:
                out='gun shot'
            elif a==7:
                out='jackhammer'
            elif a==8:
                out='siren'
            else:
                out='street music'
            st.success('the given audio sample is predicted as:- '+str(out))
st.success('DEVELOPED BY V.A.SAIRAM')
st.success('email= sairamadithya2002@gmail.com')
st.success('linkedin= https://www.linkedin.com/in/sairamadithya/')

