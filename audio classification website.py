#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install librosa')


# In[85]:


get_ipython().system('pip install streamlit')


# In[83]:


get_ipython().run_cell_magic('writefile', 'audio_test.py', 'import streamlit as st\nimport librosa, librosa.display\nimport pandas as pd\nimport numpy as np\nimport tensorflow as tf\nfrom tensorflow import keras\nfrom audio_recorder_streamlit import audio_recorder\nimport soundfile as sf\ndef load_model():\n        model=tf.keras.models.load_model(r"C:\\Users\\sairam\\Downloads\\urbansound8k-2.9.h5")\n        return model\nwith st.spinner(\'Model is being loaded..\'):\n        model=load_model()\nst.title(\'Audio classification project\')\nst.write(\'this website can classify audio samples into 10 different categories:-\')\nst.write(\'1.air conditioner\')\nst.write(\'2. car horn\')\nst.write(\'3. children playing\')\nst.write(\'4. dog bark\')\nst.write(\'5. drilling\')\nst.write(\'6. engine idling\')\nst.write(\'7. gun shot\')\nst.write(\'8. jackhammer\')\nst.write(\'9. siren\')\nst.write(\'10. street music\')\nst.write(\'Please record or upload the audio sample\')\nfile=st.file_uploader("Choose a file",type=[".wav",".mp3"])\nif file is None:\n        st.warning(\'Please upload a valid file!!\')\nelse:\n        st.audio(file,format="audio/wav")\n        audio, sample_rate = librosa.load(file) \n        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)\n        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)\n        mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)\n        predicted_label=model.predict(mfccs_scaled_features)\n        a=np.argmax(predicted_label,axis=1)\n        if st.button(\'predict\'):\n            if a==0:\n                out=\'air conditioner\'\n            elif a==1:\n                out=\'car horn\'\n            elif a==2:\n                out=\'children laying\'\n            elif a==3:\n                out=\'dog bark\'\n            elif a==4:\n                out=\'drilling\'\n            elif a==5:\n                out=\'engine idling\'\n            elif a==6:\n                out=\'gun shot\'\n            elif a==7:\n                out=\'jackhammer\'\n            elif a==8:\n                out=\'siren\'\n            else:\n                out=\'street music\'\n            st.success(\'the given audio sample is predicted as:- \'+str(out))')


# In[84]:


get_ipython().system('streamlit run audio_test.py')

