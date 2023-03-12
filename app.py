import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from PIL import Image
model = pickle.load(open('xgb.pkl', 'rb'))

st.title('Air Quality Index Prediction')
st.sidebar.header('Air Quality Index Data')
image = Image.open('image1.png')
st.image(image, '')

# FUNCTION


def user_report():
    avg_temp = st.sidebar.slider('Average Temperature', 15, 30, 15)
    max_temp = st.sidebar.slider('Maximum Temperature', 20, 40, 20)
    min_temp = st.sidebar.slider('Minimum Temperature', 10, 20, 10)
    humidity = st.sidebar.slider('Humidity', 20, 100, 20)
    rainfall = st.sidebar.slider('Rainfall', 0, 500, 1)
    visibility = st.sidebar.slider('Visibilty', 0, 10, 1)
    wind_speed = st.sidebar.slider('Wind Speed', 0, 20, 1)
    avg_wind_speed = st.sidebar.slider('Average Wind Speed', 0, 100, 1)

    user_report_data = {
        'T': avg_temp,
        'TM': max_temp,
        'Tm': min_temp,
        'H': humidity,
        'PP': rainfall,
        'VV': visibility,
        'V': wind_speed,
        'VM': avg_wind_speed
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


user_data = user_report()
st.header('Air Quality Data Data')
st.write(user_data)

aqi = model.predict(user_data)
st.subheader('Air Quality Index')
st.subheader(np.round(aqi[0], 2))
