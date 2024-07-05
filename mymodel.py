import streamlit as st
import pandas as pd
import pickle

df = pd.read_csv('electricity_usage.csv')

from PIL import Image 
im = Image.open("tnb_logo.png")
st.image(im, width=180, caption="TNB Logo")


col1, col2 = st.columns(2)

with col1:
    df_city = df.groupby('City')['MonthlyHours'].sum().reset_index()
    st.bar_chart(df_city, x="City", y="MonthlyHours")

with col2:
    df_city2 = df.groupby('City')['ElectricityBill'].sum().reset_index()
    st.bar_chart(df_city2, x="City", y="ElectricityBill")


fan = st.slider("Number of Fan", 5,20, value=10)
fridge = st.slider("Number of Refrigerator", 18,23, value=20)
aircon = st.slider("Number of AirCon", 0,3, value=1)
tv = st.slider("Number of TV", 5,20, value=3)
hour = st.slider("Number of Hour", 390,600, value=400)

model = pickle.load(open("LRmodel.pkl", "rb"))

new_data = {'Fan':[fan], 
            'Refrigerator':[fridge], 
            'AirConditioner': [aircon], 
            'Television':[tv], 
            'MonthlyHours':[hour]}

pred = model.predict(pd.DataFrame.from_dict(new_data))
pred = round(pred[0], 2)
st.subheader('The predicted Electricity Bill is ' + str(pred))

