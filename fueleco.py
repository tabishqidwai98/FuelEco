import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns

st.title("Fuel Economy Dataset")

st.text("lets do some prediction")

DATA_PATH = "vehicles.csv"
DATA_PATH1 = "mpg.csv"

def load_data(rows = None):
    data = pd.read_csv(DATA_PATH, nrows = rows)
    data.rename(lambda col : str(col).lower(), axis ='columns', inplace = True)
    return data
def load_data1(rows = None):
    data = pd.read_csv(DATA_PATH1, nrows = rows)
    data.rename(lambda col : str(col).lower(), axis ='columns', inplace = True)
    return data

data_load_state = st.text('loading fuel data...')
data = load_data(10000)
data_load_state.text("loaded the Dataset")
data2 = load_data1(10000)
data_load_state.text("loaded the Dataset2")

st.subheader("View Raw Data")
st.write(data)
st.write(data2)

st.markdown('<h1 style="color:red">Histogram distribution in Fuel</h1>',unsafe_allow_html=True)
column = st.selectbox("select a column from the dataset", ['barrels08'])
bins = st.slider("select number of bins",5,100,20)

histogram = data[column].plot.hist(bins=bins, title=f'{column} histogram analysis')
st.pyplot()

column = st.selectbox("select a column from the dataset", ['year'])
bins = st.slider("select number of bins",10,110,40)
histogram = data2[column].plot.hist(bins=bins, title=f'{column} histogram analysis')
st.pyplot()

st.subheader("Column Comparison in Dataset")
xcol = st.sidebar.selectbox("X axis :select a column from the dataset", data.columns)
ycol = st.sidebar.selectbox("Y axis :select a column from the dataset", data.columns)
fig = px.scatter(data,x=xcol, y=ycol,color='year')
st.plotly_chart(fig,use_container_width=True)

st.subheader("Correlation in columns with Seaborn")
xcol = st.selectbox("X axis", data.columns)
ycol = st.selectbox("Y axis", data.columns)
sns.regplot(x=xcol, y=ycol,data=data)
st.pyplot()

data2.manufacturer.value_counts().head().plot(kind='pie')
st.pyplot()