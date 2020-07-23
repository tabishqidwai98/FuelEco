import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
import pickle

st.title("Fuel Economy Dataset")

st.text("lets do some prediction")

DATA_PATH = "vehicles.csv"
DATA_PATH1 = "mpg.csv"

page = st.sidebar.selectbox("select a page",['Data Anaytics','AI Application','Diesel Prediction']) 
if page =='Data Anaytics':    

    @st.cache
    def load_data(rows = None):
        data = pd.read_csv(DATA_PATH, nrows = rows)
        data.rename(lambda col : str(col).lower(), axis ='columns', inplace = True)
        return data

    @st.cache
    def load_data1(rows = None):
        data = pd.read_csv(DATA_PATH1, nrows = rows)
        data.rename(lambda col : str(col).lower(), axis ='columns', inplace = True)
        return data


    data_load_state = st.text('loading fuel data...')
    data = load_data(10000)
    data2 = load_data1(10000)
    data_load_state.text("loaded the Datasets")

    st.subheader("View Raw Data")
    if st.checkbox("data 1"):
        st.write(data)
    #cols = ["class","displ","trans","cyl","trans.dscr","cty","hwy"]
    #st_ms = st.multiselect("Columns", data2.columns.tolist(), default=cols)
    #st.write(data2)
    if st.checkbox("data 2"):
        st.write(data2)

    st.markdown('<h1 style="color:red">Histogram distribution in Fuel</h1>',unsafe_allow_html=True)
    st.text("Dataset : data 1")
    column = st.selectbox("select a column from the dataset", ['barrels08','city08','co2TailpipeGpm','comb08','displ','engId','fuelCost08'])
    bins = st.slider("select number of bins",5,100,20)

    histogram = data[column].plot.hist(bins=bins, title=f'{column} histogram analysis')
    st.pyplot()

    st.text("Dataset : data 2")
    column = st.selectbox("select a column from the dataset", ['year'])
    bins = st.slider("select number of bins",10,110,40)
    histogram = data2[column].plot.hist(bins=bins, title=f'{column} histogram analysis')
    st.pyplot()
    
    #st.subheader("Column Comparison in Dataset with bar graph")
    #xcolumn = st.selectbox("select a column from the dataset", data.columns)
    #ycolumn = st.selectbox("select a column from dataset", data.columns)
    #plt.bar(xcolumn, ycolumn, width=20)
    #st.pyplot()
    # Data is not loading in the above graph even after re-running the streamlit

    st.subheader("Column Compariso through scatter plot")
    st.text("Dataset : data 1")
    xcolumn = st.selectbox("select a column from the dataset for comparison", data.columns)
    ycolumn1 = st.selectbox("select a first column for comparision with first column selected", data.columns)
    ycolumn2 = st.selectbox("select a second column for comparosion with first column selected", data.columns)
    ycolumn3 = st.selectbox("select a third column for comparosion with first column selected", data.columns)

    plt.scatter(xcolumn, ycolumn1, data = data )
    plt.scatter(xcolumn, ycolumn2, data = data )
    plt.scatter(xcolumn, ycolumn3, data = data )
    plt.xlabel(xcolumn)
    plt.ylabel('Consumption')
    plt.title('Comparison of columns of Dataset 1')
    plt.grid(True)
    plt.legend()
    st.pyplot()

    st.text("Dataset : data 2")
    xcolumn = st.selectbox("select a column from the dataset for comparison", data2.columns)
    ycolumn1 = st.selectbox("select a first column for comparision with first column selected", data2.columns)
    ycolumn2 = st.selectbox("select a second column for comparosion with first column selected", data2.columns)
    ycolumn3 = st.selectbox("select a third column for comparosion with first column selected", data2.columns)

    plt.scatter(xcolumn, ycolumn1, data = data2 )
    plt.scatter(xcolumn, ycolumn2, data = data2 )
    plt.scatter(xcolumn, ycolumn3, data = data2 )
    plt.xlabel(xcolumn)
    plt.ylabel('Consumption')
    plt.title('Comparison of columns of Dataset 2')
    plt.grid(True)
    plt.legend()
    st.pyplot()

    st.subheader("Column Comparison in Dataset")
    st.text("Dataset : data 1")
    st.sidebar.header("Comparision Graph")
    xcol = st.sidebar.selectbox("X axis :select a column from the dataset", data.columns)
    ycol = st.sidebar.selectbox("Y axis :select a column from the dataset", data.columns)
    fig = px.scatter(data,x=xcol, y=ycol,color='year')
    st.plotly_chart(fig,use_container_width=True)

    st.text("Dataset : data 2")
    xcol = st.sidebar.selectbox("X axis :select a column from the dataset", data2.columns)
    ycol = st.sidebar.selectbox("Y axis :select a column from the dataset", data2.columns)
    fig = px.scatter(data2,x=xcol, y=ycol,color='year')
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Pie chart comparision of vehicle")
    st.text("Dataset : data 1")
    data.model.value_counts().head().plot(kind='pie')
    st.pyplot()

    st.text("Dataset : data 2")
    data2.manufacturer.value_counts().head().plot(kind='pie')
    st.pyplot()

    #sns.set()
    #sns.heatmap(data.corr(), annot=True, fmt='.2f')
    #st.pyplot()

    #sns.heatmap(data2.corr(), annot=True, fmt='.2f')
    #st.pyplot()

    st.subheader("Column Comparison in Dataset through jointplot")
    st.text('Dataset : Data 1')
    xcolm = st.selectbox("X axis : select a column from the dataset", data.columns)
    ycolm = st.selectbox("Y axis : select a column from the dataset", data.columns)
    sns.jointplot(x=xcolm, y=ycolm, data=data)
    st.pyplot()

    st.text('Dataset : Data 2')
    xcolm = st.selectbox("X axis : select a column from the dataset", data2.columns)
    ycolm = st.selectbox("Y axis : select a column from the dataset", data2.columns)
    sns.jointplot(x=xcolm, y=ycolm, data=data2)
    st.pyplot()

    #xcolm = st.selectbox("X axis : select a column from the dataset ", data.columns)
    #ycolm = st.selectbox("Y axis : select a column from the dataset ", data.columns)
    #sns.catplot(x=xcolm, y=ycolm, kind='violin', data=data)
    #st.pyplot()

elif page == 'AI Application':
    # intro

    # form
    st.subheader('please enter data to predict the AI prediction')
    
    co2TailpipeGpm = st.number_input('Carbondioxide Emission Values Gallons per mile',min_value=0.0, max_value=1269.57)
    displ= st.number_input('Displacement is the total volume of all the cylinders in an engine',min_value=0.0, max_value=8.40)
    barrels08 = st.number_input('Cost of barrel',min_value=0.06, max_value=47.087)
    cylinders = st.number_input('Number of cylinders in a car',min_value=2.00, max_value=16.00)
    fuelCost08 = st.number_input('Cost of Fuel',min_value=450.0, max_value=5650.00)
    rangeHwy = st.number_input('Mileage over Highway',min_value=0.0, max_value=358.55)
    comb08U = st.number_input('Unadjusted estimated combined miles per gallon',min_value=0.0, max_value=140.560)
    UHighway = st.number_input('unadjusted highway miles per gallon',min_value=0.0, max_value=187.10)
    highway08 = st.number_input('estimated highway miles per gallon',min_value=9.0, max_value=132.00)
    comb08 = st.number_input('estimated combined miles per gallon ',min_value=7.0, max_value=141.00)
    city08 = st.number_input('estimated city miles per gallon',min_value=6.0, max_value=150.00)
    if st.button('predict'):
        with open('scaler.pk','rb') as f:
            scaler = pickle.load(f)
        
        with open('model.pk','rb') as f:
            model = pickle.load(f)
        
        if scaler and model:
            data = np.array([co2TailpipeGpm, displ,barrels08,cylinders,fuelCost08, rangeHwy,comb08U, UHighway ,highway08, comb08,city08] )
            features = scaler.transform(data.reshape(1,-1))
            prediction = model.predict(features)
            st.header("prediction city mpg of vehicle")
            st.success(prediction[0])

elif page=='Diesel Prediction':

    st.header("Predict Diesel Prices")
    cities = ['Mumbai','Delhi','Chennai','Kolkata']
    date = st.date_input("select a date")
    city = st.selectbox("select a Metro city",cities)
    if date and city and st.button("Predict"):
        with open('models/diesel_price_prediction.pk', 'rb') as f:
            model_dict = pickle.load(f)
            st.write(model_dict)
            # convert date to ordinal
        date_o = date.toordinal()
        city_dummies = model_dict['city_encoder'].transform([[city]]).toarray()
        st.write("date as ordinal",date_o)
        st.write("city as dummy variable", city_dummies)
        X = np.append(city_dummies[:,:-1],[[date_o]],axis=1)
        st.write("Our data for input",X)
        scaled_X = model_dict['scale'].transform(X)
        result = model_dict['model'].predict(scaled_X)
        st.write(f"price predicted is {result[0]:.2f}")