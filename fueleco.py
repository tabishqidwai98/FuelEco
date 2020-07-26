import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
import pickle
from helper import cleanChangeVal

st.markdown('<h1 style="color:red">Fuel Economy prediction</h1>',unsafe_allow_html=True)

st.text("lets do some prediction")

vehiclesUS = "vehicles.csv"
mpgUS = "mpg.csv"
crudeoil = "datasets/crudeoil.csv"
CrudeoilvsDiesel = "datasets/CrudeoilvsDiesel.csv"
CrudeoilvsGasoline = "datasets/CrudeoilvsGasoline.csv"
Diesel = "datasets/Diesel(new).csv"
petrol = "datasets/petrol(new).csv"

st.sidebar.image('fuel.png',use_column_width=True)
page = st.sidebar.selectbox("select a page",['Data Anaytics','AI Application','Diesel Prediction']) 
if page =='Data Anaytics':    

    @st.cache()
    def load_data_vehiclesUs(rows = None):
        data = pd.read_csv(vehiclesUS, nrows = rows)
        data.rename(lambda col : str(col).lower(), axis ='columns', inplace = True)
        return data

    def load_dataMpgUS_mpgUS(rows = None):
        data = pd.read_csv(mpgUS, nrows = rows)
        data.rename(lambda col : str(col).lower(), axis ='columns', inplace = True)
        return data
    
    def load_data2_crudeoil(rows = None):
        df = pd.read_csv(crudeoil,parse_dates=['Month'],dayfirst=True,index_col='Month')
        df.columns = ['Crude_Oil_Price','Diesel_Price']
        df.Crude_Oil_Price = df.Crude_Oil_Price.apply(lambda val:float(val.replace(',','')))
        df.rename(lambda col : str(col).lower(), axis ='columns', inplace = True)
        return df

    def load_data3_CrudeoilvsDiesel(rows = None):
        data = pd.read_csv(CrudeoilvsDiesel, nrows = rows, parse_dates=['Month'],dayfirst=True)
        return data

    def load_data4_CrudeoilvsGasoline(rows = None):
        data = pd.read_csv(CrudeoilvsGasoline, nrows = rows)
        data.rename(lambda col : str(col).lower(), axis ='columns', inplace = True)
        return data

    def load_data5_Diesel(rows = None):
        data = pd.read_csv(Diesel, nrows = rows)
        data.rename(lambda col : str(col).lower(), axis ='columns', inplace = True)
        return data

    def load_data6_petrol(rows = None):
        data = pd.read_csv(petrol, nrows = rows)
        data.rename(lambda col : str(col).lower(), axis ='columns', inplace = True)
        return data

    data_load_state = st.text('loading fuel data...')
    datavehiclesUS = load_data_vehiclesUs(10000)
    dataMpgUS = load_dataMpgUS_mpgUS(10000)
    datacrudeoil = load_data2_crudeoil()
    dataCrudeoilvsDiesel = load_data3_CrudeoilvsDiesel()
    dataCrudeoilvsGasoline = load_data4_CrudeoilvsGasoline()
    dataDiesel = load_data5_Diesel()
    datapetrol = load_data6_petrol()
    data_load_state.text("loaded the Datasets")

    st.subheader("View Raw Data")
    if st.checkbox("vehiclesUs"):
        st.write(data)
    #cols = ["class","displ","trans","cyl","trans.dscr","cty","hwy"]
    #st_ms = st.multiselect("Columns", dataMpgUS.columns.tolist(), default=cols)
    #st.write(dataMpgUS)
    if st.checkbox("mpgUS"):
        st.write(dataMpgUS)

    if st.checkbox("crudeoil"):
        st.write(datacrudeoil)
    
    if st.checkbox("CrudeoilvsDiesel"):
        st.write(dataCrudeoilvsDiesel)
        st.write(dataCrudeoilvsDiesel.info)

    if st.checkbox("CrudeoilvsGasoline"):
        st.write(dataCrudeoilvsGasoline)

    if st.checkbox("Diesel"):
        st.write(dataDiesel)
    
    if st.checkbox("petrol"):
        st.write(datapetrol)

    st.subheader("Histogram distribution in Fuel")
    st.text("Dataset : vehiclesUs")
    column = st.selectbox("select a column from the dataset", ['barrels08','city08','co2TailpipeGpm','comb08','displ','engId','fuelCost08'])
    bins = st.slider("select number of bins",5,100,20)
    histogram = datavehiclesUS[column].plot.hist(bins=bins, title=f'{column} histogram analysis')
    st.pyplot()

    st.text("Dataset : MpgUS")
    column = st.selectbox("select a column from the dataset", ['year'])
    bins = st.slider("select number of bins",10,110,40)
    histogram = dataMpgUS[column].plot.hist(bins=bins, title=f'{column} histogram analysis')
    st.pyplot()

    st.text("Dataset : crudeoil")
    column = st.selectbox("select a column from the dataset", datacrudeoil.columns)
    bins = st.slider("select number of bins",10,120,10)
    histogram = datacrudeoil[column].plot.hist(bins=bins, title=f'{column} histogram analysis')
    st.pyplot()


    st.text("Dataset : CrudeoilvsDiesel")
    column = st.selectbox("select a column from the dataset", dataCrudeoilvsDiesel.columns)
    bins = st.slider("select number of bins",20,110,50)
    histogram = dataCrudeoilvsDiesel[column].plot.hist(bins=bins, title=f'{column} histogram analysis')
    st.pyplot()

    st.text("Dataset : CrudeoilvsGasoline")
    column = st.selectbox("select a column from the dataset", dataCrudeoilvsGasoline.columns)
    bins = st.slider("select number of bins",15,110,15)
    histogram = dataCrudeoilvsGasoline[column].plot.hist(bins=bins, title=f'{column} histogram analysis')
    st.pyplot()

    st.text("Dataset : Diesel")
    column = st.selectbox("select a column from the dataset", dataDiesel.columns)
    bins = st.slider("select number of bins",10,130,20)
    histogram = dataDiesel[column].plot.hist(bins=bins, title=f'{column} histogram analysis')
    st.pyplot()

    st.text("Dataset : petrol")
    column = st.selectbox("select a column from the dataset", datapetrol.columns)
    bins = st.slider("select number of bins",5,100,30)
    histogram = datapetrol[column].plot.hist(bins=bins, title=f'{column} histogram analysis')
    st.pyplot()
    
    #st.subheader("Column Comparison in Dataset with bar graph")
    #xcolumn = st.selectbox("select a column from the dataset", data.columns)
    #ycolumn = st.selectbox("select a column from dataset", data.columns)
    #plt.bar(xcolumn, ycolumn, width=20)
    #st.pyplot()
    # Data is not loading in the above graph even after re-running the streamlit

    st.subheader("Column Comparison through scatter plot")
    st.text("Dataset : vehiclesUs")
    xcolumn = st.selectbox("select a column from the dataset for comparison", datavehiclesUs.columns)
    ycolumn1 = st.selectbox("select a first column for comparision with first column selected", datavehiclesUs.columns)
    ycolumn2 = st.selectbox("select a second column for comparosion with first column selected", datavehiclesUs.columns)
    ycolumn3 = st.selectbox("select a third column for comparosion with first column selected", datavehiclesUs.columns)

    plt.scatter(xcolumn, ycolumn1, data = datavehiclesUs )
    plt.scatter(xcolumn, ycolumn2, data = datavehiclesUs )
    plt.scatter(xcolumn, ycolumn3, data = datavehiclesUs )
    plt.xlabel(xcolumn)
    plt.ylabel('Consumption')
    plt.title('Comparison of columns of Dataset 1')
    plt.grid(True)
    plt.legend()
    st.pyplot()

    st.text("Dataset : MpgUS")
    xcolumn = st.selectbox("select a column from the dataset for comparison", dataMpgUS.columns)
    ycolumn1 = st.selectbox("select a first column for comparision with first column selected", dataMpgUS.columns)
    ycolumn2 = st.selectbox("select a second column for comparosion with first column selected", dataMpgUS.columns)
    ycolumn3 = st.selectbox("select a third column for comparosion with first column selected", dataMpgUS.columns)

    plt.scatter(xcolumn, ycolumn1, data = dataMpgUS )
    plt.scatter(xcolumn, ycolumn2, data = dataMpgUS )
    plt.scatter(xcolumn, ycolumn3, data = dataMpgUS )
    plt.xlabel(xcolumn)
    plt.ylabel('Consumption')
    plt.title('Comparison of columns of Dataset 2')
    plt.grid(True)
    plt.legend()
    st.pyplot()

    st.subheader("Column Comparison in Dataset")
    st.text("Dataset : vehiclesUs")
    st.sidebar.header("Comparision Graph")
    xcol = st.sidebar.selectbox("X axis :select a column from the dataset", data.columns)
    ycol = st.sidebar.selectbox("Y axis :select a column from the dataset", data.columns)
    fig = px.scatter(data,x=xcol, y=ycol,color='year')
    st.plotly_chart(fig,use_container_width=True)

    st.text("Dataset : MpgUS")
    xcol = st.sidebar.selectbox("X axis :select a column from the dataset", dataMpgUS.columns)
    ycol = st.sidebar.selectbox("Y axis :select a column from the dataset", dataMpgUS.columns)
    fig = px.scatter(dataMpgUS,x=xcol, y=ycol,color='year')
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Pie chart comparision of vehicle")
    st.text("Dataset : vehiclesUs")
    data.model.value_counts().head().plot(kind='pie')
    st.pyplot()

    st.text("Dataset : MpgUS")
    dataMpgUS.manufacturer.value_counts().head().plot(kind='pie')
    st.pyplot()

    #sns.set()
    #sns.heatmap(data.corr(), annot=True, fmt='.2f')
    #st.pyplot()

    #sns.heatmap(dataMpgUS.corr(), annot=True, fmt='.2f')
    #st.pyplot()

    st.subheader("Column Comparison in Dataset through jointplot")
    st.text('Dataset : vehiclesUs')
    xcolm = st.selectbox("X axis : select a column from the dataset", datavehiclesUs.columns)
    ycolm = st.selectbox("Y axis : select a column from the dataset", datavehiclesUs.columns)
    sns.jointplot(x=xcolm, y=ycolm, data=datavehiclesUs)
    st.pyplot()

    st.text('Dataset : MpgUS')
    xcolm = st.selectbox("X axis : select a column from the dataset", dataMpgUS.columns)
    ycolm = st.selectbox("Y axis : select a column from the dataset", dataMpgUS.columns)
    sns.jointplot(x=xcolm, y=ycolm, data=dataMpgUS)
    st.pyplot()

    #xcolm = st.selectbox("X axis : select a column from the dataset ", data.columns)
    #ycolm = st.selectbox("Y axis : select a column from the dataset ", data.columns)
    #sns.catplot(x=xcolm, y=ycolm, kind='violin', data=data)
    #st.pyplot()

elif page == 'Mileage Prediction':
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

elif page=='Petrol Prediction':

    st.header("Predict Petrol Prices")
    cities = ['Mumbai','Delhi','Chennai','Kolkata']
    date = st.date_input("select a date")
    city = st.selectbox("select a Metro city",cities)
    if date and city and st.button("Predict"):
        with open('models/petrol_price_prediction.pk', 'rb') as f:
            model_dict = pickle.load(f)
            st.write(model_dict)
            #convert date to ordinal
        date_o = date.toordinal()
        city_dummies = model_dict['city_encoder'].transform([[city]]).toarray()
        st.write("date as ordinal",date_o)
        st.write("city as dummy variable", city_dummies)
        X = np.append(city_dummies[:,:-1],[[date_o]],axis=1)
        st.write("Our data for input",X)
        scaled_X = model_dict['scale'].transform(X)
        result = model_dict['model'].predict(scaled_X)
        st.write(f"price predicted is {result[0]:.2f}")