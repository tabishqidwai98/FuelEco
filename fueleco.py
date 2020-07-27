import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
import pickle
from helper import cleanChangeVal

st.markdown('<h1 style="color:red">Fuel Economy prediction</h1>', unsafe_allow_html = True)
st.markdown("Fossil fuel is getting scarce day by day and would get completely depleted in the coming years therefore crude oil prices  fluctuates erratically which generally makes the prices of the Gasoline (Petrol), Diesel, Heavy Fuel Oil & Jet Fuel go up and down.", unsafe_allow_html= True)
st.markdown("This project is created for users to input their respective data into the prediction path and receive the predicted results by analyzing the datasets used in this project also to get a better understanding of the fuel prices through visualizations. ", unsafe_allow_html= True)

vehiclesUS = "vehicles.csv"
mpgUS = "mpg.csv"
crudeoil = "datasets/crudeoil.csv"
CrudeoilvsDiesel = "datasets/CrudeoilvsDiesel.csv"
CrudeoilvsGasoline = "datasets/CrudeoilvsGasoline.csv"
Diesel = "datasets/Diesel(new).csv"
petrol = "datasets/petrol(new).csv"

st.sidebar.image('petrol.png',use_column_width=True)
page = st.sidebar.selectbox("Select a Page",['Data Anaytics','Mileage Prediction','Diesel Prediction','Petrol Prediction']) 
if page =='Data Anaytics':   

    st.header('Data Anaytics')
    st.markdown('Data analytics (DA) is the process of examining data sets in order to draw conclusions about the information they contain, increasingly with the aid of specialized systems and software. Data analytics technologies and techniques are widely used in commercial industries to enable organizations to make more-informed business decisions and by scientists and researchers to verify or disprove scientific models, theories and hypotheses.',unsafe_allow_html= True)

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
        df.columns = ['Crude_Oil_Price','change']
        df.Crude_Oil_Price = df.Crude_Oil_Price.apply(lambda val:float(val.replace(',','')))
        df.change = df.change.apply(cleanChangeVal)
        df.rename(lambda col : str(col).lower(), axis ='columns', inplace = True)
        return df

    def load_data3_CrudeoilvsDiesel(rows = None):
        data = pd.read_csv(CrudeoilvsDiesel, nrows = rows, parse_dates=['Month'],dayfirst=True, index_col='Month')
        data.columns = ['Crude_Oil_Price','Diesel_Price']
        data.Crude_Oil_Price = data.Crude_Oil_Price.apply(lambda val:float(val.replace(',','')))
        return data

    def load_data4_CrudeoilvsGasoline(rows = None):
        data = pd.read_csv(CrudeoilvsGasoline,parse_dates=['Month'],dayfirst=True, index_col='Month' )
        data.columns = ['Crude_Oil_Price','Gasoline_Price']
        data.Crude_Oil_Price = data.Crude_Oil_Price.apply(lambda val:float(val.replace(',','')))
        return data

    def load_data5_Diesel(rows = None):
        data = pd.read_csv(Diesel,parse_dates=['Date'],index_col='Date')
        data.columns = ['city','diesel_price']
        data.rename(lambda col : str(col).lower(), axis ='columns', inplace = True)
        return data

    def load_data6_petrol(rows = None):
        data = pd.read_csv(petrol,parse_dates=['Date'],index_col='Date',)
        data.columns = ['city','petrol_price']
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
    data_load_state.text("The loaded datasets are as folows:-")

    st.subheader("View Raw Data")
    if st.checkbox("vehiclesUs"):
        st.write(datavehiclesUS)
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
    st.write(dataMpgUS.head())
    column = st.selectbox("select a column from the dataset", ['year'])
    bins = st.slider("select number of bins",10,110,40)
    histogram = dataMpgUS[column].plot.hist(bins=bins, title=f'{column} histogram analysis')
    st.pyplot()

    st.text("Dataset : crudeoil")
    st.write(datacrudeoil.head())
    column = st.selectbox("select a column from the dataset", datacrudeoil.columns)
    datacrudeoil.resample('M').mean().plot(kind='line',style='ro--',title='Avg Petrol price every month',figsize=(8,5))
    plt.xlabel('Year')
    st.pyplot()


    st.text("Dataset : CrudeoilvsDiesel")
    st.write(dataCrudeoilvsDiesel.head())
    fig = px.scatter_3d(dataCrudeoilvsDiesel, x=dataCrudeoilvsDiesel.index, y='Crude_Oil_Price',z='Diesel_Price',color='Diesel_Price',size='Crude_Oil_Price',width=500,)
    st.plotly_chart(fig)

    st.text("Dataset : CrudeoilvsGasoline")
    st.write(dataCrudeoilvsGasoline.head())
    fig = px.scatter_3d(dataCrudeoilvsGasoline, x=dataCrudeoilvsGasoline.index, y='Crude_Oil_Price',z='Gasoline_Price',color='Gasoline_Price',size='Crude_Oil_Price',width=500,)
    st.plotly_chart(fig)

    st.text("Dataset : Diesel")
    st.write(dataDiesel.head())
    cityname = st.selectbox('select a city',dataDiesel.city.unique())
    dataDiesel[dataDiesel['city']==cityname].resample('M').mean().plot(kind='line',style='ro:',title='Avg Diesel price every month',figsize=(8,5))
    plt.xlabel('')
    st.pyplot()

    st.text("Dataset : petrol")
    st.write(datapetrol.head())
    cityname = st.selectbox('select a city name',datapetrol.city.unique())
    datapetrol[datapetrol['city']==cityname].resample('M').mean().plot(kind='line',style='ro:',title='Avg Petrol price every month',figsize=(8,5))
    plt.xlabel('')
    st.pyplot()
    
    #st.subheader("Column Comparison in Dataset with bar graph")
    #xcolumn = st.selectbox("select a column from the dataset", data.columns)
    #ycolumn = st.selectbox("select a column from dataset", data.columns)
    #plt.bar(xcolumn, ycolumn, width=20)
    #st.pyplot()
    # Data is not loading in the above graph even after re-running the streamlit

    st.subheader("Column Comparison through scatter plot")
    st.text("Dataset : vehiclesUs")
    st.write(datavehiclesUS.head())
    xcolumn = st.selectbox("select a column from the dataset for comparison", datavehiclesUS.columns)
    ycolumn1 = st.selectbox("select a first column for comparision with first column selected", datavehiclesUS.columns)
    ycolumn2 = st.selectbox("select a second column for comparosion with first column selected", datavehiclesUS.columns)
    ycolumn3 = st.selectbox("select a third column for comparosion with first column selected", datavehiclesUS.columns)

    plt.scatter(xcolumn, ycolumn1, data = datavehiclesUS )
    plt.scatter(xcolumn, ycolumn2, data = datavehiclesUS )
    plt.scatter(xcolumn, ycolumn3, data = datavehiclesUS )
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
    st.write(datavehiclesUS.head())
    st.header("Comparision Graph")
    xcol = st.selectbox("X axis :choose a column from the ", datavehiclesUS.columns)
    ycol = st.selectbox("Y axis :choose a column from the dataset", datavehiclesUS.columns)
    fig = px.scatter(datavehiclesUS,x=xcol, y=ycol,color='year')
    st.plotly_chart(fig,use_container_width=True)

    st.text("Dataset : MpgUS")
    xcol = st.selectbox("X axis :select a column from the dataset", dataMpgUS.columns)
    ycol = st.selectbox("Y axis :select a column from the dataset", dataMpgUS.columns)
    fig = px.scatter(dataMpgUS,x=xcol, y=ycol,color='year')
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Pie chart comparision of vehicle")
    st.text("Dataset : vehiclesUs")
    datavehiclesUS.model.value_counts().head().plot(kind='pie')
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
    xcolm = st.selectbox("X axis : select a column from the dataset", datavehiclesUS.columns)
    ycolm = st.selectbox("Y axis : select a column from the dataset", datavehiclesUS.columns)
    sns.jointplot(x=xcolm, y=ycolm, data=datavehiclesUS)
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

    st.header('Mileage Prediction')

    st.markdown("Automobile MPG (miles per gallon) prediction is a typical RandomForestRegressor problem, in which several attributes of an automobile's profile information are used to predict another continuous attribute, **mileage**, the fuel consumption in MPG. The training data is available in the OpenEI website, Repository and contains data collected from automobiles of various makes and models. The six input attributes are Carbondioxide Emission Values Gallons per mile, displacement, Displacement is the total volume of all the cylinders in an engine, Cost of barrel, Number of cylinders in a car, Cost of Fuel etc. The output variable to be predicted is the fuel consumption in MPG.")
    
    st.markdown("**let's get the prediction**")
    
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
    
    st.header("Diesel Prediction")
    st.markdown("Diesel fuel is a mixture of hydrocarbons obtained by distillation of crude oil with boiling points in the range of 150°C  to 380°C.")
    st.markdown("This sections shows the predictive prices of diesel in certain cities of India up-to 90.5 percent ccuracy also it can provide past and present prices of diesel up-to 2017.")
    st.markdown("**lets get the prediction**")
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
        st.success(f"price predicted is {result[0]:.2f}")

elif page=='Petrol Prediction':

    st.header("Petrol Prediction")
    st.markdown("All around the world petrol is used as fuel for vehicles. It's one of the main products, which is consumed heavily worldwide.It is a derivative product of crude oil/petroleum. It is derived during fractional distillation process and has a translucent liquid form. It's not used in its crude form. Different additives are added like ethanol to use it as fuel for passenger vehicles. In the US and Latin countries, term gasoline is used, but in Europe and Asian countries it's called petrol.")
    st.markdown("This page shows the predictive prices of petrol in certain cities of India up-to 90.5 percent accuracy also it can provide past and present prices of petrol up-to 2017.")
    st.markdown("**lets get the prediction now!**")
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
        st.success(f"price predicted is {result[0]:.2f}")