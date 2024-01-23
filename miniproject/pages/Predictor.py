#Importing Libraries
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import image

st.set_page_config(page_title="Bike Sharing Demand Predictor", page_icon="https://www.shutterstock.com/image-vector/bicycle-filled-outline-icons-vector-illustration-1772580485",
                   layout="wide")

#import model
st.title(":red[  Bike Sharing Demand Predictor  ]")

#resources path

FILE_DIR1 = os.path.dirname(os.path.abspath("miniproject//pages//Predictor.py"))
FILE_DIR = os.path.join(FILE_DIR1,os.pardir)
dir_of_interest = os.path.join(FILE_DIR, "resources")

IMAGE_PATH = os.path.join(dir_of_interest, "images", "predict.png")
img = image.imread(IMAGE_PATH)
st.image(img) 

DATA_PATH = os.path.join(dir_of_interest, "data")

#Load data
DATA_PATH1=os.path.join(DATA_PATH, "BikeData.csv")
df=pd.read_csv(DATA_PATH1,encoding='latin')
df1 = df.copy

df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
df['Month'] = df['Date'].dt.month.astype('category')
df['Weekday'] = df['Date'].dt.day_name().astype('category')


def prediction(season,month,weekday,hour,temperature,humidity,visibility,windspeed,solarrdn,rainfall,snowfall):
    prediction = xgb.predict([[season,month,weekday,hour,temperature,humidity,visibility,windspeed,solarrdn,rainfall,snowfall]])
    print(prediction)
    return prediction

# ------------------------------------------------------------------------------------------------------------
st.subheader("Enter the Location details :")

col1,col2 = st.columns(2)
with col1:  
    season = st.selectbox("Season", df["Seasons"].unique())
with col2:  
    month = st.selectbox("Month", df["Month"].unique())


col3,col4 = st.columns(2)
with col3:  
    weekday = st.selectbox("Weekday", df["Weekday"].unique())
with col4:  
    hour = st.selectbox("Hour of the day", df["Hour"].unique())

left_column, middle_column, right_column = st.columns(3)
with left_column:
    temperature = st.text_input("Temperature(°C)  ",value=0.0) 
with middle_column:
    humidity = st.text_input("Humidity(%) ",value=0)
with right_column:
    visibility = st.text_input("Visibility(10m)  ",value=0)

col1,col2 = st.columns(2)
with col1:  
    windspeed = st.number_input("Wind speed (m/s) ")
with col2:
    solarrdn = st.number_input("Solar Radiation (MJ/m2) ")


col3,col4 = st.columns(2)
with col3:  
    rainfall = st.number_input("Rainfall(mm) ")
with col4:
    snowfall = st.number_input("Snowfall (cm) ")

bike_count = ''
#---------------------------------------------------------------------------------------------------------------
#Create dataframe using all these values
sample=pd.DataFrame({"Seasons":[season],"Month":[month],"Weekday":[weekday],"Hour":[hour],
                    "Temperature(°C)":float(temperature),"Humidity(%)":float(humidity),"Visibility (10m)":float(visibility),
                    "Wind speed (m/s)":[windspeed],"Solar Radiation (MJ/m2)":[solarrdn],
                    "Rainfall(mm)":[rainfall], "Snowfall (cm)":[snowfall]})

#Function to change season to number
def replace_season(season):    
    if season =='Summer':
        return 1.0
    elif season =='Winter':
        return 2.0
    elif season =='Spring':
        return 3.0
    else:
        return 4.0
df['Seasons'] = df['Seasons'].apply(replace_season)
sample['Seasons'] = sample['Seasons'].apply(replace_season)

def replace_weekday(weekday):
    if weekday =='Sunday':
        return 1.0
    elif weekday=='Monday':
        return 2.0
    elif weekday=='Tuesday':
        return 3.0
    elif weekday=='Wednesday':
        return 4.0
    elif weekday=='Thursday':
        return 5.0
    elif weekday =='Friday':
        return 6.0
    else:
        return 7.0
df['Weekday'] = df['Weekday'].apply(replace_weekday).astype('float64')
sample['Weekday'] = sample['Weekday'].apply(replace_weekday).astype('float64')

# Function to distribute hour
def distribute_hour(hour):
    if 17 <= hour <= 22:
        return 4.0  # 'Evening'
    elif 7 <= hour <= 10:
        return  2.0 # 'Morning'
    elif 11 <= hour <= 16:
        return  3.0 #'Afternoon'
    else:
        return 1.0  #'Night'

#hour_map = {'Night': 1.0,"Morning":2.0,"Afternoon":3.0, "Evening":4.0}
# Apply the hour function
df['Hour'] = df['Hour'].apply(distribute_hour).astype('float64')
sample['Hour'] = sample['Hour'].apply(distribute_hour).astype('float64')

df['Month'] = df['Month'].astype('float64')
sample['Month'] = sample['Month'].astype('float64')

df['Wind speed (m/s)'] = np.sqrt(df['Wind speed (m/s)'])
sample['Wind speed (m/s)'] = np.sqrt(sample['Wind speed (m/s)'])
# Standardizing the required column
df['Temperature(°C)'] = StandardScaler().fit_transform(df['Temperature(°C)'].values.reshape(-1, 1))
sample['Temperature(°C)'] = StandardScaler().fit_transform(sample['Temperature(°C)'].values.reshape(-1, 1))
df['Humidity(%)'] = StandardScaler().fit_transform(df['Humidity(%)'].values.reshape(-1, 1))
sample['Humidity(%)'] = StandardScaler().fit_transform(sample['Humidity(%)'].values.reshape(-1, 1))

# Normalizing the required column
df['Visibility (10m)'] = StandardScaler().fit_transform(df['Visibility (10m)'].values.reshape(-1, 1))
sample['Visibility (10m)'] = StandardScaler().fit_transform(sample['Visibility (10m)'].values.reshape(-1, 1))

#Split data into X and y
X=df.drop('Rented Bike Count', axis=1).values
y=df['Rented Bike Count'].values

#Standarizing the features
std=StandardScaler()
std_fit=std.fit(X)
X=std_fit.transform(X)

# Splitting data into 75:25 ratio
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)

xgb = pickle.load(open('miniproject/xgb_model.pkl','rb'))

#Train the model
xgb=XGBRegressor(learning_rate=0.15, n_estimators=50, max_leaves=0, random_state=42)
xgb.fit(X,y)

#Standardize the features
sample=sample.values
sample=std_fit.transform(sample)

#Prediction
if st.button('Predict Demand'):
    price=xgb.predict(sample)
    bike_cnt=bike_count[0].round(2)    
    st.subheader(":blue[The Predicted Value for Bike Rentals :] :green[{}]".format("$ "+str(bike_cnt)))
else:
    pass
