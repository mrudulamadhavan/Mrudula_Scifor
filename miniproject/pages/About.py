import streamlit as st
from matplotlib import image
import pandas as pd
import plotly.express as px
import os

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath("miniproject//pages//About.py"))

# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")


st.header(":green[Overview ]")
st.write(' The integration of rental bike systems into urban landscapes has ushered in a new era of convenience and sustainability in transportation. These systems, characterized by automated processes and strategically placed kiosks across cities, provide users with flexible and eco-friendly options for short-distance travel. However, the seamless functioning of these bike-sharing programs relies heavily on the accurate prediction of bike demand at different hours. In this dynamic urban environment, anticipating the right number of bikes needed ensures a stable supply, minimizes waiting times, and optimizes the overall user experience. This introduction sets the stage for exploring the crucial role of predictive models in the success of rental bike systems and their impact on shaping contemporary urban mobility.')

st.header(":green[Problem Statement ]")
st.write(' The critical challenge in bike rental systems is the accurate prediction of demand at varying hours. This problem statement underscores the necessity for sophisticated predictive models to anticipate the required bike count, ensuring a stable supply and minimizing user wait times. Addressing this dynamic aspect of urban mobility, influenced by factors like historical patterns and weather conditions, is crucial for the effective functioning of bike-sharing programs, emphasizing the need for innovative solutions to enhance overall system performance and user satisfaction.')

st.write('Create a web application for bike-sharing demand prediction using Streamlit and Python')
    
st.header(":green[About Dataset]")
st.write("The Bike Sharing Demand dataset contains information about bike rental in Seoul from 2017-2018. It includes hourly observations of 14 attributes, such as the date, time, number of rented bikes, weather information (Temperature, Humidity, Windspeed, Visibility, Dewpoint, Solar radiation, Snowfall, Rainfall) and other factors that may influence bike rental demand.") 
IMAGE_PATH2 = os.path.join(dir_of_interest, "images", "dataset.png")
img = image.imread(IMAGE_PATH2)
st.image(img) 

st.subheader( 'red[Target Column :] blue[Rented Bike Count]')

DATA_PATH = os.path.join(dir_of_interest, "data", "BikeData.csv")
df = pd.read_csv(DATA_PATH)
df = df[['Date', 'Hour', 'Temperature(°C)', 'Humidity(%)','Wind speed (m/s)', 'Visibility (10m)',
            'Dew point temperature(°C)','Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 
            'Seasons','Holiday', 'Functioning Day','Rented Bike Count']]


status = st.radio("Click to know more : ", ('Overview of dataset','Shape','Summary','Descriptive Statistics','Unique values of Attributes'))
st.write('-----------------------------------------------------------------------------------')
if (status == 'Overview of dataset'):    
    st.dataframe(df.head(8))

elif (status == 'Shape'):
    rows = df.count()[0]
    columns = df.shape[1] - 1
    st.text(f'Number of Rows  : {rows}')
    st.text(f' Number of Columns  : {columns}')
      
elif (status == 'Summary'):
    IMAGE_PATH3 = os.path.join(dir_of_interest, "images", "info.png")
    img = image.imread(IMAGE_PATH3)
    st.image(img)   
    
elif (status == 'Descriptive Statistics') :    
    x = df.describe(include = "object")
    st.table(x)
    y = df.describe().T
    st.table(y)
else:
    for column in df.columns.tolist():
        st.write("Unique Value Count ")
        st.write(f"  * In {column} : {df[column].nunique()} values")



def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

pub = convert_df(df)
st.write('-----------------------------------------------------------------------------------')
st.download_button(
    label="Download Dataset",
    data = pub,
    file_name='BikeData.csv',
    mime='text/csv',
)




