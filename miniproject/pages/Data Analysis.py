import streamlit as st
from matplotlib import image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np
import os
from scipy.stats import ttest_ind, f_oneway
st.set_option('deprecation.showPyplotGlobalUse', False)

# absolute path to this file

FILE_DIR = os.path.dirname(os.path.abspath("miniproject//pages//Data Analysis.py"))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")

st.title(":blue[Exploratory Data Analysis]")
DATA_PATH = os.path.join(dir_of_interest, "data", "BikeData.csv")
df = pd.read_csv(DATA_PATH,encoding='latin')

# Convert 'Date' to datetime and create new columns
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.day_name()

# Convert new columns to categorical
for col in ['Year', 'Month', 'Day']:
    df[col] = df[col].astype('category')

st.subheader('Rented Bike Count Distribution')
col1, col2 = st.columns(2)
# Histogram
fig_1 = px.histogram(df, x='Rented Bike Count', nbins=50, title="Histogram")
fig_1.update_layout(bargap=0.2)
col1.plotly_chart(fig_1, use_container_width=True)
# Box plot
fig_2 = px.box(df, x='Rented Bike Count', title="Box Plot")
col2.plotly_chart(fig_2, use_container_width=True)
st.write('Rented Bike Count is right skewed. It means that most of the data falls on the lower end of the scale, and there are relatively fewer instances where a large number of bikes were rented.')
st.write('--------------------------------------------------------------------------------------')
feature = st.selectbox("Choose any feature to see the relationship with Rented Bike Count", ['Seasons', 'Month', 'Weekday'])
col1, col2 = st.columns(2)
fig1 = px.bar(df, x=feature, y='Rented Bike Count', color=feature,title = 'Barplot')
col1.plotly_chart(fig1, use_container_width=True)
fig2 = px.box(df, y=feature, x='Rented Bike Count', color=feature,title = 'Boxplot')
col2.plotly_chart(fig2, use_container_width=True)
#st.write('---')
fig1 = px.bar(df, x='Hour', y='Rented Bike Count', color='Hour',title='Rented Bike Count by Hour')
st.plotly_chart(fig1, use_container_width=True)
st.write('The demand is higher during rush hour (i.e., 7-9 AM and 5-7 PM) when people go to offices/schools and come back in evening.')
st.write("-----------------------------------------------------------------------------------")
# Line plot for Average Rented Bike Count by Hour for each Season using Seaborn
# st.subheader('         Rented Bike Count Trend by Hour for Each Season')
plt.figure(figsize=(12, 6))
sns.lineplot(x="Hour", y="Rented Bike Count", hue="Seasons", data=df)
plt.ylabel("Avg Rented Bike Count")
plt.title("Rented Bike Count Trend by Hour for Each Season", fontsize=15)
# Display the plot in the Streamlit app
st.pyplot()
st.write('The demand is higher during rush hour (i.e., 7-9 AM and 5-7 PM) compared to non-rush hour. The Bike rentals demand trend pattern is the same for all the seasons, only levels are different. Demand level in winter is the lowest and highest in Summer')
st.write("--------------------------------------------------------------------------------")
# Line plot for Average Rented Bike Count by Hour for each weekday using Seaborn
# st.subheader('          Rented Bike Count Trend by Hour for Weekdays ')
plt.figure(figsize=(12, 6))
fig = sns.lineplot(x="Hour", y="Rented Bike Count", hue="Weekday", data=df)
plt.ylabel("Avg Rented Bike Count")
plt.title("Rented Bike Count Trend by Hour for Weekdays", fontsize=15)
# Display the plot in the Streamlit app
st.pyplot()
st.write('The bike rental pattern of weekdays and weekends is different.In the weekend the demand becomes high in the afternoon and the demand for office timings is high during weekdays.')
st.write("------------------------------------------------------------------------------------")
st.subheader('Regression Plots : Rented Bike Count vs Numerical Features')
# Set up subplots
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 8))
# Flatten the axes for easy iteration
axes = axes.flatten()

# Loop through numeric features for regression plots
numeric_features = ['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
                     'Visibility (10m)', 'Dew point temperature(°C)',
                     'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']

for i, feature in enumerate(numeric_features):
    sns.regplot(x=df[feature], y=df['Rented Bike Count'], line_kws={"color": "red"}, ax=axes[i])
    axes[i].set_title(f'Rented Bike Count vs {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Avg Rented Bike Count')

# Adjust layout and display the plots in Streamlit app
plt.tight_layout()
st.pyplot(fig)
st.write('-----------------------------------------')   
st.subheader("Correlation Heatmap")
IMAGE_PATH2 = os.path.join(dir_of_interest, "images", "heatmap.png")
img = image.imread(IMAGE_PATH2)
st.image(img)

st.write("------")
st.title(":blue[Hypothesis Testing Results]")
st.write('Hypothesis testing is a statistical method used to make inferences or draw conclusions about a population based on a sample of data. The process involves formulating a hypothesis about the population parameter, collecting and analyzing data, and then using statistical tests to determine whether there is enough evidence to reject the null hypothesis in favor of an alternative hypothesis.')
st.write('Based on the exploratory analysis conducted using charts, we formulated three hypothetical statements about the dataset and subsequently conducted hypothesis testing through code and statistical methods to draw conclusive results regarding these statements.')

# User selects a hypothetical statement
status = st.radio("Choose any Hypothetical Statements:", ('1: Rented Bike Demand in hot weather is higher compared to demand in cold weather.',
                                                          '2: Rented Bike Demand during rush hour (7-9 AM & 5-7 PM) and non-rush hour are different.',
                                                          '3: Rented Bike Demand is different in different seasons with highest in summer and lowest in winter.'))

if "1: Rented Bike Demand in hot weather is higher compared to demand in cold weather." in status:
    st.write("-------")
    hot_temps = df[df['Temperature(°C)'] >= 20]['Rented Bike Count']
    cold_temps = df[df['Temperature(°C)'] < 20]['Rented Bike Count']
    st.write(':red[Null Hypothesis]: Rented Bike Demand in hot weather is higher compared to demand in cold weather.')
    st.write(':red[Alternate Hypothesis]: No significant difference in demand for Bike rentals in hot weather compared to demand in cold weather.')
    st.text("Test Type : Two-sample T-test")
    st.text('alpha = 0.05')
    t_stat, p_val = ttest_ind(hot_temps, cold_temps, equal_var=False)
    st.write('Test Statistic:', t_stat)
    st.write('p-value:', p_val)
    st.write("-------")
    if p_val < 0.05:
        st.write("Since p-value is less than 0.05, we reject the null hypothesis.")
        st.write("i.e., Rented Bike Demand in hot weather is higher compared to demand in cold weather.")
    else:
        st.write("Since p-value is greater than 0.05, we fail to reject the null hypothesis.")
        st.write("i.e., There is no significant difference in demand for bike rentals in hot weather compared to demand in cold weather.")

elif "2: Rented Bike Demand during rush hour (7-9 AM & 5-7 PM) and non-rush hour are different." in status:
    st.write("-------")
    rush_hour = df[(df['Hour'] >= 7) & (df['Hour'] <= 9) | (df['Hour'] >= 17) & (df['Hour'] <= 19)]['Rented Bike Count']
    non_rush_hour = df[~((df['Hour'] >= 7) & (df['Hour'] <= 9) | (df['Hour'] >= 17) & (df['Hour'] <= 19))]['Rented Bike Count']
    st.write(':red[Null Hypothesis]: Rented Bike Demand during rush hour (7-9 AM & 5-7 PM) and non-rush hour are different.')
    st.write(':red[Alternate Hypothesis]: No significant difference in demand for bike rentals during rush hour (7-9 AM & 5-7 PM) compared to demand in non-rush hour.')
    st.text("Test Type : Two-sample T-test")
    st.text('alpha = 0.05')
    t_stat, p_val = ttest_ind(rush_hour, non_rush_hour, equal_var=False)
    st.write('Test Statistic:', t_stat)
    st.write('p-value:', p_val)
    st.write("-----")
    if p_val < 0.05:
        st.write("Since p-value is less than 0.05, we reject the null hypothesis.")
        st.write("i.e., Rented Bike Demand during rush hour (7-9 AM & 5-7 PM) and non-rush hour are different.")
    else:
        st.write("Since p-value is greater than 0.05, we fail to reject the null hypothesis.")
        st.write("i.e., There is no significant difference in demand for bike rentals during rush hour compared to non-rush hour.")

else:
    st.write("-------")
    st.write(':red[Null Hypothesis]: Rented Bike Demand is different in different seasons with highest in summer and lowest in winter.')
    st.write(':red[Alternate Hypothesis]: No significant difference in demand for bike rentals in different seasons with highest in summer and lowest in winter.')
    st.text("Test Type : One-way ANOVA Test")
    st.text('alpha = 0.05')
    f_stat, p_value = f_oneway(df.loc[df['Seasons'] == 'Spring', 'Rented Bike Count'],
                                df.loc[df['Seasons'] == 'Summer', 'Rented Bike Count'],
                                df.loc[df['Seasons'] == 'Autumn', 'Rented Bike Count'],
                                df.loc[df['Seasons'] == 'Winter', 'Rented Bike Count'])
    st.write('Test Statistic:', f_stat)
    st.write('p-value:', p_value)
    st.write("-----")
    if p_value < 0.05:
        st.write("Since p-value is less than 0.05, we reject the null hypothesis.")
        st.write("i.e., Rented Bike Demand is different in different seasons with highest in summer and lowest in winter.")
    else:
        st.write("Since p-value is greater than 0.05, we fail to reject the null hypothesis.")
        st.write("i.e., There is no significant difference in demand for bike rentals in different seasons with highest in summer and lowest in winter.")


