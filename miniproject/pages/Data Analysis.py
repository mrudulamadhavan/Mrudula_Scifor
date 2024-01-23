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
st.subheader('        Regression Plots :  Rented Bike Count vs Independent Variables')
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
st.write('')


fig_wind_speed = px.scatter(df, x='Wind speed (m/s)', y='Rented Bike Count',
                         labels={'Wind speed (m/s)': 'Wind Speed (m/s)', 'Rented Bike Count': 'Avg Rented Bike Count'},
                         title='Average Rented Bike Count by Wind Speed')
st.plotly_chart(fig_wind_speed,use_container_width=True)
fig_humidity = px.scatter(df, x='Humidity(%)', y='Rented Bike Count',
                       labels={'Humidity(%)': 'Humidity (%)', 'Rented Bike Count': 'Avg Rented Bike Count'},
                       title='Average Rented Bike Count by Humidity')
st.plotly_chart(fig_humidity,use_container_width=True)
fig_visibility = px.scatter(df, x='Visibility (10m)', y='Rented Bike Count',
                       labels={'Visibility (10m)': 'Visibility (10m)', 'Rented Bike Count': 'Avg Rented Bike Count'},
                       title='Average Rented Bike Count by Visibility')
st.plotly_chart(fig_visibility,use_container_width=True)

st.write('------------------------------------------------------------------------------------')


# Function to distribute hour
def distribute_hour(h):
    if 17 <= h <= 22:
        return 'Evening'
    elif 7 <= h <= 10:
        return 'Morning'
    elif 11 <= h <= 16:
        return 'Noon'
    else:
        return 'Night'

# Apply the hour function
df['Hour'] = df['Hour'].apply(distribute_hour)

target_variable = 'Rented Bike Count'

hourtype = st.selectbox("Select the Parts of Day:", df['Hour'].unique())
col1, col2 = st.columns(2)
fig_1 = px.bar(df,x='Hour',y=target_variable , title='Distribution of Bike Rentals Demand in various Parts of Day')
fig_1.update_layout(bargap=0.2)
col1.plotly_chart(fig_1, use_container_width=True)
fig_2 = px.box(df[df['Hour'] == hourtype], y=target_variable)
col2.plotly_chart(fig_2, use_container_width=True)        
st.write("-------------------------------------------------------------------------------------------")   

status = st.radio("Click to know more : ",('Correlation Heatmap','Regression Plot','Hypothesis Testing Results'))
if (status == 'Correlation Heatmap'):
    # Display the correlation heatmap using Seaborn
    corr = df.corr()
    mask = np.array(corr)
    mask[np.tril_indices_from(mask)] = False
    f, ax = plt.subplots(figsize=(20, 6))
    heatmap = sns.heatmap(corr, annot = True, fmt='.3f',mask=mask, cmap='mako',cbar=True )
    heatmap.set_title('Correlation Heatmap', pad=15)
    st.pyplot(f)

elif (status == 'Regression Plot'):
    # # Selecting numeric features for regression plots
    numeric_features = ['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
                        'Visibility (10m)', 'Dew point temperature(°C)',
                        'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)','Rented Bike Count']
    dependent_variable = 'Rented Bike Count'

    # Regression plots for independent variables
    st.subheader('Regression Plot')    
    n = 1
    plt.figure(figsize=(12, 15))
    for i in numeric_features:
        if i == 'Rented Bike Count':
            pass
        else:
            plt.subplot(4, 2, n)
            n += 1
            sns.regplot(x= df[i], y= df['Rented Bike Count'], line_kws={"color": "red"},scatter_kws={'çolor':'violet'})
            plt.title(f'Rented Bike Count vs {i}')
    # Display the plots in the Streamlit app
    st.pyplot()

else:
    st.subheader('Hypothesis Testing Results')
    st.write('Hypothesis testing is a statistical method used to make inferences or draw conclusions about a population based on a sample of data. The process involves formulating a hypothesis about the population parameter, collecting and analyzing data, and then using statistical tests to determine whether there is enough evidence to reject the null hypothesis in favor of an alternative hypothesis.')
    st.write('Based on the exploratory analysis conducted using charts, we formulated three hypothetical statements about the dataset and subsequently conducted hypothesis testing through code and statistical methods to draw conclusive results regarding these statements.')
    status = st.radio("Hypothetical Statements : ", ('1: Rented Bike Demand in hot weather is higher compared to demand in cold weather.','2. Rented Bike Demand during rush hour (7-9 AM & 5-7 PM) and non-rush hour are different.','3. Rented Bike Demand is different in different seasons with highest in summer and lowest in winter.'))
    
    if (status == '1: Rented Bike Demand in hot weather is higher compared to demand in cold weather.'):
        # Split the data into the 'hot' and 'cold' temperature groups
        hot_temps = df[df['Temperature(°C)'] >= 20]['Rented Bike Count']
        cold_temps = df[df['Temperature(°C)'] < 20]['Rented Bike Count']
        st.write('Null Hypothesis : Rented Bike Demand in hot weather is higher compared to demand in cold weather.')
        st.write('Alternate Hypothesis : No significant difference in demand for Bike rentals in hot weather compared to demand in cold weather.')
        st.text("Two-sample T-test")
        st.text('alpha = 0.05')        
        # Perform the t-test
        t_stat, p_val = ttest_ind(hot_temps, cold_temps, equal_var=False)
        st.text('Test Statistic :',t_stat)
        st.text('p-value :',p_val)
        if p_val < 0.05:
            st.text("Since p-value is less than 0.05, we reject null hypothesis.")
            st.text("ie, Rented Bike Demand in hot weather is higher compared to demand in cold weather.")
        else:
            st.text("Since p-value is greater than 0.05, we fail to reject null hypothesis.")
            st.text("ie, There is no significant difference in demand for bike rentals in hot weather compared to demand in cold weather.")
    
    elif (status ==  '2. Rented Bike Demand during rush hour (7-9 AM & 5-7 PM) and non-rush hour are different.'): 
        # Create subsets of the data based on hour
        rush_hour = df[(df['Hour'] >= 7) & (df['Hour'] <= 9) | (df['Hour'] >= 17) & (df['Hour'] <= 19)]['Rented Bike Count']
        non_rush_hour = df[~((df['Hour'] >= 7) & (df['Hour'] <= 9) | (df['Hour'] >= 17) & (df['Hour'] <= 19))]['Rented Bike Count']  
        st.write('Null Hypothesis : Rented Bike Demand during rush hour (7-9 AM & 5-7 PM) and non-rush hour are different.')
        st.write('Alternate Hypothesis : No significant difference in demand for bike rentals during rush hour (7-9 AM & 5-7 PM) compared to demand in non-rush hour.')
        st.text("Two-sample T-test")
        st.text('alpha = 0.05')
        
        # Perform the t-test
        t_stat, p_val = ttest_ind(rush_hour, non_rush_hour, equal_var=False)
        st.text('Test Statistic :',t_stat)
        st.text('p-value :',p_val)
        if p_val < 0.05:
            st.text("Since p-value is less than 0.05, we reject null hypothesis.")
            st.text("ie, Rented Bike Demand during rush hour (7-9 AM & 5-7 PM) and non-rush hour are different.")
        else:
            st.text("Since p-value is greater than 0.05, we fail to reject null hypothesis.")
            st.text("ie, There is no significant difference in demand for bike rentals during rush hour (7-9 AM & 5-7 PM) compared to demand in non-rush hour")

    else:

        st.write('Null Hypothesis :  Rented Bike Demand is different in different seasons with highest in summer and lowest in winter.')
        st.write('Alternate Hypothesis : No significant difference in demand for bike rentals in different seasons with highest in summer and lowest in winter.')
        st.test('One-way ANOVA test')
        st.text('alpha = 0.05')
        # Conduct the ANOVA test
        f_stat, p_value = f_oneway(df.loc[df['Seasons']=='Spring', 'Rented Bike Count'],
                                    df.loc[df['Seasons']=='Summer', 'Rented Bike Count'],
                                    df.loc[df['Seasons']=='Autumn', 'Rented Bike Count'],
                                    df.loc[df['Seasons']=='Winter', 'Rented Bike Count'])
        st.text('Test Statistic :',f_stat)
        st.text('p-value :',p_value)
        if p_value < 0.05:
            st.text("Since p-value is less than 0.05, we reject null hypothesis.")
            st.text("ie, Rented Bike Demand is different in different seasons with highest in summer and lowest in winter.")
        else:
            st.text("Since p-value is greater than 0.05, we fail to reject null hypothesis.")
            st.text("ie, There is no significant difference in demand for bike rentals in different seasons with highest in summer and lowest in winter.")





