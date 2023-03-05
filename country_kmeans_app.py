import streamlit as st
from keras.models import load_model
import numpy as np
import pandas as pd
import pickle
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

## set page configuration
st.set_page_config(page_title='Countries Clustering', layout='centered')
  

with open(r'C:\Users\cmedj\OneDrive\Documents\Raf Course\week7\kmeans_model.pkl','rb') as file:
     model_kmeans=pickle.load(file)

df_app=pd.read_csv("country-data-copy.csv")



## add page title and content
st.title('Unsupervised Learning on Country Data using KMeans Clustering')
col1, col2, col3=st.columns(3)
@st.cache_data

##numeric_var=['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp']
def info_callback(d1,d2,d3,d4,d5,d6,d7,d8,d9):    
            with open('dt.csv', 'w+') as f:    
                myfile=csv.writer(f)
                myfile.writerow(['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp'])
                myfile.writerow([child_mort, exports, health, imports, income,inflation, life_expec, total_fer, gdpp])
                     
                     
                
with st.form(key="my_form",clear_on_submit=True):
    
        st.write("Enter the required data:")
        
        child_mort=st.number_input(label="Please enter child mortality (death of children under 5 years of age per 1000 live births):", step=1.,format="%.2f" )
        exports=st.number_input(label="Please leave exports as 0", step=1.,format="%.2f" )
        health=st.number_input(label="Please leave health as 0", step=1.,format="%.2f" )
        imports=st.number_input(label="Please leave imports as 0:", step=1.,format="%.2f" )
        income=st.number_input(label="Please leave income as 0:", step=1.,format="%.2f" )
        inflation=st.number_input(label="leave inflation as 0:", step=1.,format="%.2f" )
        life_expec=st.number_input(label="Please leave life_expec as 0:", step=1.,format="%.2f" )
        total_fer=st.number_input(label="Please leave total_fer as 0:", step=1.,format="%.2f" )
        gdpp=st.number_input(label="Please enter the GDP per capita: ",step=1.,format="%.2f" )
        cntr=st.text_input('Please enter the name of the country or a label:')
        submitted=st.form_submit_button("Submit")
        if submitted:
            st.write("gdpp", gdpp, "child_mort", child_mort) 
            info_callback(child_mort, exports, health, imports, income,inflation, life_expec, total_fer, gdpp)

row=pd.read_csv("dt.csv")
a=len(df_app)
df_app=df_app.append(row, ignore_index=True)
st.info(" #### Show contents of the CSV file :point_down:")
sample_test=st.dataframe(pd.read_csv("dt.csv"))

#sample=pd.read_csv("dt.csv")
sample=df_app.loc[:,['gdpp','child_mort']]

## Standardisation of the columns 'gdpp' and 'child_mort'
sample['gdpp'] = (sample['gdpp'] - sample['gdpp'].min()) / (sample['gdpp'].max() - sample['gdpp'].min())
sample['child_mort'] = (sample['child_mort'] - sample['child_mort'].min()) / (sample['child_mort'].max() - sample['child_mort'].min())

## make the prediction
if st.button('Predict'):
      prediction=model_kmeans.fit_predict(sample)
      b=len(prediction) - 1
      clusters=pd.DataFrame(prediction)
      clusters = clusters.rename(columns={0: 'cluster'})
      st.write('Your entry is in cluster number: ', clusters. loc[b,'cluster'])
      
        
## Visualisation

df_app=df_app.join(clusters['cluster'])
df_app.loc[b, 'country']= 'cntr'
st.write('Please see the dataset and the scatter plot below:')
st.write(df_app)

fig = px.scatter(
    df_app,
    x="gdpp",
    y="child_mort",
    color="cluster",
    hover_name="country",
    log_x=False,
    size_max=6)


st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        