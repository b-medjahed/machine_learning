import pandas as pd
import streamlit as st
import keras
from PIL import Image
from keras_preprocessing.sequence import pad_sequences
#from keras import pad_sequences
import csv
## loading ANN model
model=keras.models.load_model(r'C:\Users\cmedj\OneDrive\Documents\Raf Course\week7\model_ANN')

## load a copy of the dataset
df=pd.read_csv('heart_failure_clinical_records_dataset.csv')

## set page configuration
st.set_page_config(page_title='Heart Failure Classifier', layout='centered')

## add page title and content
st.title('Heart Failure Classifier using Artificial Neural Network')
st.write('Please scroll down and enter your data:')

## add image
image=Image.open(r'C:\Users\cmedj\OneDrive\Documents\Raf Course\week7\heart.png')
st.image(image, use_column_width=True)


    
    

def info_callback(d1, d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12):    
            with open('data.csv', 'w+') as file:    
                myfile=csv.writer(file)
                myfile.writerow(["age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction",
                     "high_blood_pressure","platelets","serum_creatinine","serum_sodium",
                     "sex", "smoking", "time"])
                myfile.writerow([age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,
                     high_blood_pressure,platelets,serum_creatinine,serum_sodium,
                     sex, smoking, time])    
        

with st.form(key="my_form",clear_on_submit=True):
    
        st.write("Enter the required data:")
        age=st.number_input(label="Please enter the patient's age: ",step=1.,format="%.2f" )
        anaemia=st.number_input(label="Anaemia? Please enter 1 for yes/ 0 for No: ")
        creatinine_phosphokinase=st.number_input(label="Please enter creatinine_phosphokinase: ")
        diabetes=st.number_input(label="Diabetes? Please enter 1 for yes/ 0 for No: ")
        ejection_fraction=st.number_input(label="Please enter ejection_fraction: ")
        high_blood_pressure=st.number_input(label="high_blood_pressure? Please enter 1 for yes/ 0 for No: ")
        platelets=st.number_input(label="Please enter platelets: ")
        serum_creatinine=st.number_input(label="Please enter serum_creatinine ",step=1.,format="%.2f")
        serum_sodium=st.number_input(label="\Please enter serum_sodium ")
        sex=st.number_input(label="Sex? Please enter 1 for male/ 0 for female: ")
        smoking=st.number_input(label="Smoking? Please enter 1 for yes/ 0 for No: ")
        time=st.number_input(label="Please enter the patient's follow up time: " )

        submitted=st.form_submit_button("Submit")
        if submitted:
            st.write("age", age, "anaemia", anaemia, "creatinine_phosphokinase", creatinine_phosphokinase,
                     "diabetes", diabetes, "ejection_fraction", ejection_fraction,
                     "high_blood_pressure", high_blood_pressure, "platelets", platelets,
                     "serum_creatinine", serum_creatinine,"serum_sodium", serum_sodium,
                     "sex", sex, "smoking", smoking, "time", time)
            info_callback(age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,
                     high_blood_pressure,platelets,serum_creatinine,serum_sodium,
                     sex, smoking, time)

        
    
    
    
    

#st.info(" #### Show contents of the CSV file :point_down:")
sample_test=st.dataframe(pd.read_csv("data.csv"))

sample=pd.read_csv("data.csv")



## make the prediction
if st.button('Predict'):
    prediction=model.predict(sample)
    

## print the result
    #set the threshold of 0.5:
    if prediction>0.5:
             st.write('Mortality')
    else:
             st.write('No Mortality')
