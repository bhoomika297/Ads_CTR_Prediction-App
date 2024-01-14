import streamlit as st
import numpy as np

import pickle



#load the RandomForest model with the specified protocol
model = pickle.load(open('ads_CTR.pkl','rb'))

# Function to make predictions
def RDF_prediction(features):
    #replace 'model' with your actual learning model
    prediction = model.predict(features)
    return prediction


# streamlit app

def main():
    st.title("Ads click Through Rate Prediction")

    # Collecting user input
    daily_time_spent = st.slider("Daily Time Spent on Site",min_value=0.0,max_value=24.0,value=12.0,step=0.1)
    age = st.slider("Age", min_value=12,max_value=100,value=30)
    area_income = st.number_input("Area Income",min_value=1000.0,max_value=500000.0,value=1000.0,step=1000.0)
    daily_internet_usage = st.slider("Daily Inernet Usage",min_value=0.0,max_value=24.0,value=6.0,step=0.1)
    gender = st.radio("Gender", options=["Male","Female"])

    #mapping gender to numertical values
    gender_mapping = {'Male':1,'Female':0}
    gender_numeric = gender_mapping.get(gender,0)

    # Feature for prediction
    features = np.array([[daily_time_spent,age,area_income,daily_internet_usage,gender_numeric]])


    if st.button("Predict"):
        #Making Prediction
        prediction = RDF_prediction(features)

        #Displaying the prediction
        st.success(f"Prediction : User will {'click' if prediction == 1 else 'not click' } on the ad.")


# Run the streamlit app
if __name__ == "__main__":
    main()