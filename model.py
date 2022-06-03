import pandas as pd
import pickle

#we are loading the model using pickle
model = pickle.load(open('LightGBM_model.xml', 'rb'))

def model_predictions(df):
    predictions=model.predict(df)
    df['predictions']=predictions
    return(df)