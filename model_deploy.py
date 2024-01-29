'''
This script is respobsible for deploy the model
'''

#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

#Data loading
def load_data(path):
    '''
    This function returns the data set

    -----------
    Parameters
    -----------
    
    path : string
        Contains the directory and name of the file

    '''
    
    data = pd.read_csv(path)
    
    return data

test = load_data('test.csv')

#Data pre-processing
def data_prep(df):
    '''
    This function is responsible to:
        1) Sample the data set
        2) Remove nan values
        3) Tranform categorical into numerical information 
        4) Transform skewed and wide range variable distributions

    -----------
    Parameters
    -----------
    
    df : pandas dataframe
        Contains the dataframe in question

    '''
    #1) Sampling the data set
    df = df.sample(frac = 1)

    #2) Removing nan values
    df = df.dropna()

    #3) Transforming categorical to numerical information
    LE = LabelEncoder()

    df['type'] = LE.fit_transform(df['type'])
    df['sector'] = LE.fit_transform(df['sector'])

    #4) Tranforming skewed and wide range variable distributions
    df['net_usable_area'] = np.log10(df['net_usable_area'] + 10**(-3))
    df['net_area'] = np.log10(df['net_area'] + 10**(-3))
    df['price'] = np.log10(df['price'] + 10**(-3))

    return df

test = data_prep(test)

def selecting_data(df):
    '''
    This function split the corresponding data set into:
        X: input variables
        y: output variables

    -----------
    Parameters
    -----------
    
    df : pandas dataframe
        Contains the dataframe in question

    '''

    X = np.array([df['type'], df['sector'], df['net_usable_area'], 
                  df['net_area'], df['n_rooms'], df['n_bathroom'],
                  df['latitude'], df['longitude'] 
                  ]).T
    
    y = np.array([df['price']]).T

    return X, y

X_test, y_test = selecting_data(test)

#Loading and evaluating the model on the test set

def load_and_evaluate(X, y):
    '''
    This function load the trained model and provides some evaluation metrics

    -----------
    Parameters
    -----------

    X : numpy array [# props, 8]
        Array with the number of properties X property features 

    y : numpy array [# props, 1]
        Array with the price of the properties 

    '''

    #Load model from file
    loaded_model = pickle.load(open("model/pima.pickle.dat", "rb"))

    #Predicting the house prices for the selected set
    pred = loaded_model.predict(X)

    #Evaluating the model
    RMSE = np.sqrt(mean_squared_error(pred, y))
    MSE = mean_squared_error(pred, y)
    MAPE = mean_absolute_percentage_error(pred, y)

    #Printing metrics
    print("Model metrics")
    print("RMSE: ", np.sqrt(mean_squared_error(pred, y)))
    print("MAPE: ", mean_absolute_percentage_error(pred, y))
    print("MAE : ", mean_absolute_error(pred, y))

load_and_evaluate(X_test, y_test)





