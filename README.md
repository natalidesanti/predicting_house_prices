# Predicting house prices

This repo gives a comprehensive guide to predict house prices

## Features

With this package you will be able to:

* Using `Property-Friends-basic-model.ipynb`, analyse train and test set, data explore, data pre-process, grid search explore a Gradient Boost Regressor, train a Gradient Boost Regressor, load and evaluate the model in the test set

* Run a API responsible to run the trained model and evaluate it on the test set using a docker

## Requisites

All the code is done in `Python`, `txt` and `Markdown`. To run the pipeline (`Jupyter Notebook`) and model deploy it on a `Docker` (notice that they are on `requirements` file in this case), you will need the following `Python` libraries:
* pandas
* numpy
* matplotlib
* scikit-learn
* seaborn
* pickle5

## Usage

In this repo you will find the 3 main components:
* `Property-Friends-basic-model.ipynb`: which can be used as a notebook to get the model to be deployed. Use this as a typical notebook
* `model_deploy.py`: contains the instructions necessary to run the model (`model/pima.pickle.dat`)
* `Dockerfile`: contains docker instructions to build the image

To build the image, go to the main project directory and use:
'''$ docker build -t predicting_house_prices:latest .'''

To run the image:
'''docker run predicting_house_prices:latest'''

## To do

* Expand the range and hyperparameters in the GridSearch
* Include KFold/CV splitter
* Improve the logs related to the whole pipeline
* Include the security system for the API
