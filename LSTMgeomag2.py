# based on http://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# load and plot dataset
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import Series
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy as np

import pickle
import sys

# load dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

'''
#Figure out how scaler works.

scaler = MinMaxScaler(feature_range=(-1,1))
scaler = scaler.fit(train[:,0,0].reshape(-1,1))
train_scaled = scaler.transform(train[:,0,0].reshape(-1,1))[:,0]
train_unscaled = scaler.inverse_transform(train_scaled.reshape(-1,1))[:,0]

print train[:10,0,0]
print train_scaled[:10]
print train_unscaled[:10]

sys.exit()
'''

# scale train and test data to [-1, 1]. 
def scale(train, test):
    # create list of scalers
    scalerlist = list()
    train_scaled = np.zeros(train.shape)
    test_scaled = np.zeros(test.shape)
    for i in range(train.shape[2]):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train[:,0,i].reshape(-1,1))
        train_scaled[:,:,i] = scaler.transform(train[:,:,i])
        test_scaled[:,:,i] = scaler.transform(test[:,:,i])
        #scaler.attr = i
        scalerlist.append(scaler)
    return scalerlist, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scalerlist, value):
    inverted = np.zeros(value.shape)
    for i in range(value.shape[2]):
        inverted[:,:,i] = scalerlist[i].inverse_transform(value[:,:,i])
    return inverted

# LSTM is a type of RNN. Does not need window lagged observation (stateful=True)
# reset_states() clears state of LSTM. Not used in present one-pass learning.
# Takes input in matrix with dimensions [samples, time steps, features]

def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    # create and compile the network
    model = Sequential() 
    # on short trainings, a huge model doesn't seem to do much good.
    model.add(LSTM(neurons*X.shape[2], batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    #model.add(LSTM(neurons*X.shape[2], batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=False, return_sequences=True))
    #model.add(LSTM(2*neurons*X.shape[2], stateful=True, return_sequences=True))
    #model.add(LSTM(neurons*X.shape[2], stateful=False))
    model.add(Dense(X.shape[2]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        #model.reset_states()
    return model

# make a one-step forecast ##Upgrade this to 60 step forecasts
def forecast_lstm(model, batch_size, X):
    X = X.reshape(X.shape[0], 1, X.shape[1])
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0]

# import data
DIR = '/home/handmer/Documents/FDL/'
timeseries=pickle.load(open(DIR + 'H_2016_minutes_week.pkl', 'rb'))
series=np.array(timeseries)

# transform data to be stationary
diff_values = np.nan_to_num(np.diff(series))

# transform data to be supervised learning
supervised_values = np.zeros((diff_values.shape[1]+1,2,diff_values.shape[0]))
supervised_values[1:,0] = diff_values.T
supervised_values[:-1,1] = diff_values.T

# split data into train and test-sets

train, test = supervised_values[:-int(len(supervised_values)/100)], supervised_values[-int(len(supervised_values)/100):]

# transform the scale of the data
scalerlist, train_scaled, test_scaled = scale(train, test)

# fit the model. Hyperparams: 1 pass, 2*56 neuron "hidden layer"
lstm_model = fit_lstm(train_scaled, 1, 1, 2)

# forecast (nearly) the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(train_scaled.shape[0], 1, train_scaled.shape[2])
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the remaining test data
predictions = np.zeros((test_scaled.shape[0],test_scaled.shape[2]))
nrmse=np.zeros(test.shape[0])
np.set_printoptions(precision=2) 
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    #yhat = test_scaled[i-1, 0]
    predictions[i] = yhat
    nrmse[i] = sqrt(np.abs(np.mean(y**2-yhat**2))/(np.mean(y**2)))

print np.mean(nrmse)

train_scaled_x = np.arange(1,1+train_scaled.shape[0],1)
test_scaled_x = np.arange(1+train_scaled.shape[0],1+train_scaled.shape[0]+test_scaled.shape[0],1)

for i in range(predictions.shape[1]):
    print (i, sqrt(np.abs(np.mean(test_scaled[:,1,i]**2-predictions[:,i]**2)))/sqrt(np.mean(test_scaled[:,1,i]**2)))

out=13

#pyplot.plot(series[out])
#pyplot.show()

#pyplot.plot(train_scaled_x,train_scaled[:,0,out])
#pyplot.plot(test_scaled_x,test_scaled[:,0,out])
#pyplot.plot(test_scaled_x,predictions[:,out])
#pyplot.show()

pyplot.plot(test_scaled_x,test_scaled[:,0,out])
pyplot.plot(test_scaled_x,predictions[:,out])
pyplot.xaxis('time (minutes)')
pyplot.yaxis('B (nT)')
pyplot.show()

#pyplot.plot(nrmse)
#pyplot.show()

# track loss vs records
# array output/prediction
# null hypothesis

# check noise in data. Take diff with stencil?
# improve data saving technique
