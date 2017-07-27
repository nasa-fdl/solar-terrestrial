# based on http://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
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
import ConfigParser
import copy

# Read local config file
config = ConfigParser.RawConfigParser()
config.read('myconfig.cfg')

DIR = config.get('LSTMCFG','DIR')
FILENAME = config.get('LSTMCFG','FILENAME')
OUTPUT = config.get('LSTMCFG','OUTPUT')
TIMES = map(int, config.get('LSTMCFG','TIMES').split(','))
BATCH = int(config.get('LSTMCFG','BATCH'))
EPOCH = int(config.get('LSTMCFG','EPOCH'))
TEST_LENGTH = int(config.get('LSTMCFG','TEST_LENGTH'))
TRAIN_LENGTH = int(config.get('LSTMCFG','TRAIN_LENGTH'))
NEURONS = int(config.get('LSTMCFG','NEURONS'))

# scale train and test data to [-1, 1]. 
def scale(train, test):
    # create list of scalers
    scalerlist = list()
    train_scaled = np.zeros(train.shape)
    test_scaled = np.zeros(test.shape)
    #for i in range(0,train.shape[2],2):#Don't operate on nan flag
    for i in range(train.shape[2]):# Not operating on nans
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train[:,0,i].reshape(-1,1))
        train_scaled[:,:,i] = scaler.transform(train[:,:,i])
        test_scaled[:,:,i] = scaler.transform(test[:,:,i])
        scalerlist.append(scaler)
    return scalerlist, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scalerlist, value):
    inverted = np.zeros(value.shape)
    #for i in range(0,value.shape[2],2): # skips nan flag
        #inverted[:,:,i] = scalerlist[i/2].inverse_transform(value[:,:,i])
    for i in range(value.shape[2]): # no nan flag
        inverted[:,:,i] = scalerlist[i].inverse_transform(value[:,:,i])
    return inverted

# LSTM is a type of RNN. Does not need window lagged observation (stateful=True)
# reset_states() clears state of LSTM. Not used in present one-pass learning.
# Takes input in matrix with dimensions [samples, time steps, features]
# Based on hyperparameter study, efficiency and accuracy is maximized with more neurons, fewer layers, more epochs, bigger batch size.
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:1], train.reshape(train.shape[0],train.shape[1]*train.shape[2])
    # create and compile the network
    model = Sequential() 
    # on short trainings, a huge model doesn't seem to do much good.
    model.add(LSTM(neurons*y.shape[1], batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))#return sequences = False by default
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
    return model

# perform forecasts
def forecast_lstm(model, batch_size, X):
    #X = np.array([X,X,X,X,X,X,X,X,X,X])
    X = X.reshape(X.shape[0], 1, X.shape[1])
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0].reshape((yhat.shape[1]/X.shape[2],X.shape[2]))

# scrub outliers
def reject_outliers(data):
    m = 10
    u = np.nanmean(data)
    s = np.nanstd(data)
    filtered = [e if (u - m*s < e < u + m*s) else float('nan') for e in data]
    return filtered

# import data
timeseries=pickle.load(open(DIR + FILENAME, 'rb'))
series=np.array(timeseries)

# New data processing. This will take series data, elide nans, add a bad data flag per channel, and scratch out wild outliers. No diff anymore.

# replace outliers (faulty data) with nans
series_no_outliers = [reject_outliers(x) for x in series]

# locate nans
nans_where = [np.argwhere(np.isnan(x))[:,0] for x in series_no_outliers]

# replace nans with mean, creates riffled array with a 0-1 nan-replacement flag to tell LSTM that the data is flat.
#series_no_nans = np.zeros((2*series.shape[0],series.shape[1]))
#series_no_nans[::2] = series
#for i in range(len(series)):
#    series_mean = np.nanmean(series[i])
#    for j in nans_where[i]:
#        series_no_nans[2*i,j] = series_mean
#        series_no_nans[2*i+1,j] = 1.

# Same eating procedure, no nans.
series_no_nans = np.zeros((series.shape[0],series.shape[1])) # no nan flag
series_no_nans = copy.copy(series)
for i in range(len(series)):
    series_mean = np.nanmean(series[i])
    for j in nans_where[i]:
        series_no_nans[i,j] = series_mean

# Take derivatives of array
series_no_nans_diff = np.diff(series_no_nans)

# transform data to be supervised learning
predict_times = TIMES
max_predict=max(predict_times)
supervised_values = np.zeros((series_no_nans_diff.shape[1]+max_predict+1,len(predict_times),series_no_nans_diff.shape[0]))
for i in range(len(predict_times)):
    supervised_values[predict_times[i]:-max_predict+predict_times[i]-1,i] = series_no_nans_diff.T

# split data into train and test-sets
#test_length = 10
#train, test = supervised_values[:100], supervised_values[100:100+test_length]
#test_length = int(len(supervised_values)/100)
test_length = TEST_LENGTH
train_length = TRAIN_LENGTH
train, test = supervised_values[:train_length], supervised_values[train_length:train_length+test_length]

# transform the scale of the data
scalerlist, train_scaled, test_scaled = scale(train, test)

# fit the model. Hyperparams: 1 pass, 2*56 neuron "hidden layer", bigger batch size to run faster
lstm_model = fit_lstm(train_scaled, BATCH, EPOCH, NEURONS)

# recover trained weights, generate new lstm with single batch input.
# matching batch size
# http://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
new_model = Sequential()
new_model.add(LSTM(NEURONS*train_scaled.shape[1]*train_scaled.shape[2], batch_input_shape=(1, 1, train_scaled.shape[2]), stateful=True))#return sequences = False by default
new_model.add(Dense(train_scaled.shape[1]*train_scaled.shape[2]))
new_model.set_weights(lstm_model.get_weights())
new_model.compile(loss='mean_squared_error',optimizer='adam')

# forecast (nearly) the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(train_scaled.shape[0], 1, train_scaled.shape[2])
new_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the remaining test data
bestresult = np.zeros(test_scaled.shape)
predictions = np.zeros(test_scaled.shape)
#predictionsnonML = np.zeros(test_scaled.shape)
nrmse=np.zeros(test.shape[0])
nrmsenonML=np.zeros(test.shape[0])
np.set_printoptions(precision=2) 
for i in range(len(test_scaled)):
    X, y = test_scaled[i, 0:1], test_scaled[i]
    bestresult[i] = y
    yhat = forecast_lstm(new_model, 1, X)#test_scaled[i,0:1])
    predictions[i] = yhat
    # weighting prediction by advanced time squared. This is roughly consistent with 2nd order extrapolation methods, but can easily be tweaked.
    nrmse[i] = np.dot(np.array(TIMES)**2,np.array([sqrt(np.abs(np.mean((y[i]-yhat[i])**2/y[i]**2))) for i in range(len(TIMES))]))
    #print i, nrmse[i]

print np.mean(nrmse)

# Invert scaling
bestresult_unscale = np.insert(invert_scale(scalerlist,bestresult),0,0.,axis=0)
predictions_unscale = np.insert(invert_scale(scalerlist,predictions),0,0.,axis=0)

# invert diff # That was painful. Let's not do that again, again.
bestresult_int = np.cumsum(bestresult_unscale,axis=0)
predictions_int = np.cumsum(predictions_unscale,axis=0)
bestresult_int += series_no_nans[:,train_length-1]
predictions_int += series_no_nans[:,train_length-1]

#Define the x coordinates for plotting
train_scaled_x = np.arange(1,1+train_scaled.shape[0],1)
test_scaled_x = np.arange(train_scaled.shape[0],1+train_scaled.shape[0]+test_scaled.shape[0],1)

#save stuff to files #update this!
np.savetxt(DIR+OUTPUT+'_ideal.csv', bestresult_unscale.reshape(bestresult_unscale.shape[0],bestresult_unscale.shape[1]*bestresult_unscale.shape[2]), delimiter=",")
np.savetxt(DIR+OUTPUT+'_ML.csv', predictions_unscale.reshape(predictions_unscale.shape[0],predictions_unscale.shape[1]*predictions_unscale.shape[2]), delimiter=",")

# this measures total relative error over the entire test period for each of the different prediction times and channels.
#for i in range(predictions.shape[2]):
#    print i, ['%.2f' % sqrt(np.abs(np.mean((bestresult[:,j,i]-predictions[:,j,i#])**2/bestresult[:,j,i]**2))) for j in range(predictions.shape[1])]

# plot raw data
#pyplot.plot(series[out])
#pyplot.show()

# plot scaled data and predictions
#pyplot.plot(train_scaled_x,train_scaled[:,0,out])
#pyplot.plot(test_scaled_x,test_scaled[:,0,out])
#pyplot.plot(test_scaled_x,predictions[:,out])
#pyplot.show()

# plot un integrated data
for out in range(2):
    pyplot.plot(test_scaled_x,bestresult_unscale[:,0,out])
    for i in range(predictions_unscale.shape[1]):
        pyplot.plot(test_scaled_x+predict_times[i],predictions_unscale[:,i,out])
    pyplot.xlabel('time (minutes)')
    pyplot.ylabel('B (nT)')
    pyplot.show()
    # plot predictions over test region 
    # There's a drift due to some random vertical offset.
    pyplot.plot(test_scaled_x,bestresult_int[:,0,out])
    #pyplot.plot(test_scaled_x,series_no_nans[out,train_length-1:train_length+test_length])
    for i in range(predictions_unscale.shape[1]):
        pyplot.plot(test_scaled_x+predict_times[i],predictions_int[:,i,out])
    pyplot.xlabel('time (minutes)')
    pyplot.ylabel('B (nT)')
    pyplot.show()

#pyplot.plot(nrmse)
#pyplot.show()

# track loss vs records

