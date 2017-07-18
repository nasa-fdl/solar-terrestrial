# THIS HAS LOTS OF BUGS
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

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# create a differenced series ##Deprecated
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i-interval]
        diff.append(value)
    return Series(diff)

# invert differenced value ##Deprecated
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]. Do it individually for each datastream. Derp.
def scale(train, test):
    # create list of scalers
    scalerlist = list()
    train_scaled = train
    test_scaled = test
    for i in range(train.shape[2]):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train[:,:,i])
        train_scaled[:,:,i] = scaler.transform(train[:,:,i])
        test_scaled[:,:,i] = scaler.transform(test[:,:,i])
        scaler.attr = i
        scalerlist.append(scaler)
        
    # fit scaler
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #scaler = scaler.fit(train.flatten().reshape(-1, 1))
    # transform train
    # train = train.reshape(train.shape[0], train.shape[1])
    #train_scaled = scaler.transform(train.reshape(train.shape[0],train.shape[1]*train.shape[2]))
    # transform test
    #test = test.reshape(test.shape[0], test.shape[1])
    #test_scaled = scaler.transform(test.reshape(test.shape[0],test.shape[1]*test.shape[2]))
    #return scaler, train_scaled.reshape(train.shape), test_scaled.reshape(test.shape)
    return scalerlist, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    #new_row = [x for x in X] + [value]
    #array = np.array(new_row)
    #array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(value.reshape(1,value.shape[0]))
    return inverted[0]

# LSTM is a type of RNN. Does not need window lagged observation (stateful=True)
# reset_states() clears state of LSTM.
# Takes input in matrix with dimensions [samples, time steps, features]

def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    #X = X.reshape(X.shape[0], 1, X.shape[1]) #already shaped right!
    # create and compile the network
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(X.shape[2]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        #model.reset_states()
    return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(X.shape[0], 1, X.shape[1])
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0]

# function to partition np.arrays
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


# import data
DIR = '/home/handmer/Documents/FDL/'
timeseries=pickle.load(open(DIR + 'H_2016_minutes_week.pkl', 'rb'))
series=np.array(timeseries)

# transform data to be stationary
#raw_values = series.values
#diff_values = difference(paw_values, 1)
diff_values = np.nan_to_num(np.diff(series))

# cut into blocks -- SAVE THIS FOR AFTER
#diff_values_split = blockshaped(diff_values[:-59],int(np.floor(diff_values.shape[1]/60.)),1)

# transform data to be supervised learning
supervised_values = np.zeros((diff_values.shape[1]+1,2,diff_values.shape[0]))
supervised_values[1:,0] = diff_values.T
supervised_values[:-1,1] = diff_values.T

# split data into train and test-sets

#train, test = supervised_values[:-int(len(supervised_values)/3)], supervised_values[-int(len(supervised_values)/3):]
train, test = supervised_values[:9800], supervised_values[9800:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model 1500,1, 3000, 4?
lstm_model = fit_lstm(train_scaled, 1, 1, 2)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(train_scaled.shape[0], 1, train_scaled.shape[2])
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = np.zeros((test_scaled.shape[0],test_scaled.shape[2]))
nrmse=np.zeros(test.shape[0])
np.set_printoptions(precision=2) 
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    #print 'y, yhat'
    #print y
    #print yhat
    #print sqrt(np.abs(np.mean(y**2-yhat**2)))
    predictions[i] = yhat
    nrmse[i] = sqrt(np.abs(np.mean(y**2-yhat**2))/(np.mean(y**2)))
    #print (i, nrmse[i])
    # invert scaling
    #yhat = invert_scale(scaler, X, yhat)
    # invert differencing, store forecast
    #predictions[i] = yhat
    #expected = test[i,1]
    #print('Test=%d, Predicted=\n%.2f\n, Expected=\n%.2f' % (i, predictions[i], expected))
    #print('Test=%d, relative rmse=%.3f' % (i,sqrt(mean_squared_error(test[i,1],predictions[i]))/sqrt(np.mean(test[i,1]**2))))
    #print test[i,1]
    #print predictions[i]

print np.mean(nrmse)

# report performance
#rmse = sqrt(mean_squared_error(test[:,1], predictions))
#print('Test RMSE: %.3f' % rmse)
#error_scores.append(rmse)

# line plot of observed vs predicted
#pyplot.plot(test[:,1])
#pyplot.plot(predictions)
#pyplot.show()

#pyplot.plot(np.diff(series[0]))

train_scaled_x = np.arange(1,1+train_scaled.shape[0],1)
test_scaled_x = np.arange(1+train_scaled.shape[0],1+train_scaled.shape[0]+test_scaled.shape[0],1)

pyplot.plot(series[0])
pyplot.show()

pyplot.plot(train_scaled_x,train_scaled[:,0,0])
pyplot.plot(test_scaled_x,test_scaled[:,0,0])
pyplot.plot(test_scaled_x,predictions[:,0])
pyplot.show()

pyplot.plot(test_scaled_x,test_scaled[:,0,0])
pyplot.plot(test_scaled_x,predictions[:,0])
pyplot.show()

pyplot.plot(nrmse)
pyplot.show()

'''
# summarize results
results = DataFrame()
results['rmse'] = error_scores
print(results.describe())
results.boxplot()
pyplot.show()
'''
