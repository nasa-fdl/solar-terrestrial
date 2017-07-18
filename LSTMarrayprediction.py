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

# Read local config file
config = ConfigParser.RawConfigParser()
config.read('myconfig.cfg')

DIR = config.get('LSTMCFG','DIR')
FILENAME = config.get('LSTMCFG','FILENAME')
OUTPUT = config.get('LSTMCFG','OUTPUT')
TIMES = map(int, config.get('LSTMCFG','TIMES').split(','))

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
    X, y = train[:, 0:1], train.reshape(train.shape[0],train.shape[1]*train.shape[2])
    # create and compile the network
    model = Sequential() 
    # on short trainings, a huge model doesn't seem to do much good.
    model.add(LSTM(neurons*y.shape[1], batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    #model.add(LSTM(neurons*X.shape[2], batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=False, return_sequences=True))
    #model.add(LSTM(2*neurons*X.shape[2], stateful=True, return_sequences=True))
    #model.add(LSTM(neurons*X.shape[2], stateful=False))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        #model.reset_states()
    return model

# make a one-step forecast ##Upgrade this to 60 step forecasts
def forecast_lstm(model, batch_size, X):
    X = X.reshape(X.shape[0], 1, X.shape[1])
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0].reshape((yhat.shape[1]/X.shape[2],X.shape[2]))

# import data
timeseries=pickle.load(open(DIR + FILENAME, 'rb'))
series=np.array(timeseries)

# transform data to be stationary (take first derivative to remove slow drift)
# This won't work for a wide class of functions
diff_values = np.nan_to_num(np.diff(series))

# transform data to be supervised learning
predict_times = TIMES
max_predict=max(predict_times)
supervised_values = np.zeros((diff_values.shape[1]+max_predict+1,len(predict_times),diff_values.shape[0]))
for i in range(len(predict_times)):
    supervised_values[predict_times[i]:-max_predict+predict_times[i]-1,i] = diff_values.T

# split data into train and test-sets
#train, test = supervised_values[:100], supervised_values[100:110]
train, test = supervised_values[:-int(len(supervised_values)/100)], supervised_values[-int(len(supervised_values)/100):]

# transform the scale of the data
scalerlist, train_scaled, test_scaled = scale(train, test)

# fit the model. Hyperparams: 1 pass, 2*56 neuron "hidden layer"
lstm_model = fit_lstm(train_scaled, 1, 1, 2)

# forecast (nearly) the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(train_scaled.shape[0], 1, train_scaled.shape[2])
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the remaining test data
bestresult = np.zeros(test_scaled.shape)
predictions = np.zeros(test_scaled.shape)
predictionsnonML = np.zeros(test_scaled.shape)
nrmse=np.zeros(test.shape[0])
nrmsenonML=np.zeros(test.shape[0])
np.set_printoptions(precision=2) 
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:1], test_scaled[i]
    bestresult[i] = y
    yhat = forecast_lstm(lstm_model, 1, test_scaled[i,0:1])
    yhatnonML = 2*test_scaled[i-1] - test_scaled[i-2] #two point
    predictions[i] = yhat
    predictionsnonML[i] = yhatnonML
    #think deeply about prediction weighting
    nrmse[i] = sqrt(np.abs(np.mean((y-yhat)**2/y**2)))
    nrmsenonML[i] = sqrt(np.abs(np.mean((y-yhatnonML)**2/y**2)))
    print i, nrmse[i], nrmsenonML[i]

print np.mean(nrmse)

#Define the x coordinates for plotting
train_scaled_x = np.arange(1,1+train_scaled.shape[0],1)
test_scaled_x = np.arange(1+train_scaled.shape[0],1+train_scaled.shape[0]+test_scaled.shape[0],1)

#save stuff to files
np.savetxt(DIR+OUTPUT+'_ideal.csv', bestresult.reshape(bestresult.shape[0],bestresult.shape[1]*bestresult.shape[2]), delimiter=",")
np.savetxt(DIR+OUTPUT+'_nonML.csv', predictionsnonML.reshape(predictionsnonML.shape[0],predictionsnonML.shape[1]*predictionsnonML.shape[2]), delimiter=",")
np.savetxt(DIR+OUTPUT+'_ML.csv', predictions.reshape(predictions.shape[0],predictions.shape[1]*predictions.shape[2]), delimiter=",")

# this measures total relative error over the entire test period for each of the different prediction times and channels.
for i in range(predictions.shape[2]):
    print i, ['%.2f' % sqrt(np.abs(np.mean((bestresult[:,j,i]-predictions[:,j,i])**2/bestresult[:,j,i]**2))) for j in range(predictions.shape[1])]

out=16

# plot raw data
#pyplot.plot(series[out])
#pyplot.show()

# plot scaled data and predictions
#pyplot.plot(train_scaled_x,train_scaled[:,0,out])
#pyplot.plot(test_scaled_x,test_scaled[:,0,out])
#pyplot.plot(test_scaled_x,predictions[:,out])
#pyplot.show()


# plot predictions over test region #fix x axis
for i in range(predictions.shape[1]):
    pyplot.plot(test_scaled_x+predict_times[i],predictions[:,i,out])
pyplot.plot(test_scaled_x,test_scaled[:,0,out])
pyplot.xlabel('time (minutes)')
pyplot.ylabel('dB/dt (nT/s, scaled)')
pyplot.show()

#pyplot.plot(nrmse)
#pyplot.show()

# track loss vs records
# null hypothesis

# check noise in data. Take diff with stencil?

