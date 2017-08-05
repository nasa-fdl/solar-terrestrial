
# coding: utf-8

# In[1]:

#Heavily modified from 
#http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
#contact: Mark Cheung, cheung@lmsal.com

from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
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
if sys.version_info <= (3,5):
    import ConfigParser
else:
    import configparser as ConfigParser
import sting


# In[2]:

# Read local config file
config = ConfigParser.RawConfigParser()
config.read('myconfig.cfg')
DIR = config.get('LSTMCFG','DIR')
FILENAME = config.get('LSTMCFG','FILENAME')
batch_size = int(config.get('LSTMCFG', 'BATCH_SIZE'))
window = int(config.get('LSTMCFG', 'WINDOW'))

print("Read configuration file")


# In[3]:

#Load combined USGS and OMNI data
#execfile('merge_geomag_omni_dataframes.py')
if ('raw_data' in locals()) == False:
    exec(open("./merge_geomag_omni_dataframes.py").read())
print("Loaded geomag and OMNI data")


# In[93]:

import sting

def remove_trend(raw_values, times):
    from sklearn.linear_model import LinearRegression
    import numpy as np
    nchannels = raw_values.shape[1]
    new_values = raw_values
    trends = []
    for i in range(nchannels):
        y = raw_values[:,i].ravel()
        mask = ~np.isnan(y)
        #print(mask.shape,times.shape)
        model = LinearRegression()
        model.fit(times[mask].reshape(-1,1), y[mask])
        #print(model.coef_)
        # calculate trend
        trend = model.predict(times.reshape(-1,1))
        new_values[:,i] = y - trend
        trends.append(model)
        
    return new_values

times    = df.loc[:,'Date']
raw_data = df.loc[:,['Field mag avg, nT', 'Bx, nT (GSE, GSM)', 'By, nT (GSE,GSM)',
                     'Bz, nT (GSE)', 'By, nT (GSM)', 'Bz, nT (GSM)', 
                     #'RMS SD B scalar, nT',
                     #'RMS SD field vector, nT', 
                     #'Flow speed, km/s', 
                     'Vx, km/s, GSE',
                     'Vy, km/s, GSE', 
                     'Vz, km/s, GSE', 
                     #'Proton density, n/cc',
                     #'Temperture, K', 'Flow pressure, nPa', 
                     'Electric Field, mV/m',
                     #'Plasma beta', 'Alfven mach number', 
                     'BOU_X', 'BOU_Y', #'BOU_Z', #'BOU_F',
                     'BRW_X', 'BRW_Y', #'BRW_Z', #'BRW_F',
                     'BSL_X', 'BSL_Y', #'BSL_Z', #'BSL_F', 
                     'CMO_X', 'CMO_Y', #'CMO_Z', #'CMO_F',
                     'DED_X', 'DED_Y', #'DED_Z', #'DED_F',
                     'FRD_X', 'FRD_Y', #'FRD_Z', #'FRD_F',
                     'FRN_X', 'FRN_Y', #'FRN_Z', #'FRN_F',
                     'GUA_X', 'GUA_Y', #'GUA_Z', #'GUA_F',
                     'HON_X', 'HON_Y', #'HON_Z', #'HON_F',
                     'NEW_X', 'NEW_Y', #'NEW_Z', #'NEW_F',
                     'SHU_X', 'SHU_Y', #'SHU_Z', #'SHU_F',
                     'SIT_X', 'SIT_Y', #'SIT_Z', #'SIT_F',
                     'SJG_X', 'SJG_Y', #'SJG_Z', #'SJG_F',
                     'TUC_X', 'TUC_Y']]#, 'TUC_Z']] #, 'TUC_F']]    

units = np.zeros(raw_data.values.shape[1])
units[0:11] = 10000.0
units[11:] = 100.0

nchannels = (raw_data.values.shape)[1]

raw_values = remove_trend(raw_data.values, times)
print(raw_data.values.shape)
print(raw_values.shape)

#raw_data.values = raw_values
#window = 72
sampled_values, time = sting.raw_to_sampled(raw_values.T,window)

# Now we create rows that comprise of timeseries data with different lags, all concatenated together.
lags = np.array([0],dtype='int')

#print("sampled_values.shape",sampled_values.shape)
#print(sampled_values[0,:])


# In[94]:

raw_data


# In[95]:

# Scale data
train_scaled, test_scaled, scalers = sting.sampled_to_scaled(sampled_values,
                                                             time, nchannels, lags, 
                                                             batch_size, units=units, 
                                                             derivative=False)

print(train_scaled.shape, test_scaled.shape)
print('len(scalers)',len(scalers))

#Save scaling functions
pickle.dump(scalers, open("scalers_window{0:03d}_lags{1:03d}.pkl".format(window,np.max(lags)),"wb"))


# In[96]:

from scipy.misc import imresize
def rebin(a, *args):
    import numpy as np
    '''rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    '''
    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape)/np.asarray(args)
    evList = ['a.reshape('] +              ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] +              [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] +              ['/factor[%d]'%i for i in range(lenShape)]
    print(''.join(evList))
    return eval(''.join(evList))


# In[101]:

# Create and fit a list of models
models = []
modeldims = []
modelnames = []
def pred(x):
    return x

# Persist model 
class persist_model:
    def __init__(self):
        self.data = []

    def fit(self):
        print("fitted")
        #self.data.append(x)

    def predict(self,x):
        return pred(x)
#End of Persist model class definition
    
pmodel = persist_model()
pmodel.fit()
models.append(pmodel)
modeldims.append(3)
modelnames.append('Persist')

from sklearn.svm import SVR
from sklearn.datasets import load_linnerud
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.multioutput import MultiOutputRegressor

# to set number of jobs to the number of cores, use n_jobs=-1
model = MultiOutputRegressor(GradientBoostingRegressor(), n_jobs=1).fit(train_scaled[0:-1,:], train_scaled[1:,:])
models.append(model)
modeldims.append(2)
modelnames.append('GradientBoostingRegressor')

model = MultiOutputRegressor(AdaBoostRegressor(), n_jobs=1).fit(train_scaled[0:-1,:], train_scaled[1:,:])
models.append(model)
modeldims.append(2)
modelnames.append('AdaBoostRegressor')

model = MultiOutputRegressor(BaggingRegressor(), n_jobs=1).fit(train_scaled[0:-1,:], train_scaled[1:,:])
models.append(model)
modeldims.append(2)
modelnames.append('BaggingRegressor')

model = MultiOutputRegressor(ExtraTreesRegressor(), n_jobs=1).fit(train_scaled[0:-1,:], train_scaled[1:,:])
models.append(model)
modeldims.append(2)
modelnames.append('ExtraTreesRegressor')

model = MultiOutputRegressor(RandomForestRegressor(), n_jobs=1).fit(train_scaled[0:-1,:], train_scaled[1:,:])
models.append(model)
modeldims.append(2)
modelnames.append('RandomForestRegressor')

model = MultiOutputRegressor(SVR(), n_jobs=1).fit(train_scaled[0:-1,:], train_scaled[1:,:])
models.append(model)
modeldims.append(2)
modelnames.append('SVR')


# In[102]:

#def custom_loss(y_true, y_pred):
#    import numpy as np
#    return (y_true[18:] - y_pred[18:])**2

def custom_loss(y_true, y_pred):
    # y_pred is n-dimensional, y_true is n+1 dimensional.
    import tensorflow as tf
    import numpy as np
    #wnp = np.zeros(y_pred.shape[1]    
    Tloss = tf.losses.mean_squared_error(y_true, y_pred)
    return Tloss

# fit an LSTM network to training data
def fit_lstm_shallow(train, batch_size, nb_epoch, neurons):
    import numpy as np 
    from keras.models import Sequential
    from keras.layers import Dense, Convolution1D
    from keras.layers import LSTM
    wall = int((np.floor(train.shape[0]/batch_size)-1)*batch_size)
    X, y = train[0:wall,:], train[1:1+wall,:]
    
    #X, y = train[0:-lag, :], train[lag:,:]
    print(X.shape,y.shape)
    X = X.reshape(X.shape[0], 1, X.shape[1]) 
    print(X.shape,y.shape)
    
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))#, return_sequences=True,dropout=0.2))
    model.add(Dense(y.shape[1]))
    model.compile(loss=custom_loss, optimizer='adam')
    
    modelpred = Sequential()
    modelpred.add(LSTM(neurons, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True)) #, return_sequences=True, dropout=0.2))
    modelpred.add(Dense(y.shape[1]))
    modelpred.compile(loss=custom_loss, optimizer='adam')
        
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=(i%20==0), shuffle=False)
        model.reset_states()
    return model, modelpred

# LSTM models
lstm_model, lstm_model_pred = fit_lstm_shallow(train_scaled, 128, 401, nchannels)
models.append(lstm_model_pred)
modeldims.append(3)
modelnames.append('LSTM1')

lstm_model, lstm_model_pred = fit_lstm_shallow(train_scaled, 128, 401, 2*nchannels)
models.append(lstm_model_pred)
modeldims.append(3)
modelnames.append('LSTM2')

lstm_model, lstm_model_pred = fit_lstm_shallow(train_scaled, 128, 401, 3*nchannels)
models.append(lstm_model_pred)
modeldims.append(3)
modelnames.append('LSTM3')

lstm_model.save_weights('lstm_model_window{0:03d}_lags{1:03d}.h5'.format(window,np.max(lags)))
lstm_model_pred.load_weights('lstm_model_window{0:03d}_lags{1:03d}.h5'.format(window,np.max(lags)))
model_json = lstm_model_pred.to_json()
with open("model_window{0:03d}_lags{1:03d}.json".format(window,np.max(lags)), "w") as json_file:
    json_file.write(model_json)
    


# In[ ]:

def metrics(X,X2):
    from scipy.stats import pearsonr
    import numpy as np
    return (np.sum(np.sqrt((X - X2)**2)), (pearsonr(X.ravel(), X2.ravel()))[0])
    
def predict_ahead(X,lookahead,model,dims):
    X_pred = np.zeros([X.shape[0]-lookahead,X.shape[1]])
    for n in range(X.shape[0]-lookahead):
        yhat = X[n,:].reshape([1,1,X.shape[1]])
        
        for l in range(lookahead):
            if dims == 3:
                yhat2 = yhat.reshape([1,1,X.shape[1]])
                yhat2[0,0,0:11] = X[n,0:11] # Do not update solar wind data
            if dims == 2:
                yhat2 = yhat.reshape([1,X.shape[1]])            
                yhat2[0,0:11] = X[n,0:11] # Do not update solar wind data
            if dims == 1:
                yhat2 = yhat.reshape(X.shape[1])            
                yhat2[0:11] = X[n,0:11] # Do not update solar wind data
                    
            yhat = model.predict(yhat2)
            
        X_pred[n,:] = yhat

    return X[lookahead+range(X.shape[0]-lookahead),:], X_pred[:,:]


# How many steps ahead to predict?
lookaheads = np.arange(5,dtype='int')+1

# Metrics
mets = []
mets = np.zeros([nchannels,len(lookaheads),len(models)])

import os
from scipy.signal import correlate

# Now we use trained models to make predictions of the geomagnetic field
for l in range(len(lookaheads)):
    print('Look aheads = ', lookaheads[l])
    for m in range(len(models)):
        print(m,modelnames[m],modeldims[m])
        X, X_pred = predict_ahead(test_scaled, lookaheads[l], models[m], modeldims[m])
        inverted_predictions = X_pred.reshape([X_pred.shape[0],lags.shape[0],nchannels])
        inverted_test = X.reshape([X.shape[0],lags.shape[0],nchannels])
       
        for c in range(nchannels):
            mets[c,l,m] = (metrics(X[:,c],X_pred[:,c]))[0]
        
        for c in range(len(scalers)):
            sc = scalers[c]
            sc.inverse_transform(inverted_predictions[:,:,c])
            sc.inverse_transform(inverted_test[:,:,c])
                
            # Plot ground truth (from test data) and predicted curves
            f, (ax1, ax2) = pyplot.subplots(1, 2, sharey=False, figsize=(16, 8), dpi=80)
            pyplot.figure(c)
            a = inverted_test[:,len(lags)-1,c]
            b = inverted_predictions[:,len(lags)-1,c]
            ax1.fill(a)
            ax1.plot(b,color='orange')
            ax1.set_ylim(np.min([a.min(),b.min()]),np.max([a.max(),b.max()]))
            ax1.set_ylim(-1,1)
            ax1.set_xlim(250,400)
            corr = pearsonr(a[250:400],b[250:400])
            ax1.set_title("{0}, pearson r={1:.2f}, p={2:.3f}".format(raw_data.columns[c],corr[0],corr[1]))
            
            #Plot cross and autocorrelations for X and X_pred
            ax2.plot(correlate(a[250:400],b[250:400]),'--')
            ax2.plot(correlate(b[250:400],b[250:400]))
            ax2.plot(correlate(a[250:400],a[250:400]))
            corr = pearsonr(a[250:400],b[250:400])
            ax2.set_title("{0}, pearson r={1:.2f}, p={2:.3f}, (X,X_pred){3:.3f},(X_pred,X_pred){4:.3f},(X,X){5:.3f}".format(
                            raw_data.columns[c],corr[0],corr[1],
                          (correlate(a[250:400],b[250:400])).argmax(),
                          (correlate(b[250:400],b[250:400])).argmax(),
                          (correlate(a[250:400],a[250:400])).argmax()))
            
            directory = './window{0:03d}_lags{1:03d}_{2}'.format(window, np.max(lags),modelnames[m])
            try:
                os.stat(directory)
            except:
                os.mkdir(directory)

            f.savefig("{0}/lookahead{1:03d}_c{2:03d}.jpg".format(directory,lookaheads[l],c))            


# In[110]:

def mystr(a):
    return "{0:.03f}".format(a)

# Write out metrics
f = open("model_window{0:03d}_lags{1:03d}.metrics".format(window,np.max(lags)), "w")
for c in range(nchannels):
    for m in range(len(models)):
        f.write("Model #{0}".format(modelnames[m]))
        f.write(";, "+raw_data.columns[c]  + '  ;    '+'  ;   '.join(map(mystr,mets[c,:,m])))
        f.write("\n")
f.close()

