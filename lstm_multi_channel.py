
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
#OUTPUT = config.get('LSTMCFG','OUTPUT')
#TIMES = map(int, config.get('LSTMCFG','TIMES').split(','))
#nb_epoch = int(config.get('LSTMCFG', 'NB_EPOCH'))


# In[3]:

#Load USGS data
#OBSERVATORIES = ('BOU', 'BRW', 'BSL', 'CMO', 'DED', 'FRD', 'FRN', 'GUA', 'HON', 'NEW', 'SHU', 'SIT', 'SJG', 'TUC')
#CHANNELS = ('X', 'Y', 'Z', 'F') 

# Number of magnetic channels/components per observatory
#nchannels = len(CHANNELS)

#execfile('merge_geomag_omni_dataframes.py')
if ('raw_data' in locals()) == False:
    exec(open("./merge_geomag_omni_dataframes.py").read())


# In[4]:

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
                     'Bz, nT (GSE)', 'By, nT (GSM)', 'Bz, nT (GSM)', 'RMS SD B scalar, nT',
                     'RMS SD field vector, nT', 'Flow speed, km/s', 'Vx, km/s, GSE',
                     'Vy, km/s, GSE', 'Vz, km/s, GSE', 'Proton density, n/cc',
                     'Temperture, K', 'Flow pressure, nPa', 'Electric Field, mV/m',
                     'Plasma beta', 'Alfven mach number', 
                     'BOU_X', 'BOU_Y', 'BOU_Z', #'BOU_F',
                     'BRW_X', 'BRW_Y', 'BRW_Z', #'BRW_F',
                     'BSL_X', 'BSL_Y', 'BSL_Z', #'BSL_F', 
                     'CMO_X', 'CMO_Y', 'CMO_Z', #'CMO_F',
                     'DED_X', 'DED_Y', 'DED_Z', #'DED_F',
                     'FRD_X', 'FRD_Y', 'FRD_Z', #'FRD_F',
                     'FRN_X', 'FRN_Y', 'FRN_Z', #'FRN_F',
                     'GUA_X', 'GUA_Y', 'GUA_Z', #'GUA_F',
                     'HON_X', 'HON_Y', 'HON_Z', #'HON_F',
                     'NEW_X', 'NEW_Y', 'NEW_Z', #'NEW_F',
                     'SHU_X', 'SHU_Y', 'SHU_Z', #'SHU_F',
                     'SIT_X', 'SIT_Y', 'SIT_Z', #'SIT_F',
                     'SJG_X', 'SJG_Y', 'SJG_Z', #'SJG_F',
                     'TUC_X', 'TUC_Y', 'TUC_Z']] #, 'TUC_F']]    

units = np.zeros(raw_data.values.shape[1])
units[0:18] = 100000.0
units[18:] = 100.0


nchannels = (raw_data.values.shape)[1]

raw_values = remove_trend(raw_data.values, times)
print(raw_data.values.shape)
print(raw_values.shape)

#raw_data.values = raw_values
window = 60 
sampled_values, time = sting.raw_to_sampled(raw_values.T,window)

# Now we create rows that comprise of timeseries data with different lags, all concatenated together.
lags = np.array([0],dtype='int')

#print("sampled_values.shape",sampled_values.shape)
#print(sampled_values[0,:])


# In[5]:

print(raw_data.values.shape)

#get_ipython().magic('matplotlib inline')
for i in range(nchannels):
    if np.isnan(np.nanmean(sampled_values[i,:])):
        print(i,np.nanmean(sampled_values[i,:]))


# In[6]:

# Scale data
train_scaled, test_scaled, scalers = sting.sampled_to_scaled(sampled_values,
                                                             time, nchannels, lags, 
                                                             batch_size, units=units, 
                                                             derivative=False)

print(train_scaled.shape, test_scaled.shape)
print('len(scalers)',len(scalers))

#Save scaling functions
pickle.dump(scalers, open("scalers_window{0:03d}_lags{1:03d}.pkl".format(window,np.max(lags)),"wb"))


# In[92]:

from scipy.misc import imresize
#%matplotlib inline
#import matplotlib.pyplot as plt
#from scipy import interpolate
#time = np.arange(np.floor(raw_values.shape[1]/window),dtype='int')*window
#print(time)
#print(raw_values.shape)
#i= 0
#y = raw_values[i,time].ravel()
#good = np.isfinite(y)
#f = interpolate.interp1d(time[good].astype(float), y[good],fill_value='extrapolate')
#sampled_values[i,:] = f(time.astype(float))
#sampled_values, time = sting.raw_to_sampled(raw_values,window)
#plt.plot(raw_values[i,:])

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

#img = imresize(np.isfinite(raw_data.values).astype(int),(timeseries.shape[0], timeseries.shape[0]),interp='nearest')
#print(img.max(),img.min())
#pyplot.imshow(img)

#print(raw_values.min(),raw_values.max())
#test = rebin(np.isfinite(timeseries),timeseries.shape[0],timeseries.shape[0])#)
#pyplot.colorbar()
#pyplot.show()


# In[93]:

# Create and fit a list of models
models = []

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

pmodel = persist_model()
pmodel.fit()
models.append(pmodel)


# In[94]:

# LSTM model
lstm_model, lstm_model_pred = sting.fit_lstm_deep(train_scaled, batch_size, 1201, nchannels)
models.append(lstm_model_pred)

lstm_model, lstm_model_pred = sting.fit_lstm_deep(train_scaled, batch_size, 1201, 2*nchannels)
models.append(lstm_model_pred)

lstm_model, lstm_model_pred = sting.fit_lstm_deep(train_scaled, batch_size, 1201, 3*nchannels)
models.append(lstm_model_pred)

#Save trained model. The difference between lstm_model and lstm_model_pred is that the latter has batch_size=1
lstm_model.save_weights('lstm_model_window{0:03d}_lags{1:03d}.h5'.format(window,np.max(lags)))
lstm_model_pred.load_weights('lstm_model_window{0:03d}_lags{1:03d}.h5'.format(window,np.max(lags)))
model_json = lstm_model_pred.to_json()
with open("model_window{0:03d}_lags{1:03d}.json".format(window,np.max(lags)), "w") as json_file:
    json_file.write(model_json)


# In[ ]:




# In[95]:

def metrics(X,X2):
    from scipy.stats import pearsonr
    import numpy as np
    return (np.sum(np.sqrt((X - X2)**2)), (pearsonr(X.ravel(), X2.ravel()))[0])
    
def predict_ahead(X,lookahead,model):
    X_pred = np.zeros([X.shape[0]-lookahead,X.shape[1]])
    for n in range(X.shape[0]-lookahead):
        yhat = X[n,:].reshape([1,1,X.shape[1]])
        #print(yhat.shape)
        for l in range(lookahead):
            #print(yhat.reshape([1,1,X.shape[1]]))
            yhat = model.predict(yhat.reshape([1,1,X.shape[1]]))
        X_pred[n,:] = yhat
    return X[lookahead+range(X.shape[0]-lookahead),:], X_pred[:,:]

lookaheads = np.arange(12,dtype='int')+1

mets = []
mets = np.zeros([nchannels,len(lookaheads),len(models)])
for l in range(len(lookaheads)):
    print('Look aheads = ', lookaheads[l])
    for m in range(len(models)):
        X, X_pred = predict_ahead(test_scaled, lookaheads[l], models[m])
        for c in range(nchannels):
            mets[c,l,m] = (metrics(X[:,c],X_pred[:,c]))[0]


# In[ ]:




# In[96]:

def mystr(a):
    return "{0:.03f}".format(a)

#print('; '.join(map(str,raw_data.columns)))
#for l in range(len(lookaheads)):
#    print('; '.join(map(str,pr[:,l])))
f = open("model_window{0:03d}_lags{1:03d}.metrics".format(window,np.max(lags)), "w")

for c in range(nchannels):
    print("=======================================================================================")
    print("lmst:"+raw_data.columns[c]  + '\t'+'\t'.join(map(mystr,mets[c,:,0])))
    print("pers:"+raw_data.columns[c]  + '\t'+'\t'.join(map(mystr,mets[c,:,1])))
for c in range(nchannels):
    #f.write("================================================================================================================\n")
    for m in range(len(models)):
        f.write("Model #{0:3d}".format(m))
        f.write(":"+raw_data.columns[c]  + ','+','.join(map(mystr,mets[c,:,m])))
        f.write("\n")
f.close()  


# In[ ]:




# In[ ]:




# In[100]:

#inverted_predictions = predictions.reshape([predictions.shape[0],lags.shape[0],nchannels])
#inverted_test = test_scaled.reshape([predictions.shape[0],lags.shape[0],nchannels])
#print("predictions.shape",inverted_predictions.shape)
#print("inverted_test.shape",inverted_test.shape)

#for c in range(len(scalers)):
#    sc = scalers[c]
#    sc.inverse_transform(inverted_predictions[:,:,c])
#    sc.inverse_transform(inverted_test[:,:,c])


# In[ ]:

#get_ipython().magic('matplotlib inline')


# In[101]:

from scipy.stats import pearsonr

for c in range(nchannels):
    pyplot.figure(c)
    obs = int(np.floor(c/len(CHANNELS)))
    comp = c % len(CHANNELS)
    x = inverted_test[:,len(lags)-1,c]
    y = inverted_predictions[:,len(lags)-1,c]
    pyplot.fill(x)
    pyplot.plot(y,color='orange')
    #pyplot.ylim(predictions[:,len(lags)-1,c].min(),predictions[:,len(lags)-1,c].max())
    pyplot.xlim(100,200)
    pyplot.ylim(np.min([x.min(),y.min()]),np.max([x.max(),y.max()]))
    pyplot.ylim(-1,1)
    corr = pearsonr(x[100:200],y[100:200])
    #pyplot.title("{0} ({1}), pearson r={2:.2f}, p={3:.3f}".format(OBSERVATORIES[obs],CHANNELS[comp],corr[0],corr[1]))
    pyplot.title("{0}, pearson r={1:.2f}, p={2:.3f}".format(raw_data.columns[c],corr[0],corr[1]))
    pyplot.show()


# In[ ]:

#from scipy.stats import pearsonr
#from scipy.signal import correlate

#for c in range(nchannels):
#    pyplot.figure(c)
#    obs = int(np.floor(c/len(CHANNELS)))
#    comp = c % len(CHANNELS)
#    x = inverted_test[:,len(lags)-1,c]
#    y = inverted_predictions[:,len(lags)-1,c]
#    corre = correlate(x[100:200],y[100:200])
#    pyplot.plot(corre,'--')
#    pyplot.plot(correlate(x[100:200],x[100:200]))
#    print(corre.argmax(),(correlate(x[100:200],x[100:200])).argmax())
#    pyplot.plot(correlate(y[100:200],y[100:200]))
#    print(corre.argmax(),(correlate(y[100:200],y[100:200])).argmax())
    
#    corr = pearsonr(x[100:200],y[100:200])
#    pyplot.title("{0}, pearson r={1:.2f}, p={2:.3f}".format(raw_data.columns[c],corr[0],corr[1]))
#    pyplot.show()


# In[102]:

#from matplotlib.colors import LogNorm

#f = open("model_window{0:03d}_lags{1:03d}.metrics".format(window,np.max(lags)), "w")

#for c in range(inverted_test.shape[2]):
#    pyplot.figure(c)
#    obs = int(np.floor(c/len(CHANNELS)))
#    comp = c % len(CHANNELS)
#    x = inverted_test[:,len(lags)-1,c]
#    y = inverted_predictions[:,len(lags)-1,c]
#    pyplot.hist2d(x, y, bins=40, norm=LogNorm())
#    pyplot.colorbar()
#    corr = pearsonr(x,y)
#    pyplot.title("{0}, pearson r={1:.2f}, p={2:.3f}".format(raw_data.columns[c],corr[0],corr[1]))
#    pyplot.show()
#f = open("model_window{0:03d}_lags{1:03d}.metrics".format(window,np.max(lags)), "w")
#f.close()


# In[ ]:




# In[ ]:

for c in range(inverted_test.shape[2]):
    pyplot.figure(c)
    obs = int(np.floor(c/len(CHANNELS)))
    comp = c % len(CHANNELS)

    x = inverted_test[0:-2,len(lags)-1,c]
    y = inverted_test[1:-1,len(lags)-1,c]
    pyplot.hist2d(x, y, bins=40, norm=LogNorm())
    #pyplot.axis('equal')
    #pyplot.axis([-5,5,-5,5])
    pyplot.colorbar()
    corr = pearsonr(x,y)
    #pyplot.title("{0} ({1}), pearson r={2:.2f}, p={3:.3f}".format(OBSERVATORIES[obs],CHANNELS[comp],corr[0],corr[1]))
    pyplot.title("{0}, pearson r={1:.2f}, p={2:.3f}".format(raw_data.columns[c],corr[0],corr[1]))
    pyplot.show()


# In[ ]:

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[ ]:

print(raw_values.shape)
pyplot.plot(raw_values[:,50])
#pyplot.ylim(-500,500)


# In[ ]:




# In[ ]:



