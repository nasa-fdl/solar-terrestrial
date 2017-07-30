
# Replace nans with mean
def nan_to_mean(a):
    import numpy as np
    
    col_mean = np.nanmean(a,axis=0)
    #Find indicies that you need to replace
    inds = np.where(np.isnan(a))
    #Place column means in the indices. Align the arrays using take
    #print(inds)
    a[inds]=col_mean#np.take(col_mean,inds[1])
    return a

def nan_to_median(a):
    import numpy as np
    
    col_mean = np.nanmedian(a,axis=0)
    #Find indicies that you need to replace
    inds = np.where(np.isnan(a))
    #Place column means in the indices. Align the arrays using take
    #print(inds)
    a[inds]=col_mean#np.take(col_mean,inds[1])
    return a

# Running Average in time
def cumsum_sma(array, period):
    import numpy as np

    ret = np.cumsum(array, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period

# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime("190"+x, '%Y-%m')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# create a differenced series
def difference(dataset, interval=1):
    from pandas import Series

    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test, scaler=None):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from numpy import array, nanmean, nanmedian, sqrt, concatenate
    
    # fit scaler
    if scaler is None:
        scaler = StandardScaler()
        #scaler = MinMaxScaler(feature_range=(-1, 1))
        #scaler = scaler.fit(train)
        nmean = nanmean(concatenate((train,test)))
        nmedian = nanmedian(concatenate((train,test)))
        scaler.fit(array([nmedian,nmedian+2*100.0,nmedian-2*100.0]))
    else:
        print("Scaler scale, mean",scaler.scale_,scaler.mean_)
    
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    import numpy as np
    
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# inverse scaling for a forecasted value
def invert_scales(scalers, X, value):
    import numpy as np

    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

def raw_to_sampled(raw_values, window):
    import numpy as np
    from scipy import interpolate
    
    time = np.arange(np.floor(raw_values.shape[1]/window),dtype='int')*window
    print("time.shape",time.shape)
    nchannels = raw_values.shape[0]
    print("nchannels=",nchannels)
    sampled_values = np.zeros([nchannels,time.shape[0]],dtype='float')
    #print(sampled_values.shape)
    
    for i in range(nchannels):
        y = (raw_values[i,time]).reshape(time.shape[0])
        good = np.where(np.isfinite(y))
        #print("good=",good)
        #print(y[good[0]], y[good[-1]])
        #print("times[good]=",time[good])
        #print(y.shape)
        #f = interpolate.interp1d(time[good].astype(float), y[good], bounds_error=False,fill_value=np.nanmedian(y))
        #Interpolate over nans
        
        #sampled_values[i,:] = f(time)
                                 
        # Replace nans with channel mean
        raw_values[i,:] = nan_to_median(raw_values[i,:])
        # Do rolling time average
        sampled_values[i,:]=((cumsum_sma(raw_values[i,:],window))[np.arange(time.shape[0],dtype='int')*window])
        
    #print(sampled_values.shape)
    return sampled_values, time

def sampled_to_scaled(sampled_values, time, nchannels, lags, batch_size, derivative=False, scalers=None):
    import numpy as np
    if derivative:
        diff_values = np.zeros([nchannels,time.shape[0]-1],dtype='float')
        for c in range(nchannels):
            diff_values[c,:] = difference(sampled_values[c,:], 1)
    else:
        diff_values = sampled_values
        
        #diff_values = difference(sampled_values, 1)
    #print("diff_values.shape",(diff_values).shape)

    # transform data to be supervised learning
    #supervised = timeseries_to_supervised(diff_values, 1)
    #supervised_values = supervised.values
    #print(supervised_values.shape)

    supervised_values = np.zeros([diff_values.shape[1],lags.shape[0],nchannels],dtype='float')
    print("supervised_values.shape",supervised_values.shape)
    for c in range(nchannels):
        for i in range(len(lags)):
            supervised_values[:,i,c] = np.roll(diff_values[c,:].reshape(diff_values.shape[1]),-lags[i])
           # supervised_values[:i,lags[i]-i,c] = 0.0
    
    print("supervised_values.shape",supervised_values.shape)
    
    # split data into train and test-sets
    testfrac = 0.2 # Fraction of data set reserved for testing
    wall = int(np.floor(np.floor(supervised_values.shape[0]*(1.0-testfrac))/batch_size)*batch_size)#+lags.max()+1
    print(wall)
    train = supervised_values[0:wall,:,:]
    #test = supervised_values[(supervised_values.shape[0]-wall):,:,:]
    test = supervised_values[wall:,:,:]
    
    #for i in range(len(lags)):
    #    print(train[i,:,0])

    print("train.shape",train.shape)
    print("test.shape",test.shape)

    train_scaled = train
    test_scaled = test
    if scalers is None:
        scalers =[]
        for c in range(nchannels):
            this_scaler, train_scaled_channel, test_scaled_channel = scale(train[:,:,c], test[:,:,c])
            train_scaled[:,:,c] = train_scaled_channel
            test_scaled[:,:,c] = test_scaled_channel 
            scalers.append(this_scaler)
    else:
        for c in range(nchannels):
            this_scaler, train_scaled_channel, test_scaled_channel = scale(train[:,:,c], test[:,:,c], scalers[c])
            train_scaled[:,:,c] = train_scaled_channel
            test_scaled[:,:,c] = test_scaled_channel 
        
    train_scaled = train_scaled.reshape(train_scaled.shape[0], train_scaled.shape[1]*train_scaled.shape[2])
    test_scaled = test_scaled.reshape(test_scaled.shape[0], test_scaled.shape[1]*test_scaled.shape[2])
    return train_scaled, test_scaled, scalers

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    import numpy as np 
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM

    
    wall = int((np.floor(train.shape[0]/batch_size)-1)*batch_size)
    X, y = train[0:wall,:], train[1:1+wall,:]
    
    #X, y = train[0:-lag, :], train[lag:,:]
    print(X.shape,y.shape)
    X = X.reshape(X.shape[0], 1, X.shape[1]) 
    print(X.shape,y.shape)
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, return_sequences=True,dropout=0.2))
    #model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True,dropout=0.2))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    modelpred = Sequential()
    modelpred.add(LSTM(neurons, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True, return_sequences=True, dropout=0.2))
    #modelpred.add(LSTM(neurons, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True, dropout=0.2))
    modelpred.add(Dense(y.shape[1]))
    modelpred.compile(loss='mean_squared_error', optimizer='adam')
        
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=(i%20==0), shuffle=False)
        model.reset_states()
        
    return model, modelpred

# fit an LSTM network to training data
def fit_lstm_deep(train, batch_size, nb_epoch, neurons):
    import numpy as np 
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM

    wall = int((np.floor(train.shape[0]/batch_size)-1)*batch_size)
    X, y = train[0:wall,:], train[1:1+wall,:]
    
    #X, y = train[0:-lag, :], train[lag:,:]
    print(X.shape,y.shape)
    X = X.reshape(X.shape[0], 1, X.shape[1]) 
    print(X.shape,y.shape)
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, return_sequences=True,dropout=0.2))
    #model.add(LSTM(neurons))#, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, return_sequences=False,dropout=0.2))
    #model.add(LSTM(neurons))#, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, return_sequences=False,dropout=0.2))
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, dropout=0.2))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    modelpred = Sequential()
    modelpred.add(LSTM(neurons, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True, return_sequences=True, dropout=0.2))
    #model.add(LSTM(neurons))#, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True, return_sequences=False, dropout=0.2))
    #model.add(LSTM(neurons))#, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True, return_sequences=False, dropout=0.2))
    modelpred.add(LSTM(neurons, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True, dropout=0.2))
    modelpred.add(Dense(y.shape[1]))
    modelpred.compile(loss='mean_squared_error', optimizer='adam')
        
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=(i%20==0), shuffle=False)
        model.reset_states()
  
    return model, modelpred

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat