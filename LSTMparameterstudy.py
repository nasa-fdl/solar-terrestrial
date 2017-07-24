'''Example script showing how to use stateful RNNs
to model long sequences efficiently. Adapted from Keras examples.
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM


# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 50
epochs = 5000
neurons = 100
length = 500
period = 100
layers = 1
# number of elements ahead that are used to make the prediction
lahead = 1



def gen_cosine_amp(amp=100, period=period, x0=0, xn=length, step=1, k=0.01):
    """Generates an absolute cosine time series with the amplitude
    exponentially decreasing

    Arguments:
        amp: amplitude of the cosine function
        period: period of the cosine function
        x0: initial x of the time series
        xn: final x of the time series
        step: step of the time series discretization
        k: exponential rate
    """
    cos = np.zeros(((xn - x0) * step, 1, 1))
    for i in range(len(cos)):
        idx = x0 + i * step
        cos[i, 0, 0] = amp * np.cos(2 * np.pi * idx / period)
        cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k * idx)
    return cos


print('Generating Data...')
cos = gen_cosine_amp()
print('Input shape:', cos.shape)

expected_output = np.zeros((len(cos), 1))
for i in range(len(cos) - lahead):
    expected_output[i, 0] = np.mean(cos[i + 1:i + lahead + 1])

print('Output shape:', expected_output.shape)

print('Creating Model...')
model = Sequential()
if layers<2:
    model.add(LSTM(neurons,
                   input_shape=(tsteps, 1),
                   batch_size=batch_size,
                   return_sequences=False,
                   stateful=True))
else:
    model.add(LSTM(neurons,
                   input_shape=(tsteps, 1),
                   batch_size=batch_size,
                   return_sequences=True,
                   stateful=True))
    for i in range(layers-2):
        model.add(LSTM(neurons,
                       return_sequences=True,
                       stateful=True))
    model.add(LSTM(neurons,
                   return_sequences=False,
                   stateful=True))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)

    # Note that the last state for sample i in a batch will
    # be used as initial state for sample i in the next batch.
    # Thus we are simultaneously training on batch_size series with
    # lower resolution than the original series contained in cos.
    # Each of these series are offset by one step and can be
    # extracted with cos[i::batch_size].

    model.fit(cos, expected_output,
              batch_size=batch_size,
              epochs=1,
              verbose=1,
              shuffle=False)
    model.reset_states()

print('Predicting')
#Invert and shrink the test set!
predicted_output = model.predict(-0.5*cos, batch_size=batch_size)

print('Plotting Results')
plt.plot(-0.5*expected_output)
plt.plot(predicted_output)
plt.title('Expected and Predicted\ntsteps = '+str(tsteps)+', batch_size = '+str(batch_size)+', epochs = '+str(epochs)+'\nneurons = '+str(neurons)+', layers = '+str(layers)+', length = '+str(length)+', period = '+str(period))
plt.show()
