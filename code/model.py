from preprocess_data import * 
from numpy.random import seed 
seed(1)

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from keras.layers import Input, LSTM, Dense
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from keras.models import Model
import matplotlib.pyplot as plt
import os
from time import time

from util import evaluate_model  

alpha = 1e-6

x_train = x_train.reshape ((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_val = x_val.reshape ((x_val.shape[0], x_val.shape[1] * x_val.shape[2]))
x_test = x_test.reshape ((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

# Linear regression
model = LinearRegression()
start_time = time()    
model.fit(x_train, yield_train)
print("Total train time: ",time()-start_time)
dir_ = 'results/%s'\
            %('linear')

if not os.path.exists(dir_):
    os.makedirs(dir_)
    
train_metrics = evaluate_model (x_train, yield_train, y_train, states_train, 'train', dir_, model)
val_metrics = evaluate_model (x_val, yield_val, y_val, states_val, 'val', dir_, model)
test_metrics = evaluate_model (x_test, yield_test, y_test, states_test, 'test', dir_, model)

# Lasso
model = Lasso(alpha = alpha, random_state = 1)
start_time = time()    
model.fit(x_train, yield_train)
print("Total train time: ",time()-start_time)
dir_ = 'results/%s'\
            %('lasso')
if not os.path.exists(dir_):
    os.makedirs(dir_)
train_metrics = evaluate_model (x_train, yield_train, y_train, states_train, 'train', dir_, model)
val_metrics = evaluate_model (x_val, yield_val, y_val, states_val, 'val', dir_, model)
test_metrics = evaluate_model (x_test, yield_test, y_test, states_test, 'test', dir_, model)

# SVR
model = SVR(kernel = 'rbf', epsilon = 0.1)
start_time=time()    
model.fit(x_train, yield_train)
print("Total train time: ",time()-start_time)
dir_ = 'results/%s'\
            %('svr_rbf')
if not os.path.exists(dir_):
    os.makedirs(dir_)
train_metrics = evaluate_model (x_train, yield_train, y_train, states_train, 'train', dir_, model)
val_metrics = evaluate_model (x_val, yield_val, y_val, states_val, 'val', dir_, model)
test_metrics = evaluate_model (x_test, yield_test, y_test, states_test, 'test', dir_, model)

# MLP
model = MLPRegressor(solver='lbfgs', alpha=alpha,
                     hidden_layer_sizes=(5, 2), random_state=1)
start_time=time()    
model.fit(x_train, yield_train)
print("Total train time: ",time()-start_time)
dir_ = 'results/%s'\
            %('MLP')
if not os.path.exists(dir_):
    os.makedirs(dir_)

train_metrics = evaluate_model (x_train, yield_train, y_train, states_train, 'train', dir_, model)
val_metrics = evaluate_model (x_val, yield_val, y_val, states_val, 'val', dir_, model)
test_metrics = evaluate_model (x_test, yield_test, y_test, states_test, 'test', dir_, model)

#LSTM
shape = x_train.shape[2] 
encoder_input = Input(shape = (Tx, shape))

lstm_1, state_h, state_c = LSTM(128, return_state=True, return_sequences=True)(encoder_input)
lstm_1 = Dropout (0.2)(lstm_1) 
lstm_2, state_h, state_c = LSTM(128, return_state=True, return_sequences=False)(lstm_1)
lstm_2 = Dropout (0.2)(lstm_2) 

yhat = Dense (1, activation = "linear")(lstm_2)
    
pred_model = Model(encoder_input, yhat) 
pred_model.summary()

pred_model.compile(loss='mean_squared_error', optimizer = Adam(lr=0.001)) 

hist = pred_model.fit (x_train, yield_train,
                  batch_size = 512,
                  epochs = 100,
                  verbose = 2,
                  shuffle = True,
                  validation_data=(x_val, yield_val))

loss = hist.history['loss']
val_loss = hist.history['val_loss']

plt.figure()
plt.plot(loss)
plt.plot(val_loss)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.savefig('%s/loss_plot.png'%(dir_))
plt.close()

print("Total train time: ",time()-start_time)
dir_ = 'results/%s'\
            %('MLP')
if not os.path.exists(dir_):
    os.makedirs(dir_)

train_metrics = evaluate_model (x_train, yield_train, y_train, states_train, 'train', dir_, model)
val_metrics = evaluate_model (x_val, yield_val, y_val, states_val, 'val', dir_, model)
test_metrics = evaluate_model (x_test, yield_test, y_test, states_test, 'test', dir_, model)
