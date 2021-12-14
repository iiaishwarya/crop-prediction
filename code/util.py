import matplotlib.pyplot as plt
import numpy as np
from time import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.preprocessing import MinMaxScaler


def get_data(data, cluster):
    data = data
    cluster = cluster
    return data, cluster


def actual_pred_plot(y_actual, y_pred, n_samples=60, dir='default'):
    plt.figure()
    plt.plot(y_actual[: n_samples, -1])
    plt.plot(y_pred[: n_samples, -1])
    plt.legend(['Ground Truth', 'Model Prediction'], loc='upper right')
    plt.savefig('%s/plot_truth.png' % (dir))
    plt.close()


def scatter_plot(y_actual, y_pred, dir='default'):
    plt.figure()
    plt.scatter(y_actual[:, -1], y_pred[:, -1])
    plt.plot([y_actual.min(), y_actual.max()], [
             y_actual.min(), y_actual.max()], 'k--', lw=4)
    plt.title('Predicted Value Vs Actual Value')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.savefig('%s/scatter_plot.png' % (dir))
    plt.close()


def evaluate_model(x_data, yield_data, y_data, states_data, dataset, dir_='default', model=''):
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    scaler_x = scaler_x.fit(x_data)   
    scaler_y = scaler_y.fit(yield_data)    

    if dataset == "test":
        start_time = time()
    yield_data_hat = model.predict(x_data)
    if dataset == "test":
        print("Total testing time: ", time()-start_time)

    yield_data_hat = yield_data_hat.reshape((yield_data_hat.shape[0], 1))
    yield_data_hat = scaler_y.inverse_transform(yield_data_hat)

    yield_data = scaler_y.inverse_transform(yield_data)

    metric_dict = {}

    data_rmse = sqrt(mean_squared_error(yield_data, yield_data_hat))
    metric_dict['rmse'] = data_rmse
    print('%s RMSE: %.3f' % (dataset, data_rmse))

    data_mae = mean_absolute_error(yield_data, yield_data_hat)
    metric_dict['mae'] = data_mae
    print('%s MAE: %.3f' % (dataset, data_mae))

    data_r2score = r2_score(yield_data, yield_data_hat)
    metric_dict['r2_score'] = data_r2score
    print('%s r2_score: %.3f' % (dataset, data_r2score))

    y_data = np.append(y_data, yield_data_hat, axis=1)   # (10336, 7)

    if dataset == 'test':
        actual_pred_plot(yield_data, yield_data_hat, 60, dir_)
        scatter_plot(yield_data, yield_data_hat, dir_)
