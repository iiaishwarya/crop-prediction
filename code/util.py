from numpy.random import seed 
seed(1)
import tensorflow

tensorflow.random.set_seed(2)

def get_data(data_x, data_x_mg_cluster):
    data_x = data_x
    data_x_mg_cluster = data_x_mg_cluster
    
    return data_x, data_x_mg_cluster
