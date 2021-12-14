import numpy as np
import pandas as pd

data = np.load('dataset/Training/inputs_others_train.npy')
data_weather = np.load('dataset/Training/inputs_weather_train.npy', mmap_mode='r+')
data_yield = np.load('dataset/Training/yield_train.npy')
data_dir="data_subset"

mg = data[:,0]
year_data=np.zeros((93028, 3)) 
weather=np.zeros((93028, 214, 8)) 

for index, i in enumerate(data_weather):
    year_data[index,0]=data_yield[index] 
    year_data[index,1]=data[:,3][index]   
    year_data[index,2]=data[:,4][index]   

    weather[index,:,0]= mg[index] 
    weather[index,:,1]= i[:,1] 
    weather[index,:,2]= i[:, 2] 
    weather[index,:,3]= i[:, 3] 
    weather[index,:,4]= i[:, 4] 
    weather[index,:,5]= i[:, 5] 
    weather[index,:,6]= i[:, 6] 

state_list=list()
state_list = pd.DataFrame(state_list).to_csv('%s/state_list_all_data.csv'%(data_dir))
X= weather[:, 0:210, :]  
n_samples = X.shape[0]     
n_timesteps = X.shape[1]   
n_variables = X.shape[2]   

data_interval_3= 30
n_timesteps_3 = n_timesteps/ data_interval_3  # 7
X_monthly = np.zeros((n_samples, 7, n_variables)) 


def data_reshape(dataset, time_interval): 
    nb_timesteps = 210/time_interval   
    nb_timesteps = int(nb_timesteps)
    
    data_reshaped = np.zeros ((nb_timesteps, 8)) 
    for i in range(0, nb_timesteps):
        range_1 = i * time_interval
        range_2 = (i + 1) * time_interval
        data = dataset [range_1:range_2, :]  # (7, 8)
        data_reshaped[i, 0] = np.mean (data[:, 0], axis=0)   
        data_reshaped[i, 1] = np.mean (data[:, 1], axis=0)   
        data_reshaped[i, 2] = np.mean (data[:, 2], axis=0)  
        data_reshaped[i, 3] = np.mean (data[:, 3], axis=0)   
        data_reshaped[i, 4] = np.amax (data[:, 4], axis=0)  
        data_reshaped[i, 5] = np.amax (data[:, 5], axis=0)  
        data_reshaped[i, 6] = np.amin (data[:, 6], axis=0) 
        data_reshaped[i, 7] = np.mean (data[:, 7], axis=0)           
    return data_reshaped


for index in range(0, n_samples):
    X_index = X [index, :, :]
    X_monthly[index, :, :] = data_reshape (X_index, 30)      

np.save("%s/mg_weather_variables_all_data_TS_214_days"%(data_dir), weather)
np.save("%s/yield_year_location_all_data"%(data_dir), year_data)
np.save("%s/mg_weather_variables_all_data_TS_7"%(data_dir), X_monthly)
