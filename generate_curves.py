import pandas as pd 
from os import walk

import matplotlib.pyplot as plt
import numpy as np

mypath = "./csv_files"
PPO_dir = "/PPO/"
CaLFP_dir = "/PPO_CaLFP/"

def read_files(path):
    # read results and append them in one data frame
    for _, _, filenames in walk(path):
        for filename in filenames:
            file = path + filename 
            if "data" not in locals():
                name = filename.split(".")[0]
                temp = pd.read_csv(file, usecols=["Step", "Value"])
                data = pd.DataFrame()
                data["Step"] = temp["Step"]
                data[name] = temp["Value"]
            else:
                name = filename.split(".")[0]
                temp = pd.read_csv(file, usecols=["Step", "Value"])
                temp = pd.read_csv(file)
                data[name] = temp["Value"]
    
    return exclude_mean_std(data)

def exclude_mean_std(data:pd.DataFrame):
    coulumns = data.columns
    # Get number of traiting steps and adding the first value
    x_data = data["Step"].tolist()
    x_data.insert(0, 0)
    
    data_mean = data[coulumns[1:]].mean(axis=1).tolist()
    data_mean.insert(0, data_mean[0])
    
    
    data_std = data[coulumns[1:]].std(axis=1).tolist()
    data_std.insert(0, 0)
    
    return x_data, data_mean, data_std

x_data, PPO_mean, PPO_std = read_files(mypath + PPO_dir)
x_data, CaLFP_mean, CaLFP_std = read_files(mypath + CaLFP_dir)

plt.xlabel("Training steps")
plt.ylabel("Average reward")

plt.plot(x_data, PPO_mean, 'c', label="PPO")
plt.fill_between(x_data, np.subtract(PPO_mean, PPO_std), np.add(PPO_mean, PPO_std), color = 'c', alpha=0.2)

plt.plot(x_data, CaLFP_mean, 'g', label="PPO_CaLF_Policy")
plt.fill_between(x_data, np.subtract(CaLFP_mean, CaLFP_std), np.add(CaLFP_mean, CaLFP_std), color = 'g', alpha=0.2)


plt.xlim(min(x_data), max(x_data))
plt.grid()

plt.legend()
plt.show()