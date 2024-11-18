import pandas as pd 
from os import walk

import matplotlib.pyplot as plt
import numpy as np

mypath = "./csv_files"


# read results and append them in one data frame
for _, _, filenames in walk(mypath):
    for filename in filenames:
        file = mypath + "/" + filename 
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

# Display the result
print(data.head(5))

coulumns = data.columns
# append the mean to the data 
data["Mean"] = data[coulumns[1:]].mean(axis=1)

# append max to the data
data["Max"] = data[coulumns[1:]].max(axis=1)

# append min to the data
data["Min"] = data[coulumns[1:]].min(axis=1)

# append min to the data
data["Std"] = data[coulumns[1:]].std(axis=1)

# append std to the data
print(data[coulumns[1:]].stack().std())

print(data.head(1))

x_data = data["Step"].tolist()
x_data.insert(0, 0)

y_data_mean = data["Mean"].tolist()
y_data_mean.insert(0, y_data_mean[0])

y_data_max = data["Max"].tolist()
y_data_max.insert(0, y_data_mean[0])

y_data_min = data["Min"].tolist()
y_data_min.insert(0, y_data_mean[0])

y_data_std = data["Std"].tolist()
y_data_std.insert(0, 0)

print(max(y_data_max))
print(min(y_data_min))

plt.xlabel("Training steps")
plt.ylabel("Average reward")

plt.plot(x_data, y_data_mean, label="PPO")
plt.fill_between(x_data, np.subtract(y_data_mean, y_data_std), np.add(y_data_mean, y_data_std), alpha=0.2)
plt.xlim(min(x_data), max(x_data))
plt.grid()
plt.legend()
plt.show()