import os
import pandas as pd

data_file_folder = ".\SOI Tax Stats"

df = []
for file in os.listdir(data_file_folder):
    if file.endswith(".csv"):
        print("Loading file {0}...".format(file))
        df.append(pd.read_csv(os.path.join(data_file_folder, file)))

print(len(df))

df_master = pd.concat(df, axis=0)
# df_master.to_csv('master file.csv', index = False)

n1mean = df_master["n1"].mean()
