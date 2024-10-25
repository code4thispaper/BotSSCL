from glob import glob
import re
import pandas as pd
import os

if __name__ == "__main__":

    datasets = glob("*.txt")

    # Make into CSV 
    for file in datasets:
        with open(file, "r") as f:
            lines = f.readlines()
            f.close()
        new_lines = [re.sub(r' +|\t', ',', line) for line in lines]
        new_lines.insert(0, 'id,label\n')
        with open(f"{re.sub(r'.txt', '.csv', file)}", "w") as g:
            g.writelines(new_lines)
            g.close()

    # Standardise data
    datasets = glob("*.csv")
    for file in datasets:
        df = pd.read_csv(file)
        if set(df['label']) == set([0, 1]):
            change_label = lambda x: 'human' if x == 0 else "bot"
            df['label'] = df['label'].apply(change_label)
        df.to_csv(file, index=False)

    # Get active users
    active_data = ["../Datasets/gilani-2017", "../Datasets/varol-icwsm"]
    for dataset in active_data:
        users = glob(f"{dataset}/users/*/profile.json")
        users = [int(k.split('/')[-2]) for k in users]
        print(len(users))
        dataset_str = dataset.split('/')[-1]
        df_original = pd.read_csv(f"{dataset_str}.csv")
        df_new = df_original.loc[df_original['id'].isin(users)]
        df_new = df_new.drop_duplicates(subset='id', keep=False)
        df_new.to_csv(f"active_users/{dataset_str}.csv", index=False)
        print(df_new.shape)
        print(f"{dataset_str} finished cleaning")

