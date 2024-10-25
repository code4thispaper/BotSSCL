import numpy as np
from glob import glob

if __name__ == "__main__":

    datasets = ['gilani', 'varol']
    dimensions = ['16', '32']
    for data in datasets:
        for dim in dimensions:
            a_total = np.load(f"Data/{data}_test_{dim}_with_tweets_0.3_corrupted_0.npy")
            for i in range(1, 4):
                a_temp = np.load(f"{data}_test_{dim}_with_tweets_0.3_corrupted_{i}.npy")
                a_total = np.concatenate((a_total, a_temp), axis=0)
            np.save(f"Data/{data}_test_{dim}_with_tweets_0.3_corrupted.npy", a_total)