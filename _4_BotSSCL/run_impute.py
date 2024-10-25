
from glob import glob
import numpy as np
import os

def get_partitions(no_rows, no_partitions):
    partition_size = int(no_rows / no_partitions)
    indices = [k * partition_size for k in range(no_partitions)]
    indices.append(no_rows)
    ranges = [(indices[k], indices[k + 1]) for k in range(no_partitions)]
    return ranges

if __name__ == "__main__":

    file_path = "Data/*_with_tweets.npy"
    files = glob(file_path)
    corr_rate = 0.3

    for file in files:
        shape = np.load(file).shape
        partitions = get_partitions(shape[0], 4)
        for k in range(len(partitions)):
            start = partitions[k][0]
            end = partitions[k][1]
            command = f"""#!/bin/sh
            source venv/bin/activate
            python3 impute_data.py {file} {corr_rate} {start} {end} {k}
            """
            script_name = f"run_impute_timeline_{file.split('/')[-1]}_{start}_{end}.sh"
            with open(script_name, "w") as f:
                f.write(command)
            os.system(f'sh {script_name}')
            
