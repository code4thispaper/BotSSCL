import pandas as pd
import sys
from glob import glob

if __name__ == "__main__":

    d = sys.argv[1]
    files = glob(f"{d}-sample-modified-tweets*.csv")
    files.sort()

    df_combined = pd.read_csv(files[0])
    
    for i in range(1, len(files)):
        df_temp = pd.read_csv(files[i])
        df_combined = pd.concat(df_combined, df_temp, ignore_index=True)
    
    df_combined.to_csv(f"{d}-sample-modified-tweets.csv")


