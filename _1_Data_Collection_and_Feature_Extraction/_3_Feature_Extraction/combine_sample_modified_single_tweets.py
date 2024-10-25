import pandas as pd
import sys

if __name__ == "__main__":
    
    d = sys.argv[1]

    df_modified_timeline = pd.read_csv(f"{d}_sample_modified_single_tweets_timeline_clean.csv")
    df_clean = pd.read_csv(f'{d}-clean.csv')
    merged_df = pd.merge(df_modified_timeline, df_clean, on='id', suffixes=('_left', '_right'))
    right_columns = []
    non_left_right =[]
    left_columns = []
    for col in merged_df.columns:
        if "_right" in col:
            right_columns.append(col)
        elif "_left" in col:
            left_columns.append(col)
        else:
            non_left_right.append(col)
    
    # Remove right columns
    merged_df = merged_df[non_left_right + right_columns]

    # Rename columns
    renamed_columns = [k.replace("_right", "") for k in right_columns]
    rename = {}
    for i in range(len(renamed_columns)):
        rename[right_columns[i]] = renamed_columns[i]
    merged_df.drop(columns=['default_profile_timeline'],inplace=True)
    merged_df.rename(columns=rename, inplace=True)

    merged_df.drop(columns=['dataset'])

    merged_df.to_csv(f'{d}-clean-sample-modified-single-tweets.csv', index=False)
    print(merged_df.shape)