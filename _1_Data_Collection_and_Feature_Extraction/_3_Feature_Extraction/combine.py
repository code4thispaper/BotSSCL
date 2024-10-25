import pandas as pd

if __name__ == "__main__":
    
    modes = ['clean']
    dataset = ['varol-icwsm', 'gilani-2017']

    for mode in modes:

        df_timeline = pd.read_csv(f'Data/timeline_{mode}.csv')
        df_profile = pd.read_csv(f"Data/profile_{mode}.csv")
        df_profile = df_profile.drop(columns = ['dataset'])
        df = df_profile.merge(df_timeline, on='id')
        df = df.drop(columns=[f'default_profile_user'])
        df = df.rename(columns={'default_profile_timeline': 'default_profile'})

        for d in dataset:
            df_temp = df[df['dataset'] == d]
            df_temp = df_temp.drop(columns=['dataset'])
            labels = pd.read_csv(f'../Data Collection/Raw Labels/active_users/{d}.csv')
            labels['label'] = labels['label'].apply(lambda x: 1 if x == "bot" else 0)
            df_temp = df_temp.merge(labels, on='id')
            df_temp.to_csv(f"{d}-{mode}.csv", index=False)
