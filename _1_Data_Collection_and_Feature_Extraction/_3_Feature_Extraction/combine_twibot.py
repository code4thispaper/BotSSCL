import pandas as pd

if __name__ == "__main__":
    
    modes = ['clean']
    dataset = ['twibot-s1', 'twibot-s2']

    for mode in modes:

        df_timeline = pd.read_csv(f'timeline_{mode}_twibot.csv')
        df_profile = pd.read_csv(f"profile_{mode}_twibot.csv")
        df_profile = df_profile.drop(columns = ['dataset'])
        df = df_profile.merge(df_timeline, on='id')

        for d in dataset:
            df_temp = df[df['dataset'] == d]
            df_temp = df_temp.drop(columns=['dataset'])
            labels = pd.read_csv(f'../datasets/active_users/{d}.csv')
            labels['label'] = labels['label'].apply(lambda x: 1 if x == "bot" else 0)
            df_temp = df_temp.merge(labels, on='id')
            df_temp.to_csv(f"{d}-{mode}.csv", index=False)
