import json
import pandas as pd
from glob import glob

if __name__ == "__main__":

    folders = glob("*/")

    df_as_dict = {}

    models = []
    for folder in folders:
        files = glob(f"{folder}/*/*.json")
        for f in files:
            with open(f, "r") as g:
                data = json.load(g)
            models.append(f.split('/')[0])
            if 'model' not in data:
                data = data['report']
                if 'model' in data:
                    data = data['model']
            else:
                data = data['model']
                if 'report' in data:
                    data = data['report']
            col = f"{f}".replace(".json", "")
            rows = {
                "Accuracy": data["accuracy"],
                "Precision": data["macro avg"]["precision"],
                "Recall": data["macro avg"]["recall"],
                "F1 Score": data["macro avg"]["f1-score"],
            }
            df_as_dict[col] = rows
    
    df = pd.DataFrame().from_dict(df_as_dict).transpose()
    df.reset_index(inplace=True)
    df['Dataset'] = df['index'].apply(lambda x: x.split('/')[-2])
    df['Model'] = models
    df_final = df[['Model', 'Dataset', 'F1 Score', 'Precision', 'Recall', 'Accuracy']]
    df_final = df_final.sort_values(by='F1 Score', ascending=False)
    
    # Split into seperate csv files
    df_varol = df_final[df_final['Dataset'] == 'varol-icwsm']
    df_gilani = df_final[df_final['Dataset'] == 'gilani-2017']
    df_twibots1 = df_final[df_final['Dataset'] == 'twibot-s1']
    df_twibots2 = df_final[df_final['Dataset'] == 'twibot-s2']
    df_varol.to_csv('varol.csv', index=False)
    df_gilani.to_csv('gilani.csv', index=False)
    df_twibots1.to_csv('twibot-s1.csv', index=False)
    df_twibots2.to_csv('twibot-s2.csv', index=False)


