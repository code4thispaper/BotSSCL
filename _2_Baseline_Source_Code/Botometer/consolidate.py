import ujson
from glob import glob
from sklearn.metrics import classification_report
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import metrics

if __name__ == "__main__":
    
    folders = ['collected_data/varol-icwsm', 'collected_data/gilani-2017']

    for folder in folders:
        data = glob(f"{folder}/*.json")
        ids = []
        score = []
        for d in data:
            with open(d, "r") as f:
                user = ujson.load(f)
                f.close()
            try:
                ids.append("u" + user['user']['user_data']['id_str'])
                score.append(user['display_scores']['english']['overall'])
            except:
                continue
        
        df = pd.DataFrame()
        df['id'] = ids
        df['score'] = score
        bot_human_threshold = 0.7
        df['pred'] = df.apply(lambda row: "bot" if row['score'] > 5 * bot_human_threshold else "human", axis=1)

        # Add label
        df1 = pd.read_csv(f"../../datasets/{folder}/label.csv")
        df = df.merge(df1, on=['id'])

        # Get data on test split
        df2 = pd.read_csv(f"../../datasets/{folder}/split.csv")
        df2 = df2[df2["split"] == "test"]
        df_combined = df.merge(df2, on=['id'])

        # Record metrics
        report = classification_report(df_combined['label'], df_combined['pred'], output_dict=True)

        # Make CM
        actual = df_combined['label']
        predicted = df_combined['pred']
        confusion_matrix = metrics.confusion_matrix(actual, predicted)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
        cm_display.plot()
        plt.title(f"[{folder.capitalize()}] Botometer ({bot_human_threshold} Threshold) Confusion Matrix on Test Dataset Split")
        plt.savefig(f"{folder}_{bot_human_threshold}_cm.png")
        plt.clf()
        plt.cla()
        plt.close()

        final_json = {
            "embedding": report,
            'cm': confusion_matrix.tolist()

        }
        filename = f"{folder}_{bot_human_threshold}.json"
        with open(filename, 'w') as f:
            ujson.dump(final_json, f, indent=4)
