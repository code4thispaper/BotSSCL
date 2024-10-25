from digitaldna import TwitterDDNASequencer
from digitaldna import LongestCommonSubsequence
from digitaldna.verbosity import Verbosity
import os
import pandas as pd
from glob import glob
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import ujson

if __name__ == "__main__":

    files = ['gilani-2017_timelines.json', 'varol-icwsm_timelines.json']
    
    for j in files:

        alphabet = 'b6_content'
        alphabet_type = alphabet.split('_')[0].upper()
        model = TwitterDDNASequencer(input_file=j, alphabet=alphabet)
        arr = model.fit_transform()
        X = [k[-1] for k in arr]
        cwd = os.getcwd()
        estimator = LongestCommonSubsequence(in_path='', out_path='{}/data/glcr_cache'.format(cwd), verbosity=Verbosity.FILE_EXTENDED)
        y = estimator.fit_predict(X)
        df = pd.DataFrame()
        df['id'] = ['u' + k[0] for k in arr]
        df['pred'] = y
        df['pred'] = df['pred'].apply(lambda x: "bot" if x else "human")
        
        # Add labels
        df_compare = pd.read_csv(f"../TwiBot-22/datasets/{j.split('_')[0]}/split.csv")
        df_temp = pd.read_csv(f"../TwiBot-22/datasets/{j.split('_')[0]}/label.csv")
        df_combined = df_compare.merge(df)
        df_combined = df_combined.merge(df_temp)
        df_combined = df_combined.loc[df_combined['split'] == "test"]

        # Make confusion matrix
        actual = df_combined['label']
        predicted = df_combined['pred']
        confusion_matrix = metrics.confusion_matrix(actual, predicted)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
        cm_display.plot()
        plt.title(f"[{j.split('_')[0].capitalize()}] DigitialDNA ({alphabet_type}) Confusion Matrix on Test Dataset Split")
        plt.savefig(f"{j.split('_')[0]}_{alphabet_type}_cm.png")
        plt.clf()
        plt.cla()
        plt.close()

        report = classification_report(df_combined['label'], df_combined['pred'], output_dict=True)

        # Record metrics
        final_json = {
            "model": report,
            'cm': confusion_matrix.tolist()
        }
        filename = f"{j.split('_')[0]}_{alphabet_type}.json"
        with open(filename, 'w') as f:
            ujson.dump(final_json, f, indent=4)
