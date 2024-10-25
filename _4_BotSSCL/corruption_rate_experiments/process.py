import json
import pandas as pd
from glob import glob
import re

if __name__ == "__main__":

    files = glob("varol-icwsm*.json")
    files.sort()

    df = pd.DataFrame()
    input_d = []
    output_d = []
    corr_rate = []
    precision = []
    recall = []
    f1 = []
    accuracy = []
    performance_time = []

    for f in files:
        input_d.append(f.split('_')[1])
        output_d.append(f.split('_')[2])
        corr_rate.append(re.sub(".json", "", f.split('_')[3]))
        with open(f, "r") as d:
            data = json.load(d)
            report = data['reports'][4]['macro avg']
            precision.append(report['precision'])
            recall.append(report['recall'])
            f1.append(report['f1-score'])
            accuracy.append(data['reports'][4]['accuracy'])
            performance_time.append(data['performance_time'][0])
            d.close()
        
    df['Input'] = input_d
    df['Output'] = output_d
    df['Corruption Rate'] = corr_rate
    df['Precision'] = precision
    df['Recall'] = recall
    df['F1 Score'] = f1
    df['Accuracy'] = accuracy
    df['Time (s)'] = performance_time

    df.to_csv("corr_rate_experiment.csv", index=False)
