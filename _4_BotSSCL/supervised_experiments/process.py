import json
import pandas as pd
from glob import glob
import re

if __name__ == "__main__":

    files = glob("*.json")
    files.sort()

    df = pd.DataFrame()
    input_d = []
    output_d = []
    sup_version = []
    precision = []
    recall = []
    f1 = []
    accuracy = []
    performance_time = []
    dataset = []

    for f in files:
        dataset.append(f.split('_')[0])
        input_d.append(f.split('_')[1])
        output_d.append(f.split('_')[2])
        if f.split('_')[-1] == 'v2.json':
            sup_version.append(2)
        else:
            sup_version.append(1)
        with open(f, "r") as d:
            data = json.load(d)
            report = data['reports'][4]['macro avg']
            precision.append(report['precision'])
            recall.append(report['recall'])
            f1.append(report['f1-score'])
            accuracy.append(data['reports'][4]['accuracy'])
            performance_time.append(data['performance_time'][0])
            d.close()
        
    df['Dataset'] = dataset
    df['Input'] = input_d
    df['Output'] = output_d
    df['Sup Version'] = sup_version
    df['Precision'] = precision
    df['Recall'] = recall
    df['F1 Score'] = f1
    df['Accuracy'] = accuracy
    df['Time (s)'] = performance_time

    df.to_csv("sup_experiment.csv", index=False)
