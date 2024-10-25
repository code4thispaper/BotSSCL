import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

def remove_botometer(df):
    # Remove baselines to not be considered.
    df = df[df.Model != "Botometer_0.6"]
    df = df[df.Model != "Botometer_0.7"]
    df = df[df.Model != "Botometer Lite"]
    df['Model'] = df['Model'].apply(lambda x: "Botometer" if x == "Botometer_0.5" else x)
    return df

if __name__ == "__main__":
    
    # Get data
    df1 = remove_botometer(pd.read_csv("varol.csv"))
    df2 = remove_botometer(pd.read_csv("gilani.csv"))

    models_varol = df1['Model'].apply(lambda x: re.sub("DigitalDNA", "DDNA", x)).tolist()
    varol_metrics = {}
    metric_labels = ["F1 Score","Precision","Recall","Accuracy"]
    for metric in metric_labels:
        varol_metrics[metric] = df1[metric].tolist()
    
    models_gilani= df2['Model'].apply(lambda x: re.sub("DigitalDNA", "DDNA", x)).tolist()
    gilani_metrics = {}
    for metric in metric_labels:
        gilani_metrics[metric] = df2[metric].tolist()

    x = np.arange(len(models_varol))
    width = 0.15
    multiplier = 0
    fig, ax = plt.subplots(2, layout="constrained",figsize = (6,4), dpi=400)

    # Format varol graph
    i = 0
    fill = ['/', 'x', 'o' '.', '*']
    for attribute, measurement in varol_metrics.items():
        offset = width * multiplier
        rects = ax[0].bar(x + offset, measurement, width, label=attribute, edgecolor="black", fill=False, hatch=fill[i])
        multiplier += 1
        i += 1
    ax[0].set_xticks(x + width, models_varol)
    ax[0].legend(loc='upper center', fontsize="10", ncol=len(df1.columns), handleheight=2)
    ax[0].set_ylim(0.3, 1)

    # Format gilani graph
    multiplier = 0
    i = 0
    for attribute, measurement in gilani_metrics.items():
        offset = width * multiplier
        rects = ax[1].bar(x + offset, measurement, width, label=attribute, edgecolor="black", fill=False, hatch=fill[i])
        multiplier += 1
        i += 1
    ax[1].set_xticks(x + width, models_gilani)
    ax[1].legend(loc='upper center', fontsize="10", ncol=len(df2.columns), handleheight=2)
    ax[1].set_ylim(0.3, 1)

    ax[0].set_ylabel("Performance")
    ax[1].set_ylabel("Performance")

    plt.savefig("baselines.png")
