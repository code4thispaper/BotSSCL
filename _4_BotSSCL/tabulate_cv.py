import json
from glob import glob
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_index(index):
    if index == 16:
        return 0
    elif index == 32:
        return 1
    elif index == 64:
        return 2
    else:
        return 3

def return_aug_title(aug):
    if aug == "aug1":
        return "Augmentation 1"
    elif aug == "aug2":
        return "Augmentation 2"
    else:
        return "Augmentation 3"


def return_dataset_no(dataset):
    if dataset == "varol-icwsm":
        return 1
    else:
        return 2

def generate_heat_map(data, metric, dataset, aug_method):
    # Convert the data into a DataFrame for Seaborn heatmap

    # Create the subplots and axes
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)  # Create a figure and axes

    # Create the heatmap using Seaborn on the specified axes
    sns.set(font_scale=1.5)
    sns.heatmap(data, annot=True, cmap='YlGnBu', linewidths=0.5, square=True, ax=ax, annot_kws={"size": 40})

    ax.set_title(f'{return_aug_title(aug_method)} {metric} (%)', fontdict={"fontsize": 20, 'fontweight': "bold"})  # Set the title for the axes\

    # Set custom labels for the x and y axes
    x_labels = [2 ** i for i in range(4, 8)]
    y_labels = [2 ** i for i in range(4, 8)]

    ax.set_xticklabels(x_labels, fontsize=20)
    ax.set_yticklabels(y_labels, fontsize=20)

    ax.set_ylabel('Input Dimension', fontdict={"fontsize": 18})
    ax.set_xlabel('Feature Dimension D, (Output = Dx4)', fontdict={"fontsize": 18})

    plt.savefig(f"{aug_method}/{dataset}_{aug_method}_{metric}.png", bbox_inches='tight', dpi=300)
    plt.close()


def generate_heat_maps(df, dataset, aug_method):
    
    # Generate heat maps
    df_temp = df[df['Dataset'] == dataset]
    np_accuracy = np.zeros((4,4), dtype=np.int)
    np_f1 = np.zeros_like(np_accuracy)
    np_precision = np.zeros_like(np_accuracy)
    np_recall = np.zeros_like(np_accuracy)
    for index, row in df_temp.iterrows():
        i = get_index(int(row['Input Embedding Dimension']))
        j = get_index(int(row['Output Embedding Dimension']))
        np_accuracy[i][j] = row['Accuracy'] * 100
        np_f1[i][j] = row['F1-Score'] * 100
        np_precision[i][j] = row['Precision'] * 100
        np_recall[i][j] = row['Recall'] * 100
    
    generate_heat_map(np_accuracy, 'Accuracy', dataset, aug_method)
    generate_heat_map(np_f1, "F1-Score", dataset, aug_method)
    generate_heat_map(np_precision, 'Precision', dataset, aug_method)
    generate_heat_map(np_recall, "Recall", dataset, aug_method)


def run(folder):

    files = glob(f"{folder}/*_10_strat_[1-9][0-9]*.json")
    files.sort()

    f1 = []
    dataset = []
    accuracy = []
    precision = []
    recall = []
    ins = []
    output = []
    classifier = []
    seed = []
    no_fold = []

    report_name = ["Balanced Logistic Regression", "Logistic Regression", "LinearSVC", "Random Forest"]
    for f in files:
        # Get metadata
        with open(f, "r") as j:
            data = json.load(j)
        seed.append(f.split('_')[-1].replace(".json", ""))
        no_fold.append(int(f.split('_')[3]))

        
        report = [data['reports'][4]]
        for i in range(len(report)):
            metadata = f.split('_')    
            dataset.append(metadata[0].capitalize())
            ins.append(metadata[1])
            output.append(metadata[2].split('.')[0])
            accuracy.append(report[i]["accuracy"])
            f1.append(report[i]["macro avg"]["f1-score"])
            precision.append(report[i]["macro avg"]["precision"])
            recall.append(report[i]["macro avg"]["recall"])
            classifier.append(report_name[i])
    
    df = pd.DataFrame()
    df['Dataset'] = dataset
    df['Dataset'] = df['Dataset'].apply(lambda x: x.replace(f"{folder.capitalize()}/", ""))
    df['Classifier'] = classifier
    df['Input Embedding Dimension'] = ins
    df['Output Embedding Dimension'] = output
    df['Accuracy'] = accuracy
    df['F1-Score'] = f1
    df['Precision'] = precision
    df['Recall'] = recall
    df['Fold'] = no_fold
    df['Fold'] =  df['Fold'].apply(lambda x: x + 1)
    df['Seed'] = seed

    # Save CSV
    df.sort_values(by=['Dataset', 'Fold', 'Seed'], inplace=True)
    df.to_csv(f"{folder}_10_strat_seed_cv_results.csv", index=False)



if __name__ == "__main__":

    folders = ["aug1"]
    for f in folders:
        run(f)
