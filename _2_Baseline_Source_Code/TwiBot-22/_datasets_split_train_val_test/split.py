import ujson
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":

    # Generate nodes
    folders = [
        "../../../Data Collection/Datasets/gilani-2017",
        "../../../Data Collection/Datasets/varol-icwsm"
    ]

    '''
    # Add u
    for folder in folders:
        profiles = []
        users = glob(f"{folder}/users/*/profile.json")
        print(f"{folder} {len(users)}")
        print("Started")
        for user in users:
            with open(user, "r") as f:
                data = ujson.load(f)
                data['id'] = 'u' + data['id']
                if not 'location' in data:
                    data['location'] = ""
                profiles.append(data)
        new_folder = folder.split('/')[-1]
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        with open(new_folder + "/node.json", "w") as g:
            ujson.dump(profiles, g, indent = 4)
            g.close()
        print(f"Finished {new_folder}")
    '''
    
    
    # Generate data splitting
    folders = ['gilani-2017', 'varol-icwsm']
    for folder in folders:
        
        # Copy over data
        os.system(f'cp -f ../../DataCollectionCode/datasets/active_users/{folder}.csv {folder}/label.csv')  

        # Fix label csv
        df1 = pd.read_csv(f"{folder}/label.csv")
        df1['id'] = df1['id'].apply(lambda x: 'u' + str(x))
        df1.to_csv(f"{folder}/label.csv", index=False)

        # Create split
        df2 = df1.copy(deep=True)
        train_size = 0.7
        X = df2['id']
        y = df2['label']
        X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=train_size)
        # Now since we want the valid and test size to be equal (10% each of overall data). 
        # we have to define valid_size=0.5 (that is 50% of remaining data)
        test_size = 0.67
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=test_size)
        
        split_df = pd.DataFrame().reindex_like(df1)
        split_df['id'] = list(X_train) + list(X_valid) +  list(X_test)
        split_df['label'] = ["train" for _ in range(X_train.shape[0])] + ["val" for _ in range(X_valid.shape[0])] + ["test" for _ in range(X_test.shape[0])]
        split_df.rename(columns={"label": "split"}, inplace=True)
        split_df.to_csv(f"{folder}/split.csv", index=False)
        print(f"{folder} finished")
