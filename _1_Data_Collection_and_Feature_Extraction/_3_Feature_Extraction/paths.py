from glob import glob

if __name__ == "__main__":

    folders = ['gilani-2017', 'varol-icwsm']
    
    paths = []

    for folder in folders:
        if 'political' in folder:
            continue
        user_folders = f"{folder}/users/*"
        users = glob(user_folders)
        paths = paths + [k + "\n" for k in users]
        print(f"{folder} done")

    with open("paths.txt", "w") as f:
        f.writelines(paths)
        f.close()

