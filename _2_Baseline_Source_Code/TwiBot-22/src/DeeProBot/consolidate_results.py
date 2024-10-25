import json
import statistics
from glob import glob

if __name__ == "__main__":
    
    files = glob("*results.json")
    
    for k in files:
        with open(k, 'r') as f:
            data = json.load(f)
            f.close()
        
        keys = data[0].keys()
        
        print(k)
        for key in keys:
            temp = [float(j[key]) if j[key] != 'nan' else 0 for j in data]
            print(f"{key}: mean -> {statistics.mean(temp)}, std -> {statistics.stdev(temp)}")



        
