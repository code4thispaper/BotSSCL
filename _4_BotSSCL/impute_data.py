# Original dataframe
# Make another copy
# 2074, 128 for original and copy
# Make 50% random for the first row of original data
# Fit using mice and transform using mice
# copy this row and save it (in dataframe
# Use original instance (with no imputed data)
# Repeat process for other rows
# Final transmuted dataframe 2074, 128

import numpy as np
import re
from glob import glob
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import sys

def impute(filename, corruption_rate, start, end, partition_no):

    original = np.load(filename)
    new = np.zeros_like(original, dtype=np.float32)
    corruption_length = int(corruption_rate * original.shape[1])
    imputer = IterativeImputer(random_state=100, max_iter=10)
    new_filename = re.sub('.npy', '', filename)

    for i in range(start, end):
        original_copy = original.copy()
        row = original_copy[i, :]
        # Understand permutation behaviour
        row_indices_to_impute = np.random.permutation(original.shape[1])[:corruption_length]
        print(row_indices_to_impute)
        row[row_indices_to_impute] = np.nan
        imputer.fit(original_copy)
        imputed_row = imputer.transform(original_copy)[i]
        new[i, :] = imputed_row
        print(f"Done row {i}")

    np.save(f"{new_filename}_{corruption_rate}_corrupted_{partition_no}.npy", new)

if __name__ == "__main__":

    if len(sys.argv) != 6:
        print(sys.argv)
        print("Did not input arguments")
        sys.exit(1)

    impute(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
