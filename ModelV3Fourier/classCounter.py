import os
import pandas as pd
import glob

DATA_DIR = "C:\\INFORMATICA\\TFG\\ModelV3Fourier\\Datasets\\Train_Data"

def count_samples(directory):
    """Counts the number of samples per class (0 = Normal, 1 = Spastic)."""
    class_counts = {0: 0, 1: 0}  # Dictionary to store counts

    files = glob.glob(os.path.join(directory, "*.csv"))
    
    for file in files:
        df = pd.read_csv(file)
        if "Label" in df.columns:
            counts = df["Label"].value_counts().to_dict()  # Count occurrences
            for label, count in counts.items():
                class_counts[label] = class_counts.get(label, 0) + count  # Accumulate total

    return class_counts

filtered_counts = count_samples(os.path.join(DATA_DIR, "filtered"))
original_counts = count_samples(os.path.join(DATA_DIR, "original"))

total_counts = {
    0: filtered_counts[0] + original_counts[0],
    1: filtered_counts[1] + original_counts[1]
}

print(" **Class Distribution in Training Data:**")
print(f" Normal Samples (0): {total_counts[0]}")
print(f" Spastic Samples (1): {total_counts[1]}")

#  Calculate percentage
total_samples = total_counts[0] + total_counts[1]
print(f"\n **Class Imbalance:**")
print(f" Normal: {total_counts[0] / total_samples:.2%}")
print(f" Spastic: {total_counts[1] / total_samples:.2%}")
