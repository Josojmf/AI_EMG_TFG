import os
import pandas as pd

def add_zero_column_to_csv_files(folder_path, new_column_name="label"):

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return
    
    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):  # Process only CSV files
            file_path = os.path.join(folder_path, file_name)
            
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Add the new column with zeros
            df[new_column_name] = 0
            
            # Overwrite the original file with the modified DataFrame
            df.to_csv(file_path, index=False)
            
            print(f"File modified and saved: {file_path}")

def process_multiple_folders(base_folder, start=0, end=14, folder_prefix="S"):
    for i in range(start, end + 1):
        folder_name = f"{folder_prefix}{i}"
        folder_path = os.path.join(base_folder, folder_name)
        print(f"Processing folder: {folder_name}")
        
        add_zero_column_to_csv_files(folder_path)

# Example usage
base_folder = '.'  # Update this to the directory where folders S0, S1, ..., S14 are located
process_multiple_folders(base_folder, start=0, end=14)
