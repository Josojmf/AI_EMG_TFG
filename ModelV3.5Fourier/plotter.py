import pandas as pd
import matplotlib.pyplot as plt

def plot_signal(file_path):
    # Carga del dataset
    data = pd.read_csv(file_path)

    # Asegurar que la columna 'Time' es el índice
    data.set_index('Time', inplace=True)

   
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Signal'], label='Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.title('Signal over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

file_path = 'C:\\INFORMATICA\\TFG\\ModelV3\\Datasets\\Train_Data\\filtered\\ECG_filtered_augmented.csv'
plot_signal(file_path)
