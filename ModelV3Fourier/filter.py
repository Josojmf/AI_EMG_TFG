import pandas as pd
import matplotlib.pyplot as plt
import subprocess

# Function to run the VHDL simulation
def run_vhdl_filter(input_signal, vhdl_executable, output_csv):
    """
    Sends an input signal to a VHDL filter simulation and retrieves the filtered output.
    :param input_signal: Path to the input CSV file.
    :param vhdl_executable: Path to the VHDL simulation executable.
    :param output_csv: Path where the filtered output CSV will be saved.
    """
    # Call the VHDL filter simulation (ensure your VHDL environment supports this)
    process = subprocess.run([vhdl_executable, input_signal, output_csv], check=True)
    if process.returncode == 0:
        print(f"VHDL Filter successfully executed. Filtered output saved to {output_csv}")
    else:
        print("Error in VHDL filter execution!")

# Load the unfiltered signal from CSV
input_csv = 'path_to_your_unfiltered_signal.csv'  # Replace with your actual file path
data = pd.read_csv(input_csv)

# Assuming the first column contains the unfiltered signal
unfiltered_signal = data.iloc[:, 0]

# Path to the VHDL simulation executable
vhdl_executable = 'path_to_vhdl_filter_executable'  # Replace with the VHDL simulation executable path
filtered_csv = 'filtered_output.csv'  # Output file path for filtered signal

# Run the VHDL filter simulation
run_vhdl_filter(input_csv, vhdl_executable, filtered_csv)

# Load the filtered signal
filtered_data = pd.read_csv(filtered_csv)
filtered_signal = filtered_data.iloc[:, 0]

# Plot unfiltered and filtered signals
plt.figure(figsize=(12, 6))

# Plot unfiltered signal
plt.subplot(2, 1, 1)
plt.plot(unfiltered_signal, label="Unfiltered Signal", color="blue")
plt.title("Unfiltered Signal")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Plot filtered signal
plt.subplot(2, 1, 2)
plt.plot(filtered_signal, label="Filtered Signal", color="green")
plt.title("Filtered Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Show and save the plot
plt.tight_layout()
plt.savefig("comparison_plot.png")
plt.show()
