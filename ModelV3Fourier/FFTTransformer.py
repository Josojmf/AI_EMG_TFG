import numpy as np
from scipy.signal import butter, filtfilt, welch


class FFTTransformer:
    def __init__(self, fs=1000, lowcut=5, highcut=450, order=4, window_size=200):
        self.fs = fs
        self.lowcut = lowcut  # Ensures we capture subtle variations
        self.highcut = highcut
        self.order = order
        self.window_size = window_size
        self.stride = int(self.window_size * 0.15)  # 🔹 Increased resolution on bursts

    def estimate_lowcut(self, data):
        """Dynamically estimates a lowcut frequency based on high-frequency power balance."""
        f, Pxx = welch(data, fs=self.fs, nperseg=256)

        # 🔹 Emphasizing high-frequency components more
        spectral_entropy = -np.sum((Pxx / np.sum(Pxx)) * np.log2(Pxx / np.sum(Pxx)))
        high_freq_energy = np.sum(Pxx[f > 200]) / np.sum(Pxx)

        estimated_lowcut = max(30, min(120, 250 - (spectral_entropy * 60) - (high_freq_energy * 50)))
        return estimated_lowcut

    def butter_bandpass_filter(self, data):
        """Applies a Butterworth bandpass filter with optimized frequency selection."""
        estimated_lowcut = self.estimate_lowcut(data)
        self.lowcut = max(5, min(estimated_lowcut, 180))  

        nyq = 0.5 * self.fs  # Nyquist frequency
        low = max(0.01, self.lowcut / nyq)  
        high = min(0.98, self.highcut / nyq)  

        if low >= high:
            high = min(0.99, high * 1.05)  # 🔹 Less aggressive correction

        print(f"[FILTER] Adjusted Bandpass: {low*nyq:.2f}Hz - {high*nyq:.2f}Hz")

        b, a = butter(self.order, [low, high], btype='band')
        filtered = filtfilt(b, a, np.array(data, dtype=np.float64))
        return filtered

    def apply_fft(self, data):
        """Extracts FFT features with enhanced normalization for better detection."""
        filtered_data = self.butter_bandpass_filter(data)
        fft_features = []

        for i in range(0, len(filtered_data) - self.window_size + 1, self.stride):
            segment = filtered_data[i:i + self.window_size]
            fft_values = np.abs(np.fft.rfft(segment))

            # 🔹 Log transformation emphasizes low-amplitude high-freq bursts
            fft_values = np.log1p(fft_values)

            # 🔹 Improved Standardization (Keeps relative magnitude)
            if np.std(fft_values) > 1e-8:
                fft_values = (fft_values - np.min(fft_values)) / (np.max(fft_values) - np.min(fft_values) + 1e-8)

            fft_features.append(fft_values)

        fft_array = np.array(fft_features)

        print(f"[FFT] Extracted Shape: {fft_array.shape}")
        print(f"[DEBUG] First FFT Feature Sample: {fft_array[0, :10]}")

        return fft_array
