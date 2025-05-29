import numpy as np
from scipy.signal import butter, filtfilt, welch

class FFTTransformerGPU:
    def __init__(self, fs=1000, lowcut=5, highcut=450, order=4, window_size=200):
        self.fs = fs
        self.lowcut = lowcut  # Captura mejor las variaciones sutiles
        self.highcut = highcut
        self.order = order
        self.window_size = window_size
        self.stride = int(self.window_size * 0.15)  # 🔹 Mayor resolución en ráfagas

    def estimate_lowcut(self, data):
        """Estima dinámicamente la frecuencia de corte baja basada en el balance de energía de alta frecuencia."""
        f, Pxx = welch(data, fs=self.fs, nperseg=min(256, len(data)))  # 🔹 Garantiza `nperseg` válido

        # 🛠 **Solución al error de dimensiones**
        min_size = min(f.shape[0], Pxx.shape[0])
        f = f[:min_size]
        Pxx = Pxx[:min_size]

        # 🔹 Entropía espectral para ponderar alta frecuencia
        spectral_entropy = -np.sum((Pxx / np.sum(Pxx)) * np.log2(Pxx / np.sum(Pxx) + 1e-8))
        high_freq_energy = np.sum(Pxx[f > 200]) / (np.sum(Pxx) + 1e-8)  # 🔹 Evita división por cero

        estimated_lowcut = max(30, min(120, 250 - (spectral_entropy * 60) - (high_freq_energy * 50)))
        return estimated_lowcut

    def butter_bandpass_filter(self, data):
        """Aplica un filtro pasa banda Butterworth con selección de frecuencia optimizada."""
        estimated_lowcut = self.estimate_lowcut(data)
        self.lowcut = max(5, min(estimated_lowcut, 180))  

        nyq = 0.5 * self.fs  # Frecuencia de Nyquist
        low = max(0.01, self.lowcut / nyq)  
        high = min(0.98, self.highcut / nyq)  

        if low >= high:
            high = min(0.99, high * 1.05)  # 🔹 Corrección menos agresiva

        print(f"[FILTER] Ajustado Bandpass: {low*nyq:.2f}Hz - {high*nyq:.2f}Hz")

        b, a = butter(self.order, [low, high], btype='band')
        filtered = filtfilt(b, a, np.array(data, dtype=np.float64))
        return filtered

    def apply_fft(self, data):
        """Extrae características FFT con normalización mejorada para detección más precisa."""
        filtered_data = self.butter_bandpass_filter(data)
        fft_features = []

        for i in range(0, len(filtered_data) - self.window_size + 1, self.stride):
            segment = filtered_data[i:i + self.window_size]
            fft_values = np.abs(np.fft.rfft(segment))

            # 🔹 Transformación logarítmica enfatiza ráfagas de alta frecuencia
            fft_values = np.log1p(fft_values)

            # 🔹 Estandarización mejorada (conserva magnitudes relativas)
            if np.std(fft_values) > 1e-8:
                fft_values = (fft_values - np.min(fft_values)) / (np.max(fft_values) - np.min(fft_values) + 1e-8)

            fft_features.append(fft_values)

        fft_array = np.array(fft_features)

        print(f"[FFT] Shape Extraído: {fft_array.shape}")
        print(f"[DEBUG] Primera Muestra de FFT: {fft_array[0, :10]}")

        return fft_array