import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

class point:
    def __init__(self, value=0, time=0, freq=0, time_reassigned=0, freq_reassigned=0):
        self.value = value
        self.time = time
        self.freq = freq
        self.time_reassigned = time_reassigned
        self.freq_reassigned = freq_reassigned

def synthesis(points, padding=0, overlap=1):
    # Initialize the window
    window_padded_size = 2 * (len(points[0]) - 1)
    window_size = window_padded_size // (1 + padding)
    padding_samples = (window_padded_size - window_size) // 2
    # Initialize the audio
    hop_size = window_size // (2 * overlap)
    num_hops = len(points) + 2 * overlap - 1
    audio = np.zeros(num_hops * hop_size)
    # Iterate over the windows
    for w in range(len(points)):
        # Fill the FFT
        fft_input = np.array([p.value for p in points[w]])
        # Execute the inverse FFT
        fft_output = ifftshift(ifft(fftshift(fft_input)))
        padding_samples = min(padding_samples, len(fft_output) // 2)
        # Apply the weighted overlap add
        for i in range(window_size):
            idx = i + padding_samples
            if 0 <= idx < len(fft_output):
                value = fft_output[i + padding_samples].real / (overlap * len(fft_output))
                audio[i + w * hop_size] += value
                hop_size = window_size // (2 * overlap)
    return audio

def analysis(audio, sample_rate, window_size=0.05, padding=0, overlap=1):
    # Make sure inputs are positive
    assert sample_rate > 0
    assert window_size > 0
    # Convert the window size to samples
    N = round(window_size * sample_rate)
    while N % (2 * overlap) != 0:
        N += 1
    N_padded = N * (1 + padding)
    # Initialize the windows
    window = np.zeros(N_padded)
    window_t = np.zeros(N_padded)
    window_d = np.zeros(N_padded)
    # Determine samples used for padding
    padding_samples = (N_padded - N) // 2
    # Compute the number of windows
    num_hops = len(audio) // (N // (2 * overlap))
    num_windows = num_hops - (2 * overlap - 1)
    # Initialize the spectral points
    points = [[] for _ in range(num_windows)]

    # Iterate over the windows
    for w in range(num_windows):
        for i in range(N):
            n = i - (N - 1) / 2
            a = audio[i + w * N // (2 * overlap)]
            window[i + padding_samples] = a * hann(n, N)
            window_t[i + padding_samples] = a * hann_t(n, N, sample_rate)
            window_d[i + padding_samples] = a * hann_d(n, N, sample_rate)
        # Execute the FFT
        fft_output = fftshift(fft(fftshift(window)))
        fft_t_output = fftshift(fft(fftshift(window_t)))
        fft_d_output = fftshift(fft(fftshift(window_d)))
        # Compute the center time
        t = ((N - 1) / 2 + w * N / (2 * overlap)) / sample_rate
        for i in range(N_padded // 2 + 1):
            X = fft_output[i]
            X_t = fft_t_output[i]
            X_d = fft_d_output[i]
            p = point(X, t, (2 * np.pi * i * sample_rate) / N_padded)
            conj_over_norm = np.conj(X) / np.abs(X)
            dphase_domega = np.real(X_t * conj_over_norm)
            dphase_dt = -np.imag(X_d * conj_over_norm)
            p.time_reassigned = p.time + dphase_domega
            p.freq_reassigned = p.freq + dphase_dt
            points[w].append(p)
    return points

def hann(n, N):
    """Hann window."""
    return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))

# Derivative of the Hann window with respect to time
def hann_t(n, N, sample_rate):
    """Time-scaled Hann window."""
    return (n / sample_rate) * hann(n, N)

# Derivative of the Hann window with respect to frequency
def hann_d(n, N, sample_rate):
    """Frequency-scaled derivative of the Hann window."""
    return (np.pi * sample_rate) / (N - 1) * np.sin(2 * np.pi * n / (N - 1))

def plot_spectrogram(audio, sample_rate, title="Spectrogram"):
    # Plot the spectrogram of the audio signal
    f, t, Sxx = spectrogram(audio, sample_rate)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto')
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Power [dB]")
    plt.grid(True)
    plt.show()

def plot_fft(sine_wave, sample_rate, title="FFT"):
    N = len(sine_wave)
    T = 1.0 / sample_rate
    yf = np.fft.fft(sine_wave)
    xf = np.fft.fftfreq(N, T)

    # Plot the FFT
    plt.figure(figsize=(10, 6))
    plt.plot(xf, 2.0/N * np.abs(yf))
    plt.title('FFT of Sine Wave')
    plt.xlabel('Frequency (Hz)')
    plt.xlim(0)
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def plot_waveform(audio, sample_rate, duration, title="Waveform", xlabel="Time [s]", ylabel="Amplitude"):
    # Plot the waveform of the audio signal
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    plt.figure(figsize=(10, 4))
    plt.plot(t, audio)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.xlim(0,0.05)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
