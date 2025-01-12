import numpy as np
import soundfile as sf


def generate_sine_wave(freqs, total_time, sample_rate):
    """
    Generates sine waves
    """
    if type(freqs) == list:
        time_values = np.arange(0, total_time, 1 / sample_rate )
        signal = np.sin(2 *np.pi * freqs[0] * time_values)
        for f in freqs[1:]:
            signal = signal + np.sin(2 *np.pi * f * time_values)
        return signal
    elif type(freqs) == int:
        time_values = np.arange(0, total_time, 1 / sample_rate )
        return np.sin(2 *np.pi * freqs * time_values)
    else:
        print('unsuported type')
        return []


def generate_gaussian(mu, sigma, N):
    """
    Generates gaussian on interval [0,1],
    centered on mu with std of sigma,
    composed of N points
    """
    t = np.arange(0,N)/N
    gauss = np.exp(-(t-mu)**2/(2*sigma**2))
    return gauss

def normalize(x, vmin=0.2):
    x = x + np.max(x) * vmin
    return x/np.sum(x)

def generate_sawtooth_wave(freqs, total_time, sample_rate):
    pass

def main():
    sample_rate = 4000  # Sample rate in Hz
    duration = 1  # Duration in seconds

    # Generate two sine waves with different frequencies
    signal1 = generate_sine_wave(440, duration, sample_rate)  # 440 Hz sine wave
    signal2 = generate_sine_wave(880, duration, sample_rate)  # 880 Hz sine wave

    # Save the generated signals as WAV files
    sf.write('sounds/input_file1.wav', signal1, sample_rate)
    sf.write('sounds/input_file2.wav', signal2, sample_rate)

if __name__ == "__main__":
    main()
