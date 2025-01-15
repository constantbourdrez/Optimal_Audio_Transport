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

def generate_triangle_wave(freqs, total_time, sample_rate):
    """
    Generates a triangle wave for a given list of frequencies.

    Parameters:
    freqs (list): List of frequencies (in Hz) for the triangle wave.
    total_time (float): Total duration of the wave in seconds.
    sample_rate (int): The number of samples per second (sampling rate).

    Returns:
    numpy.ndarray: The generated triangle wave signal.
    """
    t = np.arange(0, total_time, 1 / sample_rate )
    wave = np.zeros_like(t)  # Initialize wave array

    # Generate the triangle wave for each frequency and add them together
    for freq in freqs:
        wave += 2 * np.abs(2 * ((t * freq) % 1) - 1) - 1  # Triangle wave formula

    # Normalize the wave to avoid clipping
    wave /= len(freqs)

    return wave



def generate_sawtooth_wave(freqs, total_time, sample_rate):
    """
    Generate a sawtooth wave for given frequencies.

    Parameters:
    - freqs (list or array): Frequencies of the sawtooth wave (one per channel).
    - total_time (float): Total duration of the wave in seconds.
    - sample_rate (int): Sampling rate in samples per second.

    Returns:
    - np.ndarray: A 2D array where each row is a sawtooth wave for a frequency in `freqs`.
    """
    time_values = np.arange(0, total_time, 1 / sample_rate )
    wave = 0
    for freq in freqs:
        wave += 2 * (time_values * freq - np.floor(time_values * freq + 0.5))  # Sawtooth wave formula


    return np.array([wave]).T

def generate_rectangular_wave(freqs, total_time, sample_rate):
    """
    Generate a rectangular wave for given frequencies.

    Parameters:
        freqs (list or np.ndarray): Frequencies in Hz for the rectangular wave.
        total_time (float): Total duration of the wave in seconds.
        sample_rate (float): Sampling rate in Hz.

    Returns:
        np.ndarray: Generated rectangular wave signal.
    """
    t = np.arange(0, total_time, 1 / sample_rate )
    wave = np.zeros_like(t)  # Initialize wave

    for freq in freqs:
        wave += np.sign(np.sin(2 * np.pi * freq * t))  # Add rectangular wave for each frequency

    # Normalize wave to range [-1, 1]
    wave = wave / np.max(np.abs(wave))

    return wave



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
