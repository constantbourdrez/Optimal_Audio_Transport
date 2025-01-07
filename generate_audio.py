import numpy as np
import soundfile as sf

def generate_sine_wave(frequency, duration, sample_rate, amplitude=0.7):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)

def main():
    sample_rate = 4000  # Sample rate in Hz
    duration = 1  # Duration in seconds

    # Generate two sine waves with different frequencies
    signal1 = generate_sine_wave(440, duration, sample_rate)  # 440 Hz sine wave
    signal2 = generate_sine_wave(880, duration, sample_rate)  # 880 Hz sine wave

    # Save the generated signals as WAV files
    sf.write('input_file1.wav', signal1, sample_rate)
    sf.write('input_file2.wav', signal2, sample_rate)

if __name__ == "__main__":
    main()
