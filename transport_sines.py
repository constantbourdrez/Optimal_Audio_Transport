import argparse
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from spectral import analysis as spectral_analysis, synthesis as spectral_synthesis
from spectral import  plot_waveform, plot_fft, plot_spectrogram
from audio_transport import interpolate
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from generate_audio import generate_sine_wave

def plot_fft_at_window(interpolated_signal, sample_rate, window_size, w, num_windows):
    N = round(window_size * sample_rate)  # Window size in samples
    fft_output = [p.value for p in interpolated_signal]
    freqs = [p.freq_reassigned for p in interpolated_signal]

    # Plot FFT of the interpolated signal
    plt.figure(figsize=(10, 6))
    plt.plot(freqs[:len(freqs)//2], np.abs(fft_output)[:len(fft_output)//2], label=f"Window {w}/{num_windows}")
    plt.title(f"FFT of Interpolated Signal (Window {w}/{num_windows})")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, 4000)
    plt.grid(True)
    plt.legend()

    # Save plot as an image for gif creation
    plt.savefig(f"fft_window_{w}.png")
    plt.close()

def create_fft_gif(num_windows, interpolated_signal, sample_rate, window_size):
    # Create frames for the gif by plotting the FFT at each window
    for w in range(num_windows):
        plot_fft_at_window(interpolated_signal[w], sample_rate, window_size, w, num_windows)

    # Create the GIF using the saved frames
    fig, ax = plt.subplots(figsize=(10, 6))
    images = []

    for w in range(num_windows):
        img = plt.imread(f"fft_window_{w}.png")
        images.append([ax.imshow(img)])

    ani = animation.ArtistAnimation(fig, images, interval=500, blit=True)
    ani.save('fft_interpolation.gif', writer=PillowWriter(fps=6))  # Save the gif
    print("GIF created successfully!")



def write_audio(filename, data, sample_rate):
    # Write audio data to a file using soundfile
    sf.write(filename, data.T, sample_rate)  # .T is to transpose if stereo



def main(output_file, plot):
    sample_rate = 10000 # samples per second
    total_time = 5  # seconds
    window_size = 0.05  # seconds
    padding = 7  # multiplies window size

    audio_left = generate_sine_wave(1000, total_time, sample_rate)
    audio_right = generate_sine_wave(20, total_time, sample_rate)

    # Plot the waveform of both channels
    if plot:
        plot_waveform(audio_left, sample_rate, total_time, title="Left Channel Waveform")
        plot_waveform(audio_right, sample_rate, total_time, title="Right Channel Waveform")

        # Plot the spectrogram of both channels
        plot_fft(audio_left, sample_rate, title="Left Channel FFT")
        plot_fft(audio_right, sample_rate, title="Right Channel FFT")

    # Perform spectral analysis
    print("Converting left input to the spectral domain")
    points_left = spectral_analysis(audio_left, sample_rate, window_size, padding)

    value = [p.value for p in points_left[0]]
    freq = [p.freq for p in points_left[0]]
    time = [p.time for p in points_left[0]]
    time_reassigned = [p.time_reassigned for p in points_left[0]]
    freq_reassigned = [p.freq_reassigned for p in points_left[0]]
    plt.figure(figsize=(10, 6))
    plt.plot(freq, value, label="Original")
    plt.plot(freq_reassigned, value, label="Reassigned")
    plt.title("Spectral Analysis")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()


    print("Converting right input to the spectral domain")
    points_right = spectral_analysis(audio_right, sample_rate, window_size, padding)

    # Initialize phases
    phases = [0] * len(points_left[0])  # Example, you may want to compute phases

    # Interpolate between left and right spectral data
    num_windows = min(len(points_left), len(points_right))
    points_interpolated = []

    print("Performing optimal transport-based interpolation")
    for w in range(num_windows):
        interpolation_factor = w / float(num_windows)
        interpolation_factor = interpolation_factor * 2 - 0.5
        interpolation_factor = min(1., max(0., interpolation_factor))

        # Call interpolate function
        interpolated_points, phases = interpolate(points_left[w], points_right[w], phases, window_size, interpolation_factor)
        points_interpolated.append(interpolated_points)

    # Optionally, track the FFT evolution here
    create_fft_gif(num_windows, points_interpolated, sample_rate, window_size)

    # Synthesize the interpolated audio back to time domain
    print("Converting the interpolation to the time domain")
    audio_interpolated = spectral_synthesis(points_interpolated, padding)
    # audio_interpolated = audio_interpolated / np.max(np.abs(audio_interpolated))  # Normalize the audio
    # print(audio_interpolated.max())

    # Plot the interpolated waveform and spectrogram
    if plot:
        duration = len(audio_interpolated) / sample_rate
        plot_waveform(audio_interpolated, sample_rate, duration, title="Interpolated Audio Waveform")
        plot_spectrogram(audio_interpolated, sample_rate)

    # Write the file
    print(f"Writing to file {output_file}")
    write_audio(output_file, audio_interpolated, sample_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a stereo audio file with a sine wave that interpolates between two different frequencies")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the interpolated waveform and spectrogram")
    parser.add_argument("output_file", help="Output audio file")  # No default, it's positional
    args = parser.parse_args()
    main(args.output_file, args.plot)
