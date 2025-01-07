import sys
import numpy as np
import soundfile as sf
from spectral import analysis, synthesis
from equal_loudness import apply as equal_loudness_apply, remove as equal_loudness_remove
from audio_transport import interpolate

window_size = 0.05  # seconds
padding = 7  # multiplies window size

def main(input_file1, time_constant_ms, output_file):
    print("Starting the script...")

    time_constant = float(time_constant_ms) / 100.0
    if time_constant < 0:
        print("Time constant must be greater than zero.")
        return 1

    # Initial interpolation factor (doesn't change within the time domain)
    interpolation_factor = np.exp(-time_constant / window_size)
    print(f"Initial Interpolation Factor: {interpolation_factor}")

    # Open the audio file
    try:
        audio, sample_rate = sf.read(input_file1, always_2d=True)
    except Exception as e:
        print(f"Error reading audio files: {e}")
        return 1

    audio = audio.T  # Transpose to match the expected shape

    # Initialize the output audio
    num_channels = audio.shape[0]
    audio_interpolated = [[] for _ in range(num_channels)]

    # Iterate over the channels
    for c in range(num_channels):
        print(f"Processing channel {c}")

        print("Converting input to the spectral domain")
        points = analysis(audio[c], sample_rate, window_size, padding)
        print("Applying equal loudness filter")
        equal_loudness_apply(points)

        # Initialize previous points (starting from the first window)
        points_prev = points[0][:]

        # Initialize phases
        phases = [0] * len(points[0])

        print("Performing optimal transport based interpolation")
        num_windows = len(points)
        points_interpolated = [[] for _ in range(num_windows)]
        for w in range(num_windows):
            points_interpolated[w], phases = interpolate(
                points_prev,
                points[w],
                phases,
                window_size * sample_rate,
                interpolation_factor
            )

            # Copy the current interpolated points to be used in the next window
            points_prev = points_interpolated[w][:]

        print("Removing equal loudness filter")
        equal_loudness_remove(points)

        print("Converting the interpolation to the time domain")
        audio_interpolated[c] = synthesis(points_interpolated, padding)

    # Write the file
    audio_interpolated = np.array(audio_interpolated).T  # Transpose back to match soundfile's expected shape
    max_val = np.max(np.abs(audio_interpolated))
    if max_val > 0:
        audio_interpolated /= max_val

    print(f"Writing to file {output_file}")
    sf.write(output_file, audio_interpolated, sample_rate)
    print(f"File {output_file} written successfully.")

if __name__ == "__main__":
    # Hardcoded input files and parameters for simplicity
    input_file1 = 'input_file1.wav'
    time_constant_ms = 50  # Set time constant in milliseconds
    output_file = 'output_file.wav'
    main(input_file1, time_constant_ms, output_file)
