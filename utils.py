import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from audio_transport import interpolate
from generate_audio import normalize

def plot_spectogram(data, figsize=(8,6)):
    fig, ax = plt.subplots(figsize=figsize)
    S = np.abs(data)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db,y_axis='log', x_axis='time', ax=ax, sr=44100)

    ax.set_title('Power spectrogram')

    fig.colorbar(img, ax=ax, format="%+2.0f dB")


def plot_progression(abs, y1, y2, interp='trivial', mapping=None, figsize=(10, 6), size=6):
    """
    Plots the interpolated progression given an interpolation type.
    Interpolates between input signals
    y1, y2: Input 1d vectors (size n)
    interp: two possible types, 'trivial' and 'displacement'. In the second kind
        a mapping matrix (nxn) is needed
    size: number of intermediate interpolated signals
    """
    ts = np.linspace(0, 1, size)
    fig, ax = plt.subplots(1, size, figsize=figsize)
    if interp == 'displacement':
        assert mapping is not None
        I,J = np.nonzero(mapping>1e-7)
        Pij = mapping[I,J]
    #y1, y2 = normalize(y1), normalize(y2)
    if interp == 'displacement':
        #interps = displacement_interpolation(y1, y2, ts, mapping)
        pass
    for t, i in zip(ts, range(size)):
        # Plot the original signals with dashed lines
        ax[i].plot(abs, y1, ls='--', alpha=0.5, color='black', label='y1 (original)')
        ax[i].plot(abs, y2, ls='--', alpha=0.5, color='dodgerblue', label='y2 (target)')


        ax[i].set_yticks([])  # Keep y-axis ticks on the first plot

        ax[i].set_xticks([])  # Remove x-axis ticks
        # Add gridlines
        ax[i].grid(True, linestyle='--', alpha=0.6)

        # Interpolate signals based on chosen method
        if interp == 'trivial':
            inte = t * y2 + (1 - t) * y1
            x_axis = abs
        elif interp == 'displacement':
            #inte = interpolate(mapping, t)
            inte = (1-t)*y1[I] + t*y2[J]
            x_axis = (1-t)*abs[I] + t*abs[J]

        # Plot the interpolated signal with a prominent color
        ax[i].plot(x_axis, inte, color='darkcyan', linewidth=2)

        # Set the title to show the interpolation factor 't'
        ax[i].set_title(f"t = {np.round(t, 1)}", fontsize=12)

        # Add a legend for the first plot
        if i == 0:
            ax[i].legend(loc='upper right', fontsize=10, frameon=False)

        # Customize tick labels and other properties
        ax[i].tick_params(axis='both', which='major', labelsize=8)

    # Adjust the layout to avoid overlapping
    fig.tight_layout()
    plt.show()
