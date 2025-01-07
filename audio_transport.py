import numpy as np
import matplotlib.pyplot as plt
from spectral import point

class spectral_mass:
    def __init__(self, left_bin=0, right_bin=0, center_bin=0, mass=0):
        self.left_bin = left_bin
        self.right_bin = right_bin
        self.center_bin = center_bin
        self.mass = mass

def interpolate(left, right, phases, window_size, interpolation_factor):
    # Group the left and right spectra
    left_masses = group_spectrum(left)
    right_masses = group_spectrum(right)

    # Get the transport matrix
    T = transport_matrix(left_masses, right_masses)
    if False:
        plot_transport_matrix(T, left_masses, right_masses)

    # Initialize the output spectral masses
    interpolated = [point(freq=p.freq) for p in left]

    # Initialize new phases
    new_amplitudes = [0] * len(phases)
    new_phases = [0] * len(phases)

    # Perform the interpolation
    for t in T:
        left_mass = left_masses[t[0]]
        right_mass = right_masses[t[1]]

        # Calculate the new bin and frequency
        interpolated_bin = round((1 - interpolation_factor) * left_mass.center_bin + interpolation_factor * right_mass.center_bin)

        # Compute the actual interpolation factor given the new bin
        interpolation_rounded = interpolation_factor
        if left_mass.center_bin != right_mass.center_bin:
            interpolation_rounded = (interpolated_bin - left_mass.center_bin) / (right_mass.center_bin - left_mass.center_bin)

        # Interpolate the frequency appropriately
        interpolated_freq = (1 - interpolation_rounded) * left[left_mass.center_bin].freq_reassigned + interpolation_rounded * right[right_mass.center_bin].freq_reassigned
        center_phase = phases[interpolated_bin] + (interpolated_freq * window_size / 2) / 2 - (np.pi * interpolated_bin)
        new_phase = center_phase + (interpolated_freq * window_size / 2) / 2 + (np.pi * interpolated_bin)

        # Place the left and right masses
        place_mass(left_mass, interpolated_bin, (1 - interpolation_factor) * t[2] / left_mass.mass, interpolated_freq, center_phase, left, interpolated, new_phase, new_phases, new_amplitudes)
        place_mass(right_mass, interpolated_bin, interpolation_factor * t[2] / right_mass.mass, interpolated_freq, center_phase, right, interpolated, new_phase, new_phases, new_amplitudes)

    # Fill the phases with the new phases
    for i in range(len(phases)):
        phases[i] = new_phases[i]

    return interpolated, phases



def place_mass(mass, center_bin, scale, interpolated_freq, center_phase, input, output, next_phase, phases, amplitudes):
    # Compute how the phase changes in each bin
    phase_shift = center_phase - np.angle(input[mass.center_bin].value)
    for i in range(mass.left_bin, mass.right_bin):
        # Compute the location in the new array
        new_i = i + center_bin - mass.center_bin
        if new_i < 0 or new_i >= len(output):
            continue
        # Rotate the output by the phase offset plus the frequency
        phase = phase_shift + np.angle(input[i].value)
        mag = scale * abs(input[i].value)
        output[new_i].value += mag * np.exp(1j * phase)
        if mag > amplitudes[new_i]:
            amplitudes[new_i] = mag
            phases[new_i] = next_phase
            output[new_i].freq_reassigned = interpolated_freq

def transport_matrix(left, right):
    # Initialize the algorithm
    T = []
    left_index = 0
    right_index = 0
    left_mass = left[0].mass
    right_mass = right[0].mass
    while True:
        if left_mass < right_mass:
            T.append((left_index, right_index, left_mass))
            right_mass -= left_mass
            left_index += 1
            if left_index >= len(left):
                break
            left_mass = left[left_index].mass
        else:
            T.append((left_index, right_index, right_mass))
            left_mass -= right_mass
            right_index += 1
            if right_index >= len(right):
                break
            right_mass = right[right_index].mass
    return T

def plot_transport_matrix(T, left, right):
    # Get the transport matrix

    # Create a 2D matrix for plotting
    transport_matrix = np.zeros((len(left), len(right)))

    # Fill the transport matrix with mass values
    for left_idx, right_idx, mass in T:
        transport_matrix[left_idx, right_idx] = mass

    # Plotting the transport matrix as a heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(transport_matrix, cmap='Blues', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Mass Transported')
    plt.title('Transport Matrix')
    plt.xlabel('Right Index')
    plt.ylabel('Left Index')
    plt.show()

def group_spectrum(spectrum):
    # Keep track of the total mass
    mass_sum = sum(abs(p.value) for p in spectrum)
    # Initialize the first mass
    masses = []
    initial_mass = spectral_mass(left_bin=0, center_bin=0)
    masses.append(initial_mass)
    sign = None
    first = True
    for i in range(len(spectrum)):
        current_sign = spectrum[i].freq_reassigned > spectrum[i].freq
        if first:
            first = False
            sign = current_sign
            continue
        if current_sign == sign:
            continue
        if sign:
            # We are falling
            left_dist = spectrum[i - 1].freq_reassigned - spectrum[i - 1].freq
            right_dist = spectrum[i].freq - spectrum[i].freq_reassigned
            if left_dist < right_dist:
                masses[-1].center_bin = i - 1
            else:
                masses[-1].center_bin = i
        else:
            # We are rising
            masses[-1].mass = sum(abs(spectrum[j].value) for j in range(masses[-1].left_bin, i))
            if masses[-1].mass > 0:
                masses[-1].mass /= mass_sum
                masses[-1].right_bin = i
                new_mass = spectral_mass(left_bin=i, center_bin=i)
                masses.append(new_mass)
        sign = current_sign
    # Finish the last mass
    masses[-1].right_bin = len(spectrum)
    masses[-1].mass = sum(abs(spectrum[j].value) for j in range(masses[-1].left_bin, len(spectrum)))
    masses[-1].mass /= mass_sum
    return masses
