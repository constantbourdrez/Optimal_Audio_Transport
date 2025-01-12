import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import soundfile as sf
import librosa
import cvxpy as cp
import ot
import torch

import utils as plot_utils
from generate_audio import normalize

WIN_LENGTH = 2206
HOP = int(WIN_LENGTH / 2)
N_FFT = 4096
SR = 44100

def stft(wave):
    s = librosa.stft(wave, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length= HOP)
    S = np.abs(s)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    frequencies = librosa.fft_frequencies(sr= SR, n_fft= N_FFT)
    return s, S_db, frequencies


def distmat(x,y):
    return  np.abs(x[:,None]-y[None,:])**2

def compute_optimal_map_lp(a, b, n, m, x, y):
    """
    Computes the optimal mapping using a linear program solver (cvxpy)
    """

    C = distmat(x,y)
    P = cp.Variable((n,m))
    u = np.ones((m,1))
    v = np.ones((n,1))
    U = [0 <= P, cp.matmul(P, u) == a.reshape(-1, 1),  cp.matmul(P.T, v) == b.reshape(-1, 1)]


    objective = cp.Minimize( cp.sum(cp.multiply(P,C)) )
    prob = cp.Problem(objective, U)
    result = prob.solve(verbose= False, solver=cp.OSQP)

    return P.value

def compute_optimal_map(x, y):
    """
    Computes 1D optimal transport map using north-west corner rule heuristic
    Runs in O(2*|n|) time
    """
    n = len(x)
    if len(x) != len(y):
        print('dimensions are not equal')
        return []

    pi = np.zeros((n, n))

    # Compute initial mass containers
    px = np.abs(x[0])
    py = np.abs(y[0])
    (i, j) = (0, 0)
    while True:
        # If there is less mass in container px,
        # transfer as much as possible from py
        if (i >= n) and (j >= n):
            break
        if px < py:
            if (i < n ):
                pi[i, j] = px
                py = py - px
                i = i + 1
                if i >= n:
                    break
                # Refills x container with next mass
                px = np.abs(x[i])

        else:
            if (j < n ):
                pi[i, j] = py
                px = px - py
                j = j + 1
                if j >= n:
                    break
                # Refills x container with next mass
                py = np.abs(y[j])

    return pi

def compute_optimal_map_unbalanced(a, b, n, m, x, y, rho):
    """
    Computes the optimal mapping using a linear program solver (cvxpy)
    """

    C = distmat(x,y)
    P = cp.Variable((n,m))
    u = np.ones((m,1))
    v = np.ones((n,1))
    q = cp.sum( cp.kl_div(cp.matmul(P,v),a[:,None]) )
    r = cp.sum( cp.kl_div(cp.matmul(P.T,u),b[:,None]) )
    constr = [0 <= P]
    objective = cp.Minimize( cp.sum(cp.multiply(P,C)) + rho*q + rho*r )

    prob = cp.Problem(objective, constr)
    result = prob.solve()
    return P.value

def smooth_transport(p, smoothing_factor= 20):
    """
    Smooth the transport plan using a simple moving average or convolution.

    Args:
        p (np.ndarray): Transport plan (2D array)
        smoothing_factor (int): The size of the smoothing window.

    Returns:
        np.ndarray: Smoothed transport plan
    """
    return np.convolve(p.flatten(), np.ones(smoothing_factor) / smoothing_factor, mode='same').reshape(p.shape)

def interpolate(p, t, mass1=None, mass2=None, smoothing_factor=0.01):
    """
    Computes spectral interpolation between two distributions with a smoother mass transfer.

    Inputs:
    - p: Optimal transport plan (2D array)
    - t: Interpolation factor (float, between 0 and 1)
    - mass1: Total mass of input base measure (if None, interpolation won't take mass into account)
    - mass2: Total mass of input target measure
    - smoothing_factor: Factor for smoothing the transport map

    Returns:
    - interp: Interpolated measure (1D vector)
    """
    n = p.shape[0]
    interp = np.zeros(n)

    # Finding non-zero entries of the map
    I,J = np.nonzero(p>1e-5)

    # Computes the displaced frequency
    # Also rounds it to the newest integer
    k = (1 - t) * I +  t *(J)
    k_floor = np.floor(k).astype(int)  # Round down
    k_ceil = np.ceil(k).astype(int)  # Round up
    v = k - k_floor  # diff between displaced frequency and closest lower original frequency

    # Iterates over non-zero entries of the transport map
    for (i, j, l) in zip(I, J, np.arange(0, len(I))):

        if (k_ceil[l] < n-1):
            # Transfers mass proportionally to nearest frequencies
            # to the right and left of displaced frequency
            interp[k_floor[l]] = interp[k_floor[l]] + p[i, j] * (1 - v[l])
            interp[k_ceil[l]] = interp[k_ceil[l]] + p[i, j] * v[l]

        elif (k_floor[l] == 0) or (k_ceil[l] == n-1):
            interp[k_floor[l]] = interp[k_floor[l]] + p[i, j]
        elif k_ceil[l] == n:
            interp[k_ceil[l] - 1] = interp[k_ceil[l] - 1] + p[i, j]

    # Uses mass information to get proportional mapping
    if mass1 != None:
        interp = interp * mass1 * (1 - t) + interp * mass2 * (t)
    return interp


def join_stfts(s1, s2, n_windows, algo, rho, verbose=True, correct_phase='repeat'):
    """
    Joins STFTs s1 and s2 while interpolating in the middle.

    Inputs:
    s1: complex spectrogram of input audio 1 (size (D, n1))
    s2: complex spectrogram of input audio 2 (size (D, n2))
    n_windows: number of frames to interpolate
    algo: algorithm to compute the optimal map ('original', 'lp', 'unbalanced')
    rho: parameter for unbalanced optimal transport (used if algo='unbalanced')
    verbose: if True, prints additional information
    correct_phase: phase correction method (not implemented in this code)

    Returns:
    new_spec: complex spectrogram (size D, (n1+n_windows+n2))
    """

    if s1.shape[0] != s2.shape[0]:
        raise ValueError("Different number of frequency bins in s1 and s2")

    new_spec = np.empty((s1.shape[0], s1.shape[1] + n_windows + s2.shape[1]), dtype=complex)

    # Fill in spectrum from first clip
    new_spec[:, :s1.shape[1]] = s1[:, :]

    # Fill in spectrum from second clip
    new_spec[:, s1.shape[1] + n_windows:] = s2[:, :]

    alpha = s1[:, -1]  # spectrum of last frame of s1
    beta = s2[:, 0]  # spectrum of first frame of s2

    # Total spectral mass of both spectra
    mass = lambda x: np.sum(np.abs(x))
    mass1 = mass(alpha)
    mass2 = mass(beta)

    # Normalize the spectra to form histograms
    norm_alpha = alpha / mass1
    norm_beta = beta / mass2


    print("Computing optimal mapping")
    if algo == "original":
        p = compute_optimal_map(norm_alpha, norm_beta)
    elif algo == "lp":
        # Determine the support of the spectra
        threshold = 1e-18  # Define a threshold for support
        support_alpha = np.where(np.abs(norm_alpha) > threshold)[0]
        support_beta = np.where(np.abs(norm_beta) > threshold)[0]

        # Restrict the computation to the support
        x_alpha = support_alpha / len(norm_alpha)
        x_beta = support_beta / len(norm_beta)
        norm_alpha_support = norm_alpha[support_alpha]
        norm_beta_support = norm_beta[support_beta]
        # Restrict `a` and `b` to their supports
        reduced_a = np.abs(norm_alpha_support)  # Ensure non-negative values
        reduced_b = np.abs(norm_beta_support)
        p = compute_optimal_map_lp(
            reduced_a, reduced_b,
            len(reduced_a), len(reduced_b),  # Update n and m
            x_alpha, x_beta
        )
        plt.imshow(p)
    elif algo == "unbalanced_slow":
        # Determine the support of the spectra
        threshold = 0 # Define a threshold for support
        support_alpha = np.where(np.abs(norm_alpha) > threshold)[0]
        support_beta = np.where(np.abs(norm_beta) > threshold)[0]

        # Restrict the computation to the support
        x_alpha = support_alpha / len(norm_alpha)
        x_beta = support_beta / len(norm_beta)
        norm_alpha_support = norm_alpha[support_alpha]
        norm_beta_support = norm_beta[support_beta]
        reduced_a = np.abs(norm_alpha_support)
        reduced_b = np.abs(norm_beta_support)
        p = compute_optimal_map_unbalanced(
            reduced_a, reduced_b,
            len(reduced_a), len(reduced_b),
            x_alpha, x_beta, rho
        )
    elif algo == "unbalanced":
        threshold = 0 # Define a threshold for support
        support_alpha = np.where(np.abs(norm_alpha) > threshold)[0]
        support_beta = np.where(np.abs(norm_beta) > threshold)[0]
        # Restrict the computation to the support
        x_alpha = support_alpha / len(alpha)
        x_beta = support_beta / len(beta)
        norm_alpha_support = norm_alpha[support_alpha]
        norm_beta_support = norm_beta[support_beta]
        cost_matrix = torch.tensor(distmat(x_alpha, x_beta), dtype= torch.float32).to(torch.device('mps'))
        a = torch.tensor(norm_alpha, dtype= torch.float32).to(torch.device('mps'))
        b = torch.tensor(norm_beta, dtype= torch.float32).to(torch.device('mps'))
        p = ot.unbalanced.lbfgsb_unbalanced(a, b, cost_matrix.to(torch.device("mps")), reg = rho, reg_m = 1).cpu().numpy()
    else:
        raise ValueError("Invalid algorithm")
    print("Optimal mapping computed")

    if verbose:
        print("Mass of signal 1:", mass1)
        print("Mass of signal 2:", mass2)

    # Computes displacement interpolation
    ts = np.linspace(0, 1, n_windows)
    phi_prev = np.angle(alpha)
    for t, i in zip(ts, range(0, n_windows)):
        # Interpolated spectrum magnitude
        interp_abs = np.abs(interpolate(p, t, mass1=mass1, mass2=mass2))
        #np.angle(new_spec[:, s1.shape[1] + i - 1]) +
        if correct_phase == 'repeat':
            interp_phase = phi_prev
        elif correct_phase == 'zero' or correct_phase == None:
            interp_phase = 0
        elif correct_phase == 'vocoder':
            # Phase computation
            freqs = SR / N_FFT / 2 * np.arange(0, len(alpha))
            #print(phi_prev)
            phi = 2 * np.pi * freqs * WIN_LENGTH / SR  +  phi_prev
            #phi = phi_prev % 2 * np.pi
            phi_prev = phi
            interp_phase = interp_abs * np.sin(phi)

        # Fills in interpolated spectrum
        new_spec[:, s1.shape[1] + i] = interp_abs + interp_phase * 1j


    #print(np.angle(new_spec[:, s1.shape[1] -1][:5]))
    #print(s1.shape)
    #print(np.angle(new_spec[:, s1.shape[1]][:5]))
    #plt.plot( np.abs(alpha[:50]))
    #plt.plot( np.abs(norm_alpha[:50]) *mass1*2)
    return new_spec



def transport(x1, x2, t1, t2, t3, sr=44100, size_window=2206, correct_phase='repeat', plot=None, write_file=None, algo = 'original', rho = None):
    """
    Computes spectral interpolation between two audio signals using 1D optimal transport
    Spectral interpolation is computed using displacement interpolation.

    Inputs:
    x1: raw audio signal 1
    x2: raw audio signal 1
    t1: number of seconds x1 will play
    t2: number of seconds x1 will play
    t3: number of seconds used to do interpolation
    sr: sampling rate (Hz)
    size_window: FFT window (frame) length
    write_file: complete name of resulting audio file (ex. text.wav)
    """
    # Number of frames needed to play requested times
    size_window = int(size_window / 2)
    n_windows1 = int(t1 * sr / size_window)
    n_windows2 = int(t2 * sr / size_window)
    n_windows3 = int(t3 * sr / size_window)

    # Compute stfts of input signals
    s1, S_db1, freq1 = stft(x1)
    s2, S_db2, freq2 = stft(x2)

    # Exclude some frames to avoid boundary effects
    begin = 10
    end = 10

    new_D = join_stfts(s1[:, :n_windows1 - end], s2[:, begin:n_windows2],n_windows3, correct_phase=correct_phase, algo=algo, rho=rho)

    if plot==1:
        plot_utils.plot_spectogram(s1, figsize=(12,8))
    elif plot==2:
        plot_utils.plot_spectogram(s2, figsize=(12,8))
    elif plot==3:
        plot_utils.plot_spectogram(new_D, figsize=(12,8))

    # Computes ifft
    I = librosa.istft(new_D, win_length=WIN_LENGTH, hop_length= HOP)

    if write_file != None:
        sf.write(write_file, I, SR)
    return
