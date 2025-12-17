# -*- coding: utf-8 -*-
"""
Simulation of 2 speakers (each with 2 utterances) + noise
Using real WAV files instead of synthetic speech

Required WAV files in the same folder:
    male_11.wav
    male_12.wav
    female_21.wav
    female_22.wav
"""

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf
import librosa
import pyroomacoustics as pra
import matplotlib.pyplot as plt



# ===================================================
# GLOBAL PARAMETERS
# ===================================================

c = 340.0
fs = 16000
n_rir = 4096
order = 10
hp_filter = 1

R = 1.3
noise_R = 0.2
radius_mics = 0.1
high = 1.0
angle = 360
distance_from_wall = 0.5

SNR_diffuse = 20
SNR_direction = 20
SNR_mic = 30
SIR = 0

beta = 0.3   # reflection-like parameter


# ===================================================
# 1. LOAD WAV UTTERANCES
# ===================================================

def load_clean_wav(filename, target_fs=16000):
    """
    Loads a WAV file, converts to mono, resamples to target_fs,
    and normalizes to [-1,1].
    """
    y, _ = librosa.load(filename, sr=target_fs, mono=True)
    y = y.astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-12)
    return y


# ===================================================
# 2. GEOMETRY HELPERS
# ===================================================

def fill_line(p1, p2, n_points):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    t = np.linspace(0.0, 1.0, int(n_points))
    xy = p1[:, None] + (p2 - p1)[:, None] * t
    return xy[0, :], xy[1, :]


def generate_room_and_positions():
    """
    Creates room geometry, mic positions, speaker positions, and noise source.
    """
    # Random room size
    L1 = 4 + 0.1 * np.random.randint(1, 21)
    L2 = 4 + 0.1 * np.random.randint(1, 21)
    L = np.array([L1, L2, 3.0])

    room_x, room_y = L1, L2

    # Circle center
    distance_total = R + distance_from_wall + noise_R
    end_x = room_x - distance_total
    end_y = room_y - distance_total

    Radius_X = (end_x - distance_total) * np.random.rand() + distance_total
    Radius_Y = (end_y - distance_total) * np.random.rand() + distance_total

    # Angles for circle points
    R_angle = np.random.randint(1, angle + 1)
    t_full = np.linspace(-np.pi, 2 * np.pi, angle + angle // 2, endpoint=False)
    t = t_full[R_angle:R_angle + angle // 2]

    x = R * np.sin(t) + Radius_X
    y = R * np.cos(t) + Radius_Y

    # Mic circle (small)
    circ_mics_x = radius_mics * np.sin(t) + Radius_X
    circ_mics_y = radius_mics * np.cos(t) + Radius_Y

    # Choose 4 mic positions
    idxs = [0, 59, 119, 179]
    mic_positions = np.array([
        [circ_mics_x[i], circ_mics_y[i], high] for i in idxs
    ]).T  # shape (3,4)

    # Choose 2 speakers
    def pick_two():
        while True:
            i1 = np.random.randint(0, len(x))
            i2 = np.random.randint(0, len(x))
            x1, y1 = x[i1], y[i1]
            x2, y2 = x[i2], y[i2]

            # jitter
            w1 = 0.01 * np.random.randint(1, 315)
            w2 = 0.01 * np.random.randint(1, 315)
            x1j = x1 + noise_R * np.sin(w1)
            y1j = y1 + noise_R * np.cos(w1)
            x2j = x2 + noise_R * np.sin(w2)
            y2j = y2 + noise_R * np.cos(w2)

            if np.hypot(x1j - x2j, y1j - y2j) >= 0.5:
                return np.array([x1j, y1j, high]), np.array([x2j, y2j, high])

    s_first, s_second = pick_two()

    # Noise source far from center
    middle = np.array([Radius_X, Radius_Y, high])
    while True:
        xn = distance_from_wall + 0.01 * np.random.randint(1, 101) * (room_x - 2 * distance_from_wall)
        yn = distance_from_wall + 0.01 * np.random.randint(1, 101) * (room_y - 2 * distance_from_wall)
        s_noise = np.array([xn, yn, high])
        if np.linalg.norm(s_noise - middle) >= 2.0:
            break

    return L, mic_positions, s_first, s_second, s_noise


# ===================================================
# 3. COMPUTE RIRs
# ===================================================

def compute_rirs_pyroom(L, mic_positions, s_first, s_second, s_noise):
    absorption = 1.0 - beta

    room = pra.ShoeBox(
        L, fs=fs,
        absorption=absorption,
        max_order=order
    )

    # Add mics
    mic_array = pra.MicrophoneArray(mic_positions, fs)
    room.add_microphone_array(mic_array)

    # Add sources
    room.add_source(s_first)
    room.add_source(s_second)
    room.add_source(s_noise)

    room.compute_rir()

    M = mic_positions.shape[1]
    h_first = np.zeros((M, n_rir), np.float32)
    h_second = np.zeros((M, n_rir), np.float32)
    h_noise = np.zeros((M, n_rir), np.float32)

    for m in range(M):
        rir0 = room.rir[m][0]
        rir1 = room.rir[m][1]
        rir2 = room.rir[m][2]

        h_first[m, :min(n_rir, len(rir0))] = rir0[:n_rir]
        h_second[m, :min(n_rir, len(rir1))] = rir1[:n_rir]
        h_noise[m, :min(n_rir, len(rir2))] = rir2[:n_rir]

    return h_first, h_second, h_noise


# ===================================================
# 4. CONVOLVE HELPERS
# ===================================================

def convolve_signals_with_rir(signal, h):
    M, hlen = h.shape
    out = []
    for m in range(M):
        out.append(fftconvolve(signal, h[m], mode='full'))
    return np.stack(out, axis=0)  # (M, N)


# ===================================================
# 5. MAIN SIMULATION
# ===================================================

def main():
    # ================================================
    #      1. LOAD CLEAN SPEECH (4 WAV FILES)
    # ================================================
    speech_11 = load_clean_wav("male_11.wav")
    speech_12 = load_clean_wav("male_12.wav")
    speech_21 = load_clean_wav("female_21.wav")
    speech_22 = load_clean_wav("female_22.wav")

    print("Loaded WAVs:")
    print("male_11:", len(speech_11))
    print("male_12:", len(speech_12))
    print("female_21:", len(speech_21))
    print("female_22:", len(speech_22))


    # ================================================
    #      2. BUILD CLEAN SIGNALS in1 / in2
    # ================================================
    pad2 = np.zeros(2 * fs, np.float32)
    pad1 = np.zeros(fs, np.float32)

    zeros_21 = np.zeros_like(speech_21)
    zeros_11 = np.zeros_like(speech_11)

    in1 = np.concatenate([pad2, speech_11, pad1, zeros_21, pad1, speech_12])
    in2 = np.concatenate([pad2, zeros_11, pad1, speech_21, pad1, speech_22])

    print("\nConstructed in1/in2:")
    print("in1 length:", len(in1))
    print("in2 length:", len(in2))


    # ================================================
    #      3. GEOMETRY + RIR GENERATION
    # ================================================
    L, mic_positions, s_first, s_second, s_noise = generate_room_and_positions()
    print("\nRoom dimensions:", L)
    print("Mic positions:\n", mic_positions)
    print("Speaker 1 position:", s_first)
    print("Speaker 2 position:", s_second)
    print("Noise source position:", s_noise)

    h_first, h_second, h_noise = compute_rirs_pyroom(L, mic_positions, s_first, s_second, s_noise)

    # Plot example of RIR
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(h_first[0])
    plt.title("RIR from Speaker 1 → Mic 1")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()


    # ================================================
    #      4. CONVOLVE CLEAN SPEECH WITH RIRs
    # ================================================
    r_first = convolve_signals_with_rir(in1, h_first)
    r_second = convolve_signals_with_rir(in2, h_second)

    Lmax = max(r_first.shape[1], r_second.shape[1])
    if r_first.shape[1] < Lmax:
        r_first = np.pad(r_first, ((0, 0), (0, Lmax - r_first.shape[1])))
    if r_second.shape[1] < Lmax:
        r_second = np.pad(r_second, ((0, 0), (0, Lmax - r_second.shape[1])))

    print("\nConvolved signals:")
    print("r_first shape:", r_first.shape)
    print("r_second shape:", r_second.shape)


    # ================================================
    #      5. NORMALIZATION
    # ================================================
    r_first = r_first / (np.max(np.abs(r_first)) + 1e-12)
    r_second = r_second / (np.max(np.abs(r_second)) + 1e-12)


    # ================================================
    #      6. NOISE (diffuse + directional + mic)
    # ================================================
    # Directional
    noise_temp_len = Lmax - n_rir + 1
    noise_temp = np.random.randn(noise_temp_len).astype(np.float32)
    r_noise = convolve_signals_with_rir(noise_temp, h_noise)
    if r_noise.shape[1] < Lmax:
        r_noise = np.pad(r_noise, ((0,0),(0,Lmax - r_noise.shape[1])))
    else:
        r_noise = r_noise[:, :Lmax]

    # Diffuse
    diffuse_noise = np.random.randn(4, Lmax).astype(np.float32)
    diffuse_noise /= (np.std(diffuse_noise, axis=1, keepdims=True) + 1e-12)

    # Mix scales
    A_before = 0.5*(np.mean(np.std(r_first,axis=1)) + np.mean(np.std(r_second,axis=1)))
    Ax_SIR = A_before / (10**(SIR/20))

    receivers = Ax_SIR * r_first + r_second

    A_x = np.mean(np.std(receivers, axis=1))
    A_diffuse = A_x / (10**(SNR_diffuse/20))
    A_dir = A_x / (10**(SNR_direction/20))
    A_mic = A_x / (10**(SNR_mic/20))

    mic_noise = A_mic * np.random.randn(4, Lmax).astype(np.float32)

    receivers_mix = receivers + mic_noise + A_diffuse*diffuse_noise + A_dir*r_noise
    receivers_mix /= (np.max(np.abs(receivers_mix)) + 1e-12)

    noise_only = mic_noise + A_diffuse*diffuse_noise + A_dir*r_noise
    noise_only /= (np.max(np.abs(noise_only)) + 1e-12)

    print("\nStatistics:")
    print("Mean std of mixture:", A_x)
    print("A_diffuse:", A_diffuse)
    print("A_dir:", A_dir)
    print("A_mic:", A_mic)


    # ================================================
    #      7. SAVE OUTPUT FILES
    # ================================================
    sf.write("first_clean.wav", in1, fs)
    sf.write("second_clean.wav", in2, fs)
    sf.write("first_reverb.wav", r_first.T, fs)
    sf.write("second_reverb.wav", r_second.T, fs)
    sf.write("noise.wav", noise_only.T, fs)

    print("\nSaved output files:")
    print("first_clean.wav")
    print("second_clean.wav")
    print("first_reverb.wav  (4 channels)")
    print("second_reverb.wav (4 channels)")
    print("noise.wav         (4 channels)")
    print("\nDone!")

    clean = receivers       # זה ה-"speech" אחרי רוורב
    noise = mic_noise + A_diffuse*diffuse_noise + A_dir*r_noise
    
    power_clean = np.mean(clean**2)
    power_noise = np.mean(noise**2)
    
    snr_actual = 10 * np.log10(power_clean / power_noise)
    print("Actual SNR =", snr_actual, "dB")

if __name__ == "__main__":
    main()
