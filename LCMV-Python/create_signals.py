from create_signals_functions import *
import matplotlib.pyplot as plt
import rir_generator as rir
import soundfile as sf
import scipy.signal as ss
import numpy as np
import os

# Ensure 'wav_files' directory exists in the same directory as this script
# Place  in the 'wav_files' directory the required audio files:
#   (male_11.wav, male_12.wav, female_21.wav, female_22.wav)


# ==========================================
# 1. HYPERPARAMETER DICTIONARY
# ==========================================
hParams = {
    # Physics & Environment
    'seed': 42,                    # Random seed for reproducibility
    'sound_speed': 340.0,
    'fs': 44100,                  # Target sampling rate
    'room_height': 3,
    'room_dim_min': 4.1,          # Minimum length/width for random room
    'room_dim_max': 6.0,          # Maximum length/width
    'rt60': 0.3,                  # Reverberation time (beta)
    
    # RIR Generation
    'rir_length': 4096,           # Samples (n)
    'mic_type': rir.mtype.omnidirectional,
    'order': -1,                  # Reflection order (-1 is max)
    'dim': 3,                     # 3D simulation
    'orientation': [0, 0],        # [Azimuth, Elevation]
    'hp_filter': True,            # Enable High-pass filter
    
    # Geometry (Mics & Speakers)
    'array_center_dist': 1.3,     # Distance of table from random center (R)
    'array_radius': 0.1,          # Radius of the mic array (radius_mics)
    'mic_height': 1.0,            # Height of mics (high)
    'wall_margin': 0.5,           # Min distance from wall (distance_from_woll)
    'speaker_min_dist': 0.5,      # Min distance between speakers
    'noise_source_dist': 2.0,     # Min distance for noise source from center
    'noise_var': 0.2,             # Variance for random placement (noise_R)
    
    # Microphone Indices (Selecting 4 mics from the 360 generated points)
    'mic_indices': [0, 59, 119, 179], 
    
    # Mixing Ratios (dB)
    'sir': -20,                     # Signal-to-Interference Ratio
    'snr_diffuse': 20,            # Diffuse (Ambient) Noise
    'snr_directional': 2,        # Directional Noise
    'snr_mic': 30                 # Microphone (Sensor) Noise
}

# Seeded RNG
rng = np.random.default_rng(hParams['seed'])

# ==========================================
# 2. MAIN SCRIPT
# ==========================================

# Setup Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
wav_dir = os.path.join(current_dir, "wav_files")

# Verify Sampling Rate matches hParams
# (Optional: You could also resample here if they don't match)
_, file_fs = sf.read(os.path.join(wav_dir, "female_21.wav"))
if file_fs != hParams['fs']:
    print(f"Warning: File sampling rate ({file_fs}) differs from hParams ({hParams['fs']})")

# --- Geometry Setup ---

# Randomize Room Dimensions
L1 = 4 + 0.1 * rng.integers(
    (hParams['room_dim_min']-4)*10,
    (hParams['room_dim_max']-4)*10 + 1
)
L2 = 4 + 0.1 * rng.integers(
    (hParams['room_dim_min']-4)*10,
    (hParams['room_dim_max']-4)*10 + 1
)
L = [L1, L2, hParams['room_height']]
room_x, room_y = L[0], L[1]

# Calculate Safe Zone for Array Placement
# We define the "end points" to ensure the array fits within the room with margins
dist_total = hParams['array_center_dist'] + hParams['wall_margin'] + hParams['noise_var']
end_point_x = room_x - dist_total
end_point_y = room_y - dist_total

# Randomize center of the array within safe zone
Radius_X = (end_point_x - dist_total) * rng.random() + dist_total
Radius_Y = (end_point_y - dist_total) * rng.random() + dist_total

# Randomize Array Orientation
angle = 360
R_angle = rng.integers(0, angle) 
t_full = np.linspace(-np.pi, 2 * np.pi, int(angle + angle/2)) 
t = t_full[R_angle : R_angle + int(angle/2)]

# Calculate Coordinates
# Table center offset (x, y)
x = hParams['array_center_dist'] * np.sin(t) + Radius_X
y = hParams['array_center_dist'] * np.cos(t) + Radius_Y

# Mic locations (circ_mics)
circ_mics_x = hParams['array_radius'] * np.sin(t) + Radius_X
circ_mics_y = hParams['array_radius'] * np.cos(t) + Radius_Y

# Line visualization (Helper)
line_x, line_y = fillline(
    [x[0], y[0]], 
    [x[int(angle/2)-1], y[int(angle/2)-1]], 
    hParams['array_center_dist']*2*100
)

# Select specific microphones from the circle
indices = [i for i in hParams['mic_indices'] if i < len(circ_mics_x)]
r = np.zeros((len(indices), 3))
for i, idx in enumerate(indices):
    r[i, :] = [circ_mics_x[idx], circ_mics_y[idx], hParams['mic_height']]

# Place Speakers (Source Locations)
next_speech = True
while next_speech:
    next_speech = False
    
    # Randomize Speaker 1
    rand1 = rng.integers(0, int(angle/2))
    w1 = 0.01 * rng.integers(1, 315)
    x1 = x[rand1] + hParams['noise_var'] * np.sin(w1)
    y1 = y[rand1] + hParams['noise_var'] * np.cos(w1)

    # Randomize Speaker 2
    rand2 = rng.integers(0, int(angle/2))
    w2 = 0.01 * rng.integers(1, 315)
    x2 = x[rand2] + hParams['noise_var'] * np.sin(w2)
    y2 = y[rand2] + hParams['noise_var'] * np.cos(w2)

    # Check Minimum Distance
    loc_xy = np.array([[x1, y1], [x2, y2]])
    dist = np.linalg.norm(loc_xy[0] - loc_xy[1])
    
    if dist < hParams['speaker_min_dist']:
        next_speech = True

s_first = [x1, y1, hParams['mic_height']]
s_second = [x2, y2, hParams['mic_height']]

# --- Prepare Audio Signals ---

speech_11 = load_clean_wav(os.path.join(wav_dir, "male_11.wav"))
speech_12 = load_clean_wav(os.path.join(wav_dir, "male_12.wav"))
speech_21 = load_clean_wav(os.path.join(wav_dir, "female_21.wav"))
speech_22 = load_clean_wav(os.path.join(wav_dir, "female_22.wav"))

fs = hParams['fs']
pad_zeros1 = np.zeros(fs) 
pad_zeros2 = np.zeros(fs * 2) 

# Timeline Construction
# Timeline 1: [Silence (2s)] [Spk1] [Silence] [Silence] [Silence] [Spk1]
in1 = np.concatenate([
    pad_zeros2, speech_11, pad_zeros1, np.zeros(len(speech_21)), pad_zeros1, speech_12
])
# Timeline 2: [Silence (2s)] [Silence] [Silence] [Spk2] [Silence] [Spk2]
in2 = np.concatenate([
    pad_zeros2, np.zeros(len(speech_11)), pad_zeros1, speech_21, pad_zeros1, speech_22
])

# Pad to match lengths
maxlen = max(len(in1), len(in2))
in1 = np.pad(in1, (0, maxlen - len(in1)), 'constant')
in2 = np.pad(in2, (0, maxlen - len(in2)), 'constant')

# --- Room Simulation (RIR) ---

common_rir_args = {
    'c': hParams['sound_speed'], 
    'fs': hParams['fs'], 
    'r': r, 
    'L': L, 
    'reverberation_time': hParams['rt60'], 
    'nsample': hParams['rir_length'], 
    'mtype': hParams['mic_type'], 
    'order': hParams['order'], 
    'dim': hParams['dim'], 
    'orientation': hParams['orientation'], 
    'hp_filter': hParams['hp_filter']
}

# Generate RIRs
h_first = rir.generate(s=s_first, **common_rir_args)
h_second = rir.generate(s=s_second, **common_rir_args)

# Convolve
receiver_first = ss.fftconvolve(in1[:, np.newaxis], h_first, mode='full')
receiver_second = ss.fftconvolve(in2[:, np.newaxis], h_second, mode='full')

# --- Noise Generation ---

# 1. Directional Noise Placement
middle = np.array([Radius_X, Radius_Y, hParams['mic_height']])
s_noise = np.array([Radius_X, Radius_Y, hParams['mic_height']])
d_noise = 0

while d_noise < hParams['noise_source_dist']:
    x_noise = hParams['wall_margin'] + 0.01 * rng.integers(0, 100) * (room_x - 2 * hParams['wall_margin'])
    y_noise = hParams['wall_margin'] + 0.01 * rng.integers(0, 100) * (room_y - 2 * hParams['wall_margin'])
    s_noise = np.array([x_noise, y_noise, hParams['mic_height']])
    d_noise = np.linalg.norm(s_noise - middle)

# 2. Generate Directional Noise Signal
noise_len_needed = receiver_first.shape[0] - hParams['rir_length'] + 1
noise_temp_mono = rng.standard_normal(noise_len_needed)
noise_temp = np.tile(noise_temp_mono[:, np.newaxis], (1, 1))

# 3. Process Directional Noise
h_noise = rir.generate(s=s_noise, **common_rir_args)
Receivers_noise = ss.fftconvolve(noise_temp, h_noise, mode='full')

# Normalize Directional Noise
std_vals = np.std(Receivers_noise, axis=1)
if np.mean(std_vals) != 0:
    Receivers_noise = Receivers_noise / np.mean(std_vals)

# 4. Generate Diffuse Noise
M = receiver_first.shape[1] 
length_receives = receiver_first.shape[0]

difuse_noise = create_difuse_noise(length_receives, M=M, Fs=fs, c=hParams['sound_speed'])

if np.mean(np.std(difuse_noise)) != 0:
    difuse_noise = difuse_noise / np.mean(np.std(difuse_noise))

# --- Mixing ---

# Normalize Reverb Signals
receiver_first = receiver_first / np.max(np.abs(receiver_first))
receiver_second = receiver_second / np.max(np.abs(receiver_second))

# Apply SIR (Speaker Balance)
A_x_before = (np.mean(np.std(receiver_first)) + np.mean(np.std(receiver_second))) / 2
Ax_SIR = A_x_before / (10**(hParams['sir']/20.0))
receivers_speech = Ax_SIR * receiver_first + receiver_second

# Determine overall level for Noise Scaling
A_x = np.mean(np.std(receivers_speech))

# Align Lengths
min_len = min(receivers_speech.shape[0], Receivers_noise.shape[0], difuse_noise.shape[0])
receivers_speech = receivers_speech[:min_len, :]
Receivers_noise = Receivers_noise[:min_len, :]
difuse_noise = difuse_noise[:min_len, :]

# Calculate Noise Amplitudes
A_n_difuse = A_x / (10**(hParams['snr_diffuse']/20.0))
A_n_direction = A_x / (10**(hParams['snr_directional']/20.0))
A_n_mic = A_x / (10**(hParams['snr_mic']/20.0))

# 5. Generate Microphone Noise
mic_noise = A_n_mic * rng.standard_normal((min_len, M))

# Final Mix
receivers_mixed = (receivers_speech + 
                   mic_noise + 
                   A_n_difuse * difuse_noise + 
                   A_n_direction * Receivers_noise)

noise_only = (mic_noise + 
              A_n_difuse * difuse_noise + 
              A_n_direction * Receivers_noise)

# Final Normalization
receivers_mixed = receivers_mixed / np.max(np.abs(receivers_mixed))

# --- Export ---
output_path = wav_dir
sf.write(os.path.join(output_path, 'first_clean.wav'), in1, fs)
sf.write(os.path.join(output_path, 'second_clean.wav'), in2, fs)
sf.write(os.path.join(output_path, 'first_reverb.wav'), receiver_first, fs)
sf.write(os.path.join(output_path, 'second_reverb.wav'), receiver_second, fs)
sf.write(os.path.join(output_path, 'noise.wav'), noise_only, fs)
sf.write(os.path.join(output_path, 'mixed_output.wav'), receivers_mixed, fs)

print("Processing complete. Files saved using hParams configuration.")