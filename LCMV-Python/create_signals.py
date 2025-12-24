from create_signals_functions import *
import matplotlib.pyplot as plt
import rir_generator as rir


# --- Main Script ---

# --- Main Script ---

# %% variens
c_k = 340.0                                       
c = 340.0
current_dir = os.path.dirname(os.path.abspath(__file__))
_, fs = sf.read(os.path.join(current_dir, "female_21.wav"))
print(f"Sampling frequency set to: {fs} Hz")
n = 4096                                          
mtype = rir.mtype.omnidirectional   # Corrected Enum usage
order = -1                                       
dim = 3                                           
orientation = [0, 0]                              
hp_filter = True                                  

# define analysis parameters
NO_S_i = 10
lottery = 2000
R = 1.3
R_small = 1.2
pad_len = 25000 
start = 1
max1 = 100
noise_R = 0.2
flag = 0
flag2 = 0
nfft = 2048
hop = 1024
num_jumps = 9
high = 1
angle = 360
distance_from_woll = 0.5
radius_mics = 0.1

# %% rir generation & covariens sounds 

L1_temp = np.random.randint(1, 21)
L1 = 4 + 0.1 * L1_temp
L2_temp = np.random.randint(1, 21)
L2 = 4 + 0.1 * L2_temp
L = [L1, L2, 3] 
room_x = L[0]
room_y = L[1]
SNR_difuse = 20
beta = 0.3 
SIR = 0
SNR_direction = 20
SNR_mic = 30

# %% create circle & line & mic location % speaker location

distance_total = R + distance_from_woll + noise_R
end_point_x = room_x - (R + distance_from_woll + noise_R)
end_point_y = room_y - (R + distance_from_woll + noise_R)

Radius_X = (end_point_x - distance_total) * np.random.rand() + distance_total
Radius_Y = (end_point_y - distance_total) * np.random.rand() + distance_total

R_angle = np.random.randint(0, angle) 
t_full = np.linspace(-np.pi, 2 * np.pi, int(angle + angle/2)) 
t = t_full[R_angle : R_angle + int(angle/2)]

x = R * np.sin(t) + Radius_X
y = R * np.cos(t) + Radius_Y
x_small = R_small * np.sin(t) + Radius_X
y_small = R_small * np.cos(t) + Radius_Y

z = 0 * t + high
circ_mics_x = radius_mics * np.sin(t) + Radius_X
circ_mics_y = radius_mics * np.cos(t) + Radius_Y

# Generate line coordinates using new function
line_x, line_y = fillline([x[0], y[0]], [x[int(angle/2)-1], y[int(angle/2)-1]], R*2*100)

indices = [0, 59, 119, 179]
indices = [i for i in indices if i < len(circ_mics_x)]

r = np.zeros((len(indices), 3))
for i, idx in enumerate(indices):
    r[i, :] = [circ_mics_x[idx], circ_mics_y[idx], high]

# %% create locations   
    
center = np.array([Radius_X, Radius_Y])
start_circ_vec = np.array([line_x[0], line_y[0]]) - center
labels_location = np.arange(5, 176, 10)
list_locations = []
next_speech = True

while next_speech:
    next_speech = False
    rand1 = np.random.randint(0, int(angle/2))
    x1 = x[rand1]
    y1 = y[rand1]
    
    w = 0.01 * np.random.randint(1, 315)
    x1 = x1 + noise_R * np.sin(w)
    y1 = y1 + noise_R * np.cos(w)

    rand2 = np.random.randint(0, int(angle/2))
    x2 = x[rand2]
    y2 = y[rand2]
    
    w = 0.01 * np.random.randint(1, 315)
    x2 = x2 + noise_R * np.sin(w)
    y2 = y2 + noise_R * np.cos(w)

    loc_xy = np.array([[x1, y1], [x2, y2]])
    dist = np.linalg.norm(loc_xy[0] - loc_xy[1])
    
    if dist < 0.5:
        next_speech = True

s_first = [x1, y1, high]
s_second = [x2, y2, high]

# %% create speakers clean

current_dir = os.path.dirname(os.path.abspath(__file__))
    
speech_11 = load_clean_wav(os.path.join(current_dir, "male_11.wav"))
speech_12 = load_clean_wav(os.path.join(current_dir, "male_12.wav"))
speech_21 = load_clean_wav(os.path.join(current_dir, "female_21.wav"))
speech_22 = load_clean_wav(os.path.join(current_dir, "female_22.wav"))

pad_zeros1 = np.zeros(fs) 
pad_zeros2 = np.zeros(fs * 2) 

in1 = np.concatenate([pad_zeros2, speech_11, pad_zeros1, np.zeros(len(speech_21)), pad_zeros1, speech_12])
vad1 = np.concatenate([pad_zeros2, np.ones(len(speech_11)), pad_zeros1, np.zeros(len(speech_21)), pad_zeros1, np.ones(len(speech_12))])

in2 = np.concatenate([pad_zeros2, np.zeros(len(speech_11)), pad_zeros1, speech_21, pad_zeros1, speech_22])
vad2 = np.concatenate([pad_zeros2, np.zeros(len(speech_11)), pad_zeros1, np.ones(len(speech_21)), pad_zeros1, np.ones(len(speech_22))])

maxlen = max(len(in1), len(in2))
in1 = np.pad(in1, (0, maxlen - len(in1)), 'constant')
in2 = np.pad(in2, (0, maxlen - len(in2)), 'constant')
vad1 = np.pad(vad1, (0, maxlen - len(vad1)), 'constant')
vad2 = np.pad(vad2, (0, maxlen - len(vad2)), 'constant')

vad = vad1 + vad2

# %% create RIRs

h_first = rir.generate(
    c=c_k, fs=fs, r=r, s=s_first, L=L, reverberation_time=beta, 
    nsample=n, mtype=mtype, order=order, dim=dim, orientation=orientation, hp_filter=hp_filter
)

receiver_first = ss.fftconvolve(in1[:, np.newaxis], h_first, mode='full')

h_second = rir.generate(
    c=c_k, fs=fs, r=r, s=s_second, L=L, reverberation_time=beta, 
    nsample=n, mtype=mtype, order=order, dim=dim, orientation=orientation, hp_filter=hp_filter
)

receiver_second = ss.fftconvolve(in2[:, np.newaxis], h_second, mode='full')

# %% create noise

middle = np.array([Radius_X, Radius_Y, high])
s_noise = np.array([Radius_X, Radius_Y, high])
d_noise = np.linalg.norm(s_noise - middle)

while d_noise < 2:
    x_noise = distance_from_woll + 0.01 * np.random.randint(0, 100) * (room_x - 2 * distance_from_woll)
    y_noise = distance_from_woll + 0.01 * np.random.randint(0, 100) * (room_y - 2 * distance_from_woll)
    s_noise = np.array([x_noise, y_noise, high])
    d_noise = np.linalg.norm(s_noise - middle)

noise_len_needed = receiver_first.shape[0] - n + 1
noise_temp_mono = np.random.randn(noise_len_needed) 
noise_temp = np.tile(noise_temp_mono[:, np.newaxis], (1, 1))

h_noise = rir.generate(
    c=c_k, fs=fs, r=r, s=s_noise, L=L, reverberation_time=beta, 
    nsample=n, mtype=mtype, order=order, dim=dim, orientation=orientation, hp_filter=hp_filter
)

Receivers_noise = ss.fftconvolve(noise_temp, h_noise, mode='full')

std_vals = np.std(Receivers_noise, axis=1)
if np.mean(std_vals) != 0:
    Receivers_noise = Receivers_noise / np.mean(std_vals)

# %% create input

M = receiver_first.shape[1] 
length_receives = receiver_first.shape[0]

# --- MODIFIED: Calling the translated create_difuse_noise function ---
# Note: we explicitly pass M, fs, and c so it matches the current setup
difuse_noise = create_difuse_noise(length_receives, M=M, Fs=fs, c=c_k)

if np.mean(np.std(difuse_noise)) != 0:
    difuse_noise = difuse_noise / np.mean(np.std(difuse_noise))

receiver_first = receiver_first / np.max(np.abs(receiver_first))
receiver_second = receiver_second / np.max(np.abs(receiver_second))

A_x_before = (np.mean(np.std(receiver_first)) + np.mean(np.std(receiver_second))) / 2
Ax_SIR = A_x_before / (10**(SIR/20.0))

receivers = Ax_SIR * receiver_first + receiver_second

A_x = np.mean(np.std(receivers))

# Align lengths
min_len = min(receivers.shape[0], Receivers_noise.shape[0], difuse_noise.shape[0])
receivers = receivers[:min_len, :]
Receivers_noise = Receivers_noise[:min_len, :]
difuse_noise = difuse_noise[:min_len, :]

A_n_difuse = A_x / (10**(SNR_difuse/20.0))
A_n_diraction = A_x / (10**(SNR_direction/20.0))
A_n_mic = A_x / (10**(SNR_mic/20.0))

mic_noise = A_n_mic * np.random.randn(min_len, M)

receivers_mixed = (receivers + mic_noise + A_n_difuse * difuse_noise + A_n_diraction * Receivers_noise)
noise_only = (mic_noise + A_n_difuse * difuse_noise + A_n_diraction * Receivers_noise)

receivers_mixed = receivers_mixed / np.max(np.abs(receivers_mixed))

# Output paths
output_path = current_dir
first_clean_file = os.path.join(output_path, 'first_clean.wav')
second_clean_file = os.path.join(output_path, 'second_clean.wav')
first_reverb_file = os.path.join(output_path, 'first_reverb.wav')
second_reverb_file = os.path.join(output_path, 'second_reverb.wav')
data_noise_file = os.path.join(output_path, 'noise.wav')
mixed_out_file = os.path.join(output_path, 'mixed_output.wav')

sf.write(first_clean_file, in1, fs)
sf.write(second_clean_file, in2, fs)
sf.write(first_reverb_file, receiver_first, fs)
sf.write(second_reverb_file, receiver_second, fs)
sf.write(data_noise_file, noise_only, fs)
sf.write(mixed_out_file, receivers_mixed, fs)

print("Processing complete. Files saved.")