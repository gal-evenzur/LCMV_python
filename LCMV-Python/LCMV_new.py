import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as ss
import scipy.linalg as sl
from LCMV_functions import stft, istft
import os

# ==========================================
# 2. MAIN SCRIPT
# ==========================================

# %% Prepare
plt.close('all')

# %% Upload signals and set parameters

# Path Setup
root_mac = os.path.dirname(os.path.abspath(__file__))
root_mac = os.path.join(root_mac, 'wav_files')

# Assuming signal_9.wav is directly inside 'wav_files' based on the request logic.
# If strict adherence to 'exp_dir' subfolder structure is needed, uncomment the join below.
# exp_dir = 'Recordings/Sim_SNR_10_SIR_0_T60_300/9/'
rec_sig_file = os.path.join(root_mac, 'mixed_output.wav') 

noise_tim_st = 0
noise_tim_fn = 1.5 # Sec
first_tim_st = 2
first_tim_fn = 3.4 # Sec
second_tim_st = 5.2
second_tim_fn = 6.6 # Sec

# Load Audio
try:
    rec_sig, fs = sf.read(rec_sig_file)
except FileNotFoundError:
    print(f"Error: File not found at {rec_sig_file}")
    exit()

# Results directory
res_dir = os.path.join(root_mac, 'Results')
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

ref = 0 # Python 0-based index (MATLAB was 1, so ref+1 became 2. Here we use column 1 as ref)

# Signal splitting
# MATLAB: signal_proc = rec_sig(:,1); signal_mics = rec_sig(:,2:5);
if rec_sig.ndim > 1:
    signal_proc = rec_sig[:, 0]
    signal_mics = rec_sig[:, 1:5] # Columns 2 to 5 (indices 1,2,3,4)
    # signal_ref = rec_sig(:,ref+1) -> Column 2 in MATLAB -> Index 1 in Python
    signal_ref = rec_sig[:, ref + 1] 
else:
    print("Error: Input file must be multichannel")
    exit()

signal_mics_file = os.path.join(res_dir, 'signal_mics.wav')
signal_proc_file = os.path.join(res_dir, 'signal_proc.wav')
signal_ref_file = os.path.join(res_dir, 'signal_ref.wav')

sf.write(signal_mics_file, signal_mics, fs)

# Normalize and write
norm_proc = 0.9 * signal_proc / np.max(np.abs(signal_proc))
sf.write(signal_proc_file, norm_proc, fs)

norm_ref = 0.9 * signal_ref / np.max(np.abs(signal_ref))
sf.write(signal_ref_file, norm_ref, fs)


# %% STFT of mic. signals

M = signal_mics.shape[1]
z = signal_mics.T # (M, Samples)

# Parameters
wlen = 1024
R = int(wlen / 4) # Hop size
nfft = wlen
K = int(nfft / 2 + 1) # n_freq_bins which interest us

# MATLAB: win = hamming(wlen, 'periodic');
win = ss.windows.hamming(wlen, sym=False) 

# L = floor(1+((length(z(1,:))-wlen)/R));
L = int(np.floor(1 + ((z.shape[1] - wlen) / R)))

F_L = K - 1
F_R = K - 1

# z_k = ones(M,K,L)
z_k = np.ones((M, K, L), dtype=complex)

for i in range(M):
    # Call custom stft function
    stft_mat, f, t = stft(z[i, :], win, R, nfft)
    # Ensure correct slicing if stft returns slightly more frames due to rounding
    z_k[i, :, :] = stft_mat[:, :L]

# Visualization 1
T = np.arange(L) / fs * R
F = np.arange(K) * fs / 2 / (K - 1)

plt.figure(1)
plt.imshow(20 * np.log10(np.abs(z_k[0, :, :]) + np.finfo(float).eps), 
           aspect='auto', origin='lower', extent=[T[0], T[-1], F[0], F[-1]])
plt.xlabel('Time[Sec]', fontsize=14)
plt.ylabel('Frequency[Hz]', fontsize=14)
plt.colorbar()
plt.title('STFT Channel 1')


# %% Generate noise correlation matrix & apply Cholesky decomposition

noise_frm_st = int(np.ceil(noise_tim_st * fs / R))
noise_frm_fn = int(np.floor(noise_tim_fn * fs / R))

z_n = z_k[:, :, noise_frm_st:noise_frm_fn]

# Visualization 2
plt.figure(2)
Tn = np.arange(z_n.shape[2]) / fs * R
plt.imshow(20 * np.log10(np.abs(z_n[0, :, :]) + np.finfo(float).eps), 
           aspect='auto', origin='lower', extent=[Tn[0], Tn[-1], F[0], F[-1]])
plt.xlabel('Time[Sec]', fontsize=14)
plt.ylabel('Frequency[Hz]', fontsize=14)
plt.colorbar()
plt.title('Noise Segment')

epsilon = 0.01

noise_cor = np.zeros((K, M, M), dtype=complex)
noise_cor_chol = np.zeros((K, M, M), dtype=complex)
inv_chol = np.zeros((K, M, M), dtype=complex)

for k in range(K): # For each frequency bin
    temp_noise = z_n[:, k, :]
    # Correlation: (1/N) * X * X'
    noise_cor[k, :, :] = temp_noise @ temp_noise.conj().T / temp_noise.shape[1]
    
    # MATLAB: chol(A) produces Upper Triangular matrix.
    # To maintain 1:1 math compatibility with the 'solve' steps later, 
    # we request Upper triangular from numpy/scipy.
    try:
        U = sl.cholesky(noise_cor[k, :, :], lower=False)
    except sl.LinAlgError:
        # Fallback regularization
        U = sl.cholesky(noise_cor[k, :, :] + epsilon * np.eye(M), lower=False)
        
    noise_cor_chol[k, :, :] = U
    
    # inv_chol calculation (Diagonal Loading on the Cholesky factor)
    inv_chol[k, :, :] = U + epsilon * np.eye(M) * np.linalg.norm(U)


# %% RTF of first speaker

first_frm_st = int(np.ceil(first_tim_st * fs / R))
first_frm_fn = int(np.floor(first_tim_fn * fs / R))

z_f = z_k[:, :, first_frm_st:first_frm_fn]

# Visualization 3
plt.figure(3)
Tf = np.arange(z_f.shape[2]) / fs * R
plt.imshow(20 * np.log10(np.abs(z_f[0, :, :]) + np.finfo(float).eps), 
           aspect='auto', origin='lower', extent=[Tf[0], Tf[-1], F[0], F[-1]])
plt.xlabel('Time[Sec]', fontsize=14)
plt.ylabel('Frequency[Hz]', fontsize=14)
plt.colorbar()
plt.title('Speaker 1 Segment')

z_f_cor = np.zeros((K, M, M), dtype=complex)
G_f = np.zeros((K, M), dtype=complex)

for k in range(K):
    temp_first = z_f[:, k, :]
    
    # CODE CONTEXT: Inside the frequency loop (k)

    # --- Step 1: Whitening ---
    # We want: x_tilde = U_inv * x
    # Solving linear system 'U * x_tilde = x' is mathematically equivalent to 'x_tilde = inv(U) * x'
    # inv_chol[k] is our Matrix U.
    temp_first_w = sl.solve(inv_chol[k, :, :], temp_first)

    # --- Step 2: Calculate R_tilde_xx ---
    # Calculate Covariance of the whitened signal (R_tilde_xx)
    N = temp_first_w.shape[1]
    z_f_cor_temp = temp_first_w @ temp_first_w.conj().T / N

    # --- Step 3: PCA (Eigen Decomposition) ---
    # finding the Eigenvectors (vals, vecs)
    vals, vecs = np.linalg.eigh(z_f_cor_temp)

    # The function returns eigenvalues in ascending order.
    # We want the largest one (The dominant signal direction).
    # This is v_max
    fi = vecs[:, -1]

    # --- Step 4: Re-coloring ---
    # We want: h = U * v_max
    # noise_cor_chol[k] is U.
    # We multiply U by the eigenvector fi
    temp = noise_cor_chol[k, :, :] @ fi # Use standard matrix multiplication

    G_f[k, :] = temp / temp[ref] # Normalize by reference mic

# Handle outliers / division by zero
for m in range(M):
    ind = np.where(np.abs(G_f[:, m]) > 3 * np.mean(np.abs(G_f[:, m])))[0]
    if len(ind) > 0:
        # 2*binornd(1,0.5)-1 -> Randomly 1 or -1
        G_f[ind, m] = 2 * np.random.binomial(1, 0.5, len(ind)) - 1

# IFFT Truncation
# Create full spectrum
G_f_full = np.vstack([G_f, np.conj(G_f[K-2:0:-1, :])])
g_f = np.fft.ifft(G_f_full, axis=0)

g_f_trc = np.zeros_like(g_f)
g_f_trc[0:F_R, :] = g_f[0:F_R, :]
g_f_trc[nfft-F_L:nfft, :] = g_f[nfft-F_L:nfft, :]
# --- PLOTTING FIGURE 4 (Speaker 1 ReIR) ---
plt.figure(4)
# Plot the matrix (plots all columns)
plt.plot(np.fft.ifftshift(g_f_trc.real, axes=0))
plt.title('ReIR - Speaker 1 (Relative Impulse Response)')
plt.xlabel('Time Samples')
plt.ylabel('Amplitude')

# Dynamic Legend Generation
# Create a list like ['Mic 1', 'Mic 2', 'Mic 3', 'Mic 4']
mic_labels = [f"Mic {m+1}" for m in range(M)]

# Identify the Reference Mic in the legend
mic_labels[ref] += " (Ref)" 

plt.legend(mic_labels, loc='upper right')
plt.grid(True, alpha=0.3) # Adding a grid helps read the chart better

G_f_trc_full = np.fft.fft(g_f_trc, axis=0)
G_f = G_f_trc_full[:K, :]


# %% RTF of second speaker

second_frm_st = int(np.ceil(second_tim_st * fs / R))
second_frm_fn = int(np.floor(second_tim_fn * fs / R))

z_s = z_k[:, :, second_frm_st:second_frm_fn]

# Visualization 5
plt.figure(5)
Ts = np.arange(z_s.shape[2]) / fs * R
plt.imshow(20 * np.log10(np.abs(z_s[0, :, :]) + np.finfo(float).eps), 
           aspect='auto', origin='lower', extent=[Ts[0], Ts[-1], F[0], F[-1]])
plt.xlabel('Time[Sec]', fontsize=14)
plt.ylabel('Frequency[Hz]', fontsize=14)
plt.colorbar()
plt.title('Speaker 2 Segment')

z_s_cor = np.zeros((K, M, M), dtype=complex)
G_s = np.zeros((K, M), dtype=complex)

for k in range(K):
    temp_second = z_s[:, k, :]
    
    # Whiten
    temp_second_w = sl.solve(inv_chol[k, :, :], temp_second)
    
    z_s_cor_temp = temp_second_w @ temp_second_w.conj().T / temp_second_w.shape[1]
    
    vals, vecs = np.linalg.eigh(z_s_cor_temp)
    fi = vecs[:, -1]
    
    temp = noise_cor_chol[k, :, :] @ fi
    G_s[k, :] = temp / temp[ref]

for m in range(M):
    ind = np.where(np.abs(G_s[:, m]) > 3 * np.mean(np.abs(G_s[:, m])))[0]
    if len(ind) > 0:
        G_s[ind, m] = 2 * np.random.binomial(1, 0.5, len(ind)) - 1

G_s_full = np.vstack([G_s, np.conj(G_s[K-2:0:-1, :])])
g_s = np.fft.ifft(G_s_full, axis=0)

g_s_trc = np.zeros_like(g_s)
g_s_trc[0:F_R, :] = g_s[0:F_R, :]
g_s_trc[nfft-F_L:nfft, :] = g_s[nfft-F_L:nfft, :]

plt.figure(6)
plt.plot(np.fft.ifftshift(g_s_trc.real, axes=0))
plt.title('ReIR - Speaker 2 (Relative Impulse Response)')
plt.xlabel('Time Samples')
plt.ylabel('Amplitude')

# Re-use the labels
plt.legend(mic_labels, loc='upper right')
plt.grid(True, alpha=0.3)

G_s_trc_full = np.fft.fft(g_s_trc, axis=0)
G_s = G_s_trc_full[:K, :]


# %% Combine RTFs

# G shape: (K, M, 2)
G = np.dstack((G_f, G_s))


# %% Generate W and apply the MVDR beamformer

W = np.zeros((M, 2, K), dtype=complex)
z_out = np.zeros((K, 2, L), dtype=complex)

eye_M = np.eye(M)
eye_2 = np.eye(2)

for k in range(K):
    g = G[k, :, :] # (M, 2)
    b = noise_cor[k, :, :] # (M, M)
    
    # inv_b = b + epsilon*norm(b)*eye(M)
    reg_b = b + epsilon * np.linalg.norm(b) * eye_M
    # Invert reg_b (Using solve for better stability implies we define what we solve against, 
    # but here we need the explicit matrix or solve against g directly).
    # MATLAB: c = inv_b \ g;
    c = sl.solve(reg_b, g)
    
    # inv_temp = g'*c + epsilon*norm(g'*c)*eye(2)
    denom = g.conj().T @ c
    reg_denom = denom + epsilon * np.linalg.norm(denom) * eye_2
    
    # W = c / inv_temp -> c * inv(inv_temp)
    # Solve X * reg_denom = c  => X = c * inv(reg_denom)
    # Or transpose: reg_denom.T * X.T = c.T
    # Easier: W = c @ inv(reg_denom)
    W[:, :, k] = c @ sl.inv(reg_denom)
    
    # z_out = W' * z_k
    z_out[k, :, :] = W[:, :, k].conj().T @ z_k[:, k, :]


# %% ISTFT

# Call custom istft
first_channel, t1 = istft(z_out[:, 0, :], win, win, R, nfft, fs)
second_channel, t2 = istft(z_out[:, 1, :], win, win, R, nfft, fs)

first_channel_file = os.path.join(res_dir, 'first_out.wav')
second_channel_file = os.path.join(res_dir, 'second_out.wav')

# Write output
norm_first = 0.9 * first_channel / np.max(np.abs(first_channel))
sf.write(first_channel_file, norm_first, fs)

norm_second = 0.9 * second_channel / np.max(np.abs(second_channel))
sf.write(second_channel_file, norm_second, fs)

plt.show()
print("Processing complete. Files saved in 'Results' folder.")