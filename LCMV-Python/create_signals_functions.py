import numpy as np
import scipy.signal as ss
import scipy.special as sp
import os
import soundfile as sf


def load_clean_wav(path):
    """Helper to load wav files safely."""
    if not os.path.exists(path):
        print(f"Warning: File {path} not found. Generating random signal.")
        return np.random.randn(16000 * 2) # 2 seconds of noise as fallback
    data, _ = sf.read(path)
    if data.ndim > 1:
        data = data[:, 0] # Take first channel if stereo
    return data

def fillline(startp, endp, pts):
    """
    Equivalent to fillline.m
    Generates coordinates for a line between two points.
    Python's linspace handles vertical/horizontal lines automatically.
    """
    pts = int(pts)
    xx = np.linspace(startp[0], endp[0], pts)
    yy = np.linspace(startp[1], endp[1], pts)
    return xx, yy

def mix_signals(n_sig, DC, method='cholesky'):
    """
    Equivalent to mix_signals.m
    Mix M mutually independent signals to exhibit specific spatial coherence.
    """
    M = n_sig.shape[1] # Number of sensors
    L = n_sig.shape[0] # Length input signal
    K = (DC.shape[2] - 1) * 2 # FFT length derived from DC matrix (K/2 + 1)
    
    # STFT parameters to match MATLAB: Window=K, Hop=K/4 -> Overlap = 3/4 * K
    nperseg = K
    noverlap = int(K * 3 / 4)
    
    # Perform STFT for each channel
    # Scipy STFT returns (f, t, Zxx). We need to arrange it for processing.
    # We use a Hanning window by default as it is standard in audio.
    N_stft = []
    for m in range(M):
        f, t, Zxx = ss.stft(n_sig[:, m], fs=16000, window='hann', nperseg=nperseg, noverlap=noverlap)
        N_stft.append(Zxx)
    
    N_stft = np.array(N_stft) # Shape: (M, Freqs, Frames)
    
    # Initialization
    # DC shape is (M, M, Freqs)
    # C shape will be (M, M, Freqs)
    C = np.zeros(DC.shape, dtype=complex)
    X_stft = np.zeros_like(N_stft, dtype=complex)
    
    num_bins = DC.shape[2]
    
    for k in range(num_bins):
        # Extract coherence matrix for this frequency bin
        DC_k = DC[:, :, k]
        
        # Ensure DC_k is symmetric/Hermitian (numerical stability)
        DC_k = (DC_k + DC_k.conj().T) / 2
        
        if method == 'cholesky':
            try:
                # np.linalg.cholesky returns Lower triangle by default (L * L.H = A)
                # MATLAB chol(A) returns Upper triangle (R.H * R = A)
                # Code uses C' * N. 
                # If C is Cholesky factor of DC, DC = C * C' (in MATLAB notation logic)
                # We need to stick to the math: X = C' * N (Matrix mult)
                # If we use numpy cholesky: L = cholesky(DC) -> DC = L @ L.H
                # We want X @ X.H = DC ideally.
                C_k = np.linalg.cholesky(DC_k) 
                # In Python we will multiply L @ N for the mixing
                C[:, :, k] = C_k
            except np.linalg.LinAlgError:
                 # Fallback for non-positive definite matrices (common in numerical errors)
                 eigval, eigvec = np.linalg.eigh(DC_k)
                 eigval[eigval < 0] = 0
                 C[:, :, k] = eigvec @ np.diag(np.sqrt(eigval))
                 
        elif method == 'eigen':
            eigval, eigvec = np.linalg.eigh(DC_k)
            # Ensure positive eigenvalues
            eigval[eigval < 0] = 0
            C[:, :, k] = eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.conj().T # Wait, sqrt(D)*V' in matlab?
            # MATLAB: C = sqrt(D) * V'. 
            # Python equivalent for reconstruction:
            C[:, :, k] = np.diag(np.sqrt(eigval)) @ eigvec.conj().T
            
        else:
             raise ValueError('Unknown method specified.')

        # Mixing: X = C * N (using broadcasting or matrix mult)
        # N_stft slice is (M, Frames). C_k is (M, M).
        # We want X(k) = C(k) * N(k). 
        # Note: The MATLAB code used X = C' * N. 
        # If C came from chol(DC) in Matlab (Upper), C'*C = DC. 
        # Here we used Numpy Cholesky (Lower), L*L' = DC.
        # So we just use L.
        
        X_stft[:, k, :] = C[:, :, k] @ N_stft[:, k, :]

    # Inverse STFT
    x_out = np.zeros((L, M))
    for m in range(M):
        _, x_rec = ss.istft(X_stft[m, :, :], fs=16000, window='hann', nperseg=nperseg, noverlap=noverlap)
        x_out[:, m] = x_rec[:L] # Trim to original length
        
    return x_out

def create_difuse_noise(L, M=4, Fs=16000, c=340, d=0.2, type_nf='spherical'):
    """
    Equivalent to create_difuse_noise.m
    Modified to accept M, Fs, c, d as arguments to match main script context.
    """
    K = 256 # FFT length
    
    # Generate M mutually independent input signals of length L
    n_sig = np.random.randn(L, M)
    
    # Generate matrix with desired spatial coherence
    # ww = 2*pi*Fs*(0:K/2)/K
    # In Python, rfftfreq gives [0, 1, ..., n/2] / (n*dt)
    freqs = np.fft.rfftfreq(K, d=1/Fs)
    ww = 2 * np.pi * freqs
    
    DC = np.zeros((M, M, len(ww)), dtype=complex)
    
    for p in range(M):
        for q in range(M):
            if p == q:
                DC[p, q, :] = np.ones(len(ww))
            else:
                dist = abs(p - q) * d
                if type_nf == 'spherical':
                    # sinc in numpy is sin(pi*x)/(pi*x). 
                    # MATLAB: sinc(ww*dist/(c*pi)) -> sin(ww*dist/c) / (ww*dist/c)
                    # We need to pass x such that pi*x = ww*dist/c
                    # x = ww*dist/(c*pi)
                    
                    # Handle w=0 case (avoid NaN, though numpy.sinc handles 0->1)
                    val = ww * dist / (c * np.pi)
                    DC[p, q, :] = np.sinc(val)
                    
                elif type_nf == 'cylindrical':
                    # bessel(0, x) -> scipy.special.jv(0, x)
                    val = ww * dist / c
                    DC[p, q, :] = sp.jv(0, val)
                else:
                    raise ValueError('Unknown noise field.')

    # Mix signals
    x = mix_signals(n_sig, DC, 'cholesky')
    return x