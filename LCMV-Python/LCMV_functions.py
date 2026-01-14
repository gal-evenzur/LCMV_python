import numpy as np
import matplotlib.pyplot as plt

def stft(x, win, hop, nfft):
    """
    Short-Time Fourier Transform.
    Translates the custom MATLAB implementation to Python.
    """
    fs = 16000
    
    # Ensure inputs are appropriate 1D arrays
    x = np.array(x).flatten()
    win = np.array(win).flatten()
    
    xlen = len(x)
    wlen = len(win)
    
    # NUP = ceil((nfft+1)/2)
    NUP = int(np.ceil((nfft + 1) / 2))
    
    # L = 1+fix((xlen-wlen)/hop)
    L = 1 + int((xlen - wlen) / hop)
    
    # STFT = zeros(NUP,L)
    STFT = np.zeros((NUP, L), dtype=complex)
    
    # MATLAB loop: for l = 0:L-1
    for l in range(L):
        # MATLAB indices: 1+l*hop : wlen+l*hop
        # Python indices: l*hop : wlen+l*hop
        start_idx = l * hop
        end_idx = start_idx + wlen
        
        # x_w = x(...) .* win
        x_w = x[start_idx:end_idx] * win
        
        # X = fft(x_w, nfft)
        X = np.fft.fft(x_w, nfft)
        
        # STFT(:, 1+l) = X(1:NUP)
        STFT[:, l] = X[:NUP]

    # t = (wlen/2:hop:wlen/2+(L-1)*hop)/fs
    # We construct the exact time points to match MATLAB
    t = (wlen/2 + np.arange(L) * hop) / fs
    
    # f = (0:NUP-1)*fs/nfft
    f = np.arange(NUP) * fs / nfft
    
    return STFT, f, t


def istft(stft_matrix, awin, swin, hop, nfft, fs):
    """
    Inverse Short-Time Fourier Transform.
    Translates the custom MATLAB implementation to Python.
    """
    # Ensure windows are flattened
    awin = np.array(awin).flatten()
    swin = np.array(swin).flatten()
    
    # L = size(stft, 2)
    L = stft_matrix.shape[1]
    
    # wlen = length(swin)
    wlen = len(swin)
    
    # xlen = wlen + (L-1)*hop
    xlen = wlen + (L - 1) * hop
    
    # x = zeros(1, xlen)
    x = np.zeros(xlen)

    # reconstruction of the whole spectrum
    # Python indexing notes: MATLAB 2:end is Python 1:end
    if nfft % 2 != 0:
        # odd nfft excludes Nyquist point
        # X = [stft; conj(flipud(stft(2:end, :)))]
        X = np.vstack([stft_matrix, np.conj(np.flipud(stft_matrix[1:, :]))])
    else:
        # even nfft includes Nyquist point
        # X = [stft; conj(flipud(stft(2:end-1, :)))]
        X = np.vstack([stft_matrix, np.conj(np.flipud(stft_matrix[1:-1, :]))])

    # columnwise IFFT on the STFT-matrix
    # xw = real(ifft(X))
    xw = np.real(np.fft.ifft(X, axis=0))
    
    # xw = xw(1:wlen, :)
    xw = xw[:wlen, :]

    # Weighted-OLA
    for l in range(L):
        # MATLAB indices: 1+(l-1)*hop : wlen+(l-1)*hop
        start_idx = l * hop
        end_idx = start_idx + wlen
        
        # x(...) = x(...) + (xw(:, l).*swin)'
        # Note: In Python, xw[:, l] is 1D, swin is 1D. We add element-wise.
        x[start_idx:end_idx] += xw[:, l] * swin

    # scaling of the signal
    # W0 = sum(awin.*swin)
    W0 = np.sum(awin * swin)
    
    # x = x.*hop/W0
    x = x * hop / W0

    # generation of the time vector
    # t = (0:xlen-1)/fs
    t = np.arange(xlen) / fs
    
    return x, t



def plot_stft(SIG, fs, R, K, L, title_suffix):
    T = np.arange(L) / fs * R
    F = np.arange(K) * fs / 2 / (K - 1)

    plt.figure()
    plt.imshow(20 * np.log10(np.abs(SIG) + np.finfo(float).eps), 
               aspect='auto', origin='lower', extent=[T[0], T[-1], F[0], F[-1]])
    plt.xlabel('Time[Sec]', fontsize=14)
    plt.ylabel('Frequency[Hz]', fontsize=14)
    plt.colorbar()
    plt.title(f'STFT {title_suffix}')
