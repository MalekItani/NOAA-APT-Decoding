from scipy.io.wavfile import read, write
from scipy.fft import rfft, irfft
import numpy as np


def bpf(x, fl, fh, fs):
    """
    Applies bandpass filter to signal sampled at fs between fl and fh
    """
    n = x.shape[-1]
    X = rfft(x)

    # Should these be divided by 2?
    F_L = n * fl // fs
    F_H = 1 + (n * fh - 1) // fs

    X[:F_L] = 0
    X[F_H:] = 0
    
    y = irfft(X)
    return y

def demodulate_am(x, fc, fs):
    """
    Demodulates AM signal
    Eventually, do corrections (sample drops, doppler, ...)
    """
    t = np.arange(0, len(x)/fs, 1/fs)

    fc = fc # TODO: Fancy math on fc to correct for effects

    carrier = np.cos(2 * np.pi * fc * t)

    y = x * carrier
    
    # Cancel 2fc oscillations with lowpass @ fc
    return bpf(y, 0, fc, fs) * 2


def read_wavfile(path: str):
    """
    Reads path to .wav file and returns data as a tuple [sr, data]
    """
    return read(path)

def write_wavfile(path: str, data: np.ndarray, sr: int):
    """
    Writes data as .wav file.
    Data should be (C x T).
    """
    return write(path, sr, data.T)

