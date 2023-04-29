from scipy.io.wavfile import read, write
from scipy.fft import rfft, irfft
import numpy as np
import scipy.signal as signal


def resample(y, fs_old, fs_new):
    T = y.shape[-1] / fs_old
    N_new = int(T * fs_new)
    return signal.resample(y, N_new)

def quantize_8bit(x: np.ndarray) -> np.ndarray:
    low, high = x.min(), x.max()
    x = 255 * (x - low) / (high - low)
    return x.astype(np.uint8)

def bpf(x, fl, fh, fs):
    """
    Applies bandpass filter to signal sampled at fs between fl and fh
    """
    n = x.shape[-1]
    X = rfft(x)

    # Should these be divided by 2?
    F_L = int(n * fl // fs)
    F_H = int(1 + (n * fh - 1) // fs)
    
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
    return bpf(y, 10, fc, fs)
    # return y

def get_envelope(y_filt):
    return np.abs(signal.hilbert(y_filt))

def decode_apt(x_line: np.ndarray) -> np.ndarray:
    assert x_line.shape == (2080)
    # Define lengths
    sync_a_length = 39
    space_a_length = 47
    image_a_length = 909
    telemetry_a_length = 45
    
    sync_b_length = 39
    space_b_length = 47
    image_b_length = 909
    telemetry_b_length = 45

    # Define order
    sync_a_idx = 0
    space_a_idx = sync_a_idx + sync_a_length
    image_a_idx = space_a_idx + space_a_length
    telemetry_a_idx = image_a_idx + image_a_length

    sync_b_idx = telemetry_a_idx + telemetry_a_length
    space_b_idx = sync_b_idx + sync_b_length
    image_b_idx = space_b_idx + space_b_length
    telemetry_b_idx = image_b_idx + image_b_length

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

