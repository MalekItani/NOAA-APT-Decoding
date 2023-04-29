import utils
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import square
import scipy.signal as signal
from scipy.fftpack import rfft, irfft


def sync_a(fs):
    f_a = 1040 # Wrong in document!!
    
    t_a = np.arange(0, (7/f_a), 1/fs)

    return square(2 * np.pi * f_a * t_a, duty=0.5)# * carrier

def sync_b(fs):
    f_b = 837 # Wrong in document!!

    t_b = np.arange(0, (7/f_b), 1/fs)
    
    return square(2 * np.pi * f_b * t_b, duty=0.5)# * carrier

def find_sync_time(x, fc, fs, t_max=None):    
    s = sync_a(fs)
    
    n = max(len(s), len(x))
    X = rfft(x, n = n)
    Y = rfft(s, n = n)

    R = (X * np.conj(Y))
    R = R / (np.abs(R))
    
    corr = irfft(R)
    
    # corr = np.correlate(x/x.max(), s)

    mid = len(corr)//2 + 1
        
    cc = np.concatenate((corr[-mid:], corr[:mid]))

    if t_max is not None:
        cc = np.concatenate([cc[-t_max+1:], cc[:t_max+1]])
    else:
        t_max = mid

    cc = np.abs(cc)[t_max:]
    
    # peaks, props = signal.find_peaks(cc, distance=5000)
    # t = np.arange(0, len(cc)/fs, 1/fs)
    # plt.plot(t, cc)
    # plt.scatter(t[peaks], cc[peaks], c='green', marker='o')
    # plt.show()
    
    tau = np.argmax(cc[:t_max])

    return tau

def main():    
    fs, y = utils.read_wavfile('data/demo.wav')
    
    pixels_per_line = 2080
    fc = 2400
    if y.dtype == np.int16:
        y = y / (2 ** 15 - 1)
        assert np.abs(y).max() < 1

    new_fs = ((fs - 1) // pixels_per_line + 1) * pixels_per_line
    y = utils.resample(y, fs, new_fs)
    fs = new_fs

    samples_per_line = fs//2
    samples_backoff = samples_per_line//2
    samples_per_pixel = samples_per_line//pixels_per_line
    print('sample/line', samples_per_line)
    print('sample/pixel', samples_per_pixel)

    y_filt = y#utils.bpf(y, 0, fs/2, fs)
    print(y_filt.shape)
    y_filt = y_filt[655157:]

    y_envelope = utils.get_envelope(y_filt)
    # print(y_envelope.shape)
    # y_envelope = signal.medfilt(y_envelope, 5)
    # y_envelope = y_envelope.reshape(len(y_envelope) // 5, 5)[:, 2]
    utils.write_wavfile('output/envelope.wav', y_envelope, fs)
    # y_envelope = utils.quantize_8bit(y_envelope)
    
    tau = find_sync_time(y_envelope[:samples_per_line], fc, fs)
    y_envelope = y_envelope[tau:]
    

    image = []
    # while len(image) < 128:
    while True:
        tau = find_sync_time(y_envelope[:pixels_per_line], fc, fs)
        y_envelope = y_envelope[tau:]
        
        print('Remaining:', len(y_envelope))
        if len(y_envelope) < samples_per_line:
            break
        
        y_line = y_envelope[:samples_per_line]
        y_line = y_line.reshape(samples_per_pixel, pixels_per_line)
        y_line = y_line[samples_per_pixel//2, :]
        image.append(y_line)

        y_envelope = y_envelope[samples_per_line:]

        # y_dem = utils.demodulate_am(y_line, fc, fs)
        # y_dec = utils.decode_apt(y_dem)
        
    # utils.write_wavfile('output/output.wav', y_dem, fs)

    # y_dem = y_dem[pixels_per_line:]
    image = np.array(image)
    image = utils.quantize_8bit(image)
    # plt.hist(image)

    # plt.plot(y_dem)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()

if __name__ == "__main__":
    main()