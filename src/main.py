import utils
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import square
import scipy.signal as signal
from scipy.fftpack import rfft, irfft
import argparse
import cv2



CC_THRESHOLD = 0.3


def sync_a(fs):
    f_a = 1040 # Wrong in document!!

    t_a = np.arange(0, (7/f_a), 1/fs)

    return square(2 * np.pi * f_a * t_a, duty=0.5)

def sync_b(fs):
    f_b = 837 # Wrong in document!!

    t_b = np.arange(0, (7/f_b), 1/fs)

    return square(2 * np.pi * f_b * t_b, duty=0.5)

def find_sync_time(x, fs, samples_per_line):
    offset = samples_per_line//2

    sa = sync_a(fs)
    sb = sync_b(fs)

    x = x > (x.max()/2)
    corr_a = signal.correlate(x, sa)
    corr_b = signal.correlate(x[offset:], sb)

    def process(xcorr):
        mid = len(xcorr)//2 + 1

        cc = np.concatenate((xcorr[-mid:], xcorr[:mid]))
        return cc, mid

    cc_a, mid_a = process(corr_a)
    cc_b, mid_b = process(corr_b)

    cc_a = np.abs(cc_a)[mid_a:]
    cc_b = np.abs(cc_b)[mid_b:]
    cc_a = cc_b[..., :cc_b.shape[-1]]
    cc = cc_a + cc_b

    # t1 = np.arange(0, len(cc_a)/fs, 1/fs)

    # plt.plot(t1, cc)
    # # plt.plot(t1, cc_a)
    # # t2 = np.arange(0, len(cc_b)/fs, 1/fs)
    # # plt.plot(t2, cc_b)
    # # plt.scatter(t[tau], cc_a[tau], c='green', marker='o')
    # plt.show()

    tau = np.argmax(cc)

    cc_threshold = (np.sum(sa == 1) + np.sum(sb == 1)) * CC_THRESHOLD

    if cc[tau] > cc_threshold:
        success = True
    else:
        success = False

    # peaks, props = signal.find_peaks(cc, distance=5000)
    # t = np.arange(0, len(cc)/fs, 1/fs)
    # plt.plot(t, cc)
    # plt.scatter(t[tau], cc[tau], c='green', marker='o')
    # plt.show()

    return tau, success

def main():
    fs, y = utils.read_wavfile('data/demo.wav')

    pixels_per_line = 2080
    if y.dtype == np.int16:
        y = y / (2 ** 15 - 1)
        assert np.abs(y).max() < 1

    if len(y.shape) == 2:
        y = y[:, 0]

    new_fs = ((fs - 1) // pixels_per_line + 1) * pixels_per_line
    y = utils.resample(y, fs, new_fs)
    fs = new_fs

    samples_per_line = fs//2
    samples_backoff = samples_per_line//4
    samples_per_pixel = samples_per_line//pixels_per_line

    y_envelope = utils.get_envelope(y)

    print("Finding first frame")
    success = False
    while not success and (len(y_envelope) > samples_per_line):
        tau, success = find_sync_time(y_envelope[:3*samples_per_line//2], fs, samples_per_line)
        y_envelope = y_envelope[samples_per_line:]

    print("First frame found!")
    current_index = tau
    image = []
    while y_envelope.shape[-1] - current_index > 2 * samples_per_line:
        sync_start_index = max([current_index - samples_backoff, 0])

        tau, success = find_sync_time(y_envelope[sync_start_index:sync_start_index + 3*samples_per_line//2], fs, samples_per_line)
        if success:
            current_index = sync_start_index + tau

        if y_envelope.shape[-1] - current_index < samples_per_line:
            break

        y_line = y_envelope[current_index:current_index+samples_per_line]
        y_line = y_line.reshape(pixels_per_line, samples_per_pixel)
        y_line = np.median(y_line, axis=1)
        image.append(y_line)

        current_index += samples_per_line

    # y_dem = y_dem[pixels_per_line:]
    image = np.array(image)
    image = utils.quantize_8bit(image)
    equ = cv2.equalizeHist(image)
    # cv2.imwrite('output.png', equ)
    # plt.hist(image)

    # plt.plot(y_dem)
    plt.imshow(equ, cmap='gray', vmin=0, vmax=255)
    plt.show()

if __name__ == "__main__":
    main()