import utils
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import square
import scipy.signal as signal
from scipy.fftpack import rfft, irfft
import argparse
import cv2
import os
import glob
from scipy.ndimage import gaussian_filter1d
import librosa






SYNC_AB_MAX_DRIFT_PER_FRAME = 5
START_FRAME_CC_THRESHOLD = 0.3
CC_THRESHOLD = 0.3
CC_THRESHOLD_NEW = 0.1


def sync_a(fs):
    f_a = 1040 # Wrong in document!!

    t_a = np.arange(0, (7/f_a), 1/fs)

    return square(2 * np.pi * f_a * t_a, duty=0.5)

def sync_b(fs):
    f_b = 837 # Wrong in document!!

    t_b = np.arange(0, (7/f_b), 1/fs)

    return square(2 * np.pi * f_b * t_b, duty=0.5)

def cross_correlate_and_order(sig, sync_signal):
    xcorr = signal.correlate(sig, sync_signal)

    mid = len(xcorr)//2 + 1

    cc = np.concatenate((xcorr[-mid:], xcorr[:mid]))
    return np.abs(cc[mid+(sync_signal.shape[0]-1):])

def sync_found(x, fs, samples_per_line, debug=False):
    offset = samples_per_line//2

    sa = sync_a(fs)
    sb = sync_b(fs)

    x = x > (x.max()/2)

    cc_a = cross_correlate_and_order(x, sa)
    cc_b = cross_correlate_and_order(x[offset:], sb)

    cc_a = cc_a[..., :cc_b.shape[-1]]
    cc = cc_a + cc_b

    if debug:
        t = np.arange(0, len(cc_a)/fs, 1/fs)
        plt.plot(t, cc_a)
        plt.plot(t, cc_b)
        plt.show()

    tau = np.argmax(cc)

    cc_threshold = (np.sum(sa == 1) + np.sum(sb == 1)) * START_FRAME_CC_THRESHOLD

    return tau, cc[tau] > cc_threshold

def find_sync_time(x, fs, samples_per_line):
    offset = samples_per_line//2

    sa = sync_a(fs)
    sb = sync_b(fs)

    x = x > (x.max()/2)

    cc_a = cross_correlate_and_order(x, sa)
    cc_b = cross_correlate_and_order(x[offset:], sb)

    cc_a = cc_a[..., :cc_b.shape[-1]]
    cc = cc_a + cc_b

    cc_a_inv = cross_correlate_and_order(x[offset:], sa)
    cc_b_inv = cross_correlate_and_order(x, sb)

    cc_b_inv = cc_b_inv[..., :cc_a_inv.shape[-1]]
    cc_inv = cc_a_inv + cc_b_inv

    add_offset = 0

    if np.max(cc_inv) > np.max(cc):
        cc = cc_inv
        add_offset = offset

    # t = np.arange(0, len(cc_a)/fs, 1/fs)
    # plt.plot(t, cc_a)
    # plt.plot(t, cc_b)
    # plt.plot(t, cc)
    # plt.show()

    tau = np.argmax(cc)

    cc_threshold = (np.sum(sa == 1) + np.sum(sb == 1)) * CC_THRESHOLD

    if cc[tau] > cc_threshold:
        success = True
    else:
        success = False

    return tau + add_offset, success

def noaa_decoder(path):
    fs, y = utils.read_wavfile(path)

    pixels_per_line = 2080
    if y.dtype == np.int16:
        y = y / (2 ** 15 - 1)
        assert np.abs(y).max() < 1

    if len(y.shape) == 2:
        y = y[:, 0]

    if (fs % pixels_per_line) > 0:
        print("Sampling rate incompatible, resampling...")
        new_fs = 12480#((fs - 1) // pixels_per_line + 1) * pixels_per_line
        # y = utils.resample(y, fs, new_fs)
        y = librosa.resample(y, fs, new_fs)
        fs = new_fs
        print("Done")

    samples_per_line = fs//2
    samples_backoff = samples_per_line //4
    samples_per_pixel = samples_per_line//pixels_per_line

    print("Computing envelope...")
    y_envelope = utils.get_envelope(y)
    print("Done")

    current_index = 0

    print("Finding first frame")
    success = False
    while (current_index + 3 * samples_per_line // 2 < len(y_envelope)):
        update, success = \
            sync_found(y_envelope[current_index:current_index + 3*samples_per_line//2],
                    fs,
                    samples_per_line)
        if not success:
            current_index += samples_per_line
        else:
            # sync_found(y_envelope[current_index:current_index + 3*samples_per_line//2],
            #     fs,
            #     samples_per_line, debug=True)
            current_index += update
            break

    if success is False:
        print("Could not find a frame. Recording may be too noisy, quitting...")

    print("Done")

    # utils.write_wavfile('output/syncd.wav', y_envelope[current_index:current_index+samples_per_line], fs)
    # test = y_envelope[current_index:current_index + samples_per_line]
    # plt.plot(test/test.max())
    # plt.show()
    # quit()

    image = []
    while y_envelope.shape[-1] - current_index > 2 * samples_per_line:
        sync_start_index = max([current_index - samples_backoff, 0])
        tau, success = find_sync_time(y_envelope[sync_start_index:sync_start_index + 3 * samples_per_line//2], fs, samples_per_line)
        if success:
            current_index = sync_start_index + tau

        if y_envelope.shape[-1] - current_index < samples_per_line:
            break

        y_line = y_envelope[current_index:current_index+samples_per_line]
        current_index += samples_per_line

        y_line = y_line.reshape(pixels_per_line, samples_per_pixel)
        y_line = np.mean(y_line, axis=1)
        image.append(y_line)

    image = np.array(image)
    image = utils.quantize_8bit(image)
    equ = cv2.equalizeHist(image)

    return equ

def main(args):
    # target_dir = Path(args.path)

    if not os.path.exists(args.path):
        print("The target directory doesn't exist")
        raise SystemExit(1)

    for entry in glob.glob(os.path.join(args.path, '*')):
        print(entry)
        decoded_image = noaa_decoder(entry)
        # plt.imshow(decoded_image, cmap='gray', vmin=0, vmax=255)
        path = os.path.join('output', f'{os.path.basename(entry)[:-4]}.png')
        cv2.imwrite(path, decoded_image)
        # cv2.imwrite('output.png', decoded_image)
        print('Image exported to: ' + path)
        print()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    main(args)