import utils



def main():
    fs, y = utils.read_wavfile('data/demo.wav')

    y_filt = utils.bpf(y, 0, fs/2, fs)
    y_dem = utils.demodulate_am(y_filt)

    utils.write_wavfile('output.wav', y_dem, fs)



if __name__ == "__main__":
    main()