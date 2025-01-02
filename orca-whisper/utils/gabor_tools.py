import numpy as np
import soundfile as sf

def get_adjusted_signal_length(signal_length, n_fft, hop_length):
    lcm = np.lcm(n_fft, hop_length)
    # print(f"LCM({n_fft}, {hop_length}) = {lcm}")
    new_length = lcm
    while new_length < signal_length:
        new_length += lcm
    # print(f"Adjusted Signal Length: {new_length}")
    # print(f"Signal Difference: {new_length - signal_length}")
    return new_length


def get_signal_0_padding(signal_length, n_fft, hop_length):
    difference = get_adjusted_signal_length(signal_length, n_fft, hop_length) - signal_length
    # print(f"Signal Difference: {difference}")
    return difference


def get_spectrogram_size_from_signal_length(signal_length, n_fft, hop_length):
    new_length = get_adjusted_signal_length(signal_length, n_fft, hop_length)
    return (n_fft // 2) + 1, new_length // hop_length

def get_time_bins(signal_length, n_fft, hop_length):
    return get_adjusted_signal_length(signal_length, n_fft, hop_length) // hop_length


def get_spectrogram_size_from_signal(signal, n_fft, hop_length):
    return get_spectrogram_size_from_signal_length(len(signal), n_fft, hop_length)


def get_hop_for_time_steps(signal_length, n_fft, time_steps, min_hop=2, max_hop=1024):
    padding = np.float('inf')
    hop = min_hop
    for i in range(min_hop, max_hop):
        if get_time_bins(signal_length, n_fft, i) == time_steps:
            adjusted_padding = get_signal_0_padding(signal_length, n_fft, i)
            if adjusted_padding < padding:
                hop = i
                padding = adjusted_padding

    return hop

if __name__ == '__main__':
    s_len = int(44100 * 0.325)
    _n_fft = 1024
    _hop = 260
    print(f"Spectrogram size with NFFT = {_n_fft} and Hop Length = {_hop} : {get_spectrogram_size_from_signal_length(s_len, _n_fft, _hop)}")

    print(get_hop_for_time_steps(signal_length=s_len, n_fft=_n_fft, time_steps=256))
