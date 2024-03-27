import numpy as np
import matplotlib.pyplot as plt
import os.path

from scipy import signal
from scipy.signal import butter, lfilter


def read_file(filename):
    matrix = np.loadtxt(filename)
    return matrix


def plot_ecg(matrix, frequency):
    t = np.arange(0, len(matrix)) / frequency

    plt.plot(t, matrix)
    plt.xlabel('Time [s]')
    plt.ylabel('Value ')
    plt.show()


def plot_ecg_time(matrix):

    plt.plot(matrix.transpose()[0], matrix.transpose()[1])

    plt.xlabel('Time [ms]')
    plt.ylabel('Value ')
    plt.show()


def generate_sine_wave(freq, length, fs):
    t = np.arange(0, length/fs, 1/fs)
    signal = np.sin(2 * np.pi * freq * t)
    return signal


def plot_frequency_spectrum(signal, fs):
    n = len(signal)
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(n, 1/fs)

    return freq[:n//2], np.abs(fft_result[:n//2])


def plot_inverse_fft(fft_result, fs, title, limit=0):
    inv_fft_result = np.fft.ifft(fft_result)
    time = np.arange(0, len(inv_fft_result)) / fs

    if limit != 0:
        plt.plot(time[:limit], np.real(inv_fft_result)[:limit])
    else:
        plt.plot(time, np.real(inv_fft_result))
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.show()
    return np.real(inv_fft_result)


def butter_lowpass(cutoff, fs, data,order=10):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)


def butter_highpass(cutoff, fs, data, order=10):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, data)


def plot_filtered_data(unfiltered_data, filtered_data, type):
    figure, axis = plt.subplots(2, 2, figsize=(10, 8))
    axis[0, 0].plot(unfiltered_data[0], filtered_data)
    axis[0, 0].set_title(f'Filtered with {type}  EKG Signal')

    axis[0, 1].plot(unfiltered_data[0], unfiltered_data[1] - filtered_data)
    axis[0, 1].set_title(f'Difference between original and filtered {type}')

    result = plot_frequency_spectrum(filtered_data, sampling_freq)
    axis[1, 0].plot(result[0], result[1])
    axis[1, 0].set_title('Filtered data Spectrum')

    result = plot_frequency_spectrum(unfiltered_data[1] - filtered_data, sampling_freq)
    axis[1, 1].plot(result[0], result[1])
    axis[1, 1].set_title('Difference Spectrum')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data = []
    filename = ''
    sampling_freq = 1000
    while True:
        print('Laboratorium 1 - EKG')
        print('1. Wczytaj plik\n2. Wyswietl plik\n3. Transformata Fouriera - zadanie 2\n'
              '4. Widmo - zadanie 3\n5. Filtracja - zadanie 4'
              '\n6. Zakoncz program')
        answer = int(input('Podaj opcje = '))
        if answer == 1:
            filename = input('Podaj nazwe pliku = ')
            if os.path.isfile(filename):
                data = read_file(filename)
            else:
                print('Nie ma takiego pliku')
        elif answer == 2:
            if filename == 'ekg1.txt' or filename == 'ekg100.txt':
                if filename == 'ekg1.txt':
                    sampling_freq = 1000
                    plot_ecg(data, sampling_freq)
                else:
                    sampling_freq = 360
                    plot_ecg(data, sampling_freq)
            elif filename == 'ekg_noise.txt':
                plot_ecg_time(data)
            else:
                print("Brak wczytanego pliku!")
        elif answer == 3:
            sampling_freq = 1000
            length = 65536
            signal_50Hz = generate_sine_wave(50, length, sampling_freq)
            signal_60Hz = generate_sine_wave(60, length, sampling_freq)

            plt.plot((signal_50Hz)[:500])
            plt.title('Sine 50Hz')
            plt.show()

            result = plot_frequency_spectrum(signal_50Hz, sampling_freq)
            plt.plot(result[0], result[1])
            plt.title('Frequency Spectrum 50Hz')
            plt.show()

            plt.plot((signal_50Hz + signal_60Hz)[:500])
            plt.title('Sine 50Hz + 60Hz')
            plt.show()

            result = plot_frequency_spectrum(signal_50Hz + signal_60Hz, sampling_freq)
            plt.plot(result[0], result[1])
            plt.title('Frequency Spectrum 50Hz + 60Hz')
            plt.show()

            inv_result = plot_inverse_fft(np.fft.fft(signal_50Hz + signal_60Hz), sampling_freq, "Inverted Fourier transform", 500)
            diff = signal_50Hz + signal_60Hz - inv_result
            plt.plot(diff[:500])
            plt.title("Signal diff")
            plt.show()

        elif answer == 4:
            sampling_freq = 1000
            ecg_signal = np.loadtxt('ekg100.txt')

            plt.plot(ecg_signal[:5000])
            plt.title("ekg100 signal")
            plt.show()

            result = plot_frequency_spectrum(ecg_signal, sampling_freq)##
            plt.plot(result[0][:5000], result[1][:5000])
            plt.title('ekg100 Fourier transform')
            plt.show()

            inv_result = plot_inverse_fft(np.fft.fft(ecg_signal), sampling_freq, "Inverted Fourier transform", 5000)

            diff = ecg_signal - inv_result
            plt.plot(diff[:500])
            plt.title("Signal diff")
            plt.show()

        elif answer == 5:
            data = np.loadtxt('ekg_noise.txt').transpose()
            sampling_freq = 360

            plt.plot(data[0], data[1])
            plt.title('Original ECG Signal')
            plt.show()

            result = plot_frequency_spectrum(data[1], sampling_freq)
            plt.plot(result[0], result[1])
            plt.title('Original ECG Spectrum')
            plt.show()

            cutoff_freq_lp = 60
            cutoff_freq_hp = 5

            data_lp = butter_lowpass(cutoff_freq_lp, sampling_freq, data[1])
            plot_filtered_data(data, data_lp, "lowpass")

            data_hp = butter_highpass(cutoff_freq_hp, sampling_freq, data_lp)
            plot_filtered_data(data, data_hp, "highpass")
            pass
        elif answer == 6:
            break

