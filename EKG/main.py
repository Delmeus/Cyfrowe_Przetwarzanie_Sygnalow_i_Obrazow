import numpy as np
import matplotlib.pyplot as plt
import os.path

from scipy import signal


def read_file(filename):
    matrix = np.loadtxt(filename)
    return matrix


def plot_ecg(matrix, frequency):
    t = np.arange(0, len(matrix)) / frequency

    plt.plot(t, matrix)
    plt.xlabel('Time [s]')
    plt.ylabel('Value ')
    plt.show()


# def plot_big_ecg(matrix, frequency):
#     time_values = np.arange(0, len(matrix) * frequency, frequency)
#     plt.plot(time_values, matrix)
#
#     plt.xlabel('Time [s]')
#     plt.ylabel('Value')
#     plt.show()


def plot_ecg_time(matrix):

    plt.plot(matrix.transpose()[0], matrix.transpose()[1])

    plt.xlabel('Time [ms]')
    plt.ylabel('Value ')
    plt.show()


def generate_sine_wave(freq, length, fs):
    t = np.arange(0, length/fs, 1/fs)
    signal = np.sin(2 * np.pi * freq * t)
    return signal


def plot_frequency_spectrum(signal, fs, title):
    n = len(signal)
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(n, 1/fs)

    plt.plot(freq[:n//2], np.abs(fft_result[:n//2]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.show()


def plot_inverse_fft(fft_result, fs, title):
    inv_fft_result = np.fft.ifft(fft_result)
    time = np.arange(0, len(inv_fft_result)) / fs

    plt.plot(time, np.real(inv_fft_result))
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.show()
    return np.real(inv_fft_result)


def apply_filter(data, fs, cutoff_freq, filter_type):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b = signal.butter(4, normal_cutoff, btype=filter_type, analog=False, output='ba')
    print(len(b))
    return signal.filtfilt(b[0], b[1], data.transpose())


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
            # t = np.arange(65536)
            # sp = np.fft.fft(np.sin(t))
            # freq = np.fft.fftfreq(t.shape[-1])
            # plt.plot(freq, sp.real, freq, sp.imag)
            # plt.show()
            sampling_freq = 1000
            length = 256
            signal_50Hz = generate_sine_wave(50, length, sampling_freq)
            signal_60Hz = generate_sine_wave(60, length, sampling_freq)

            plt.plot(signal_50Hz)
            plt.title('Sine 50Hz')
            plt.show()
            plot_frequency_spectrum(signal_50Hz, sampling_freq, 'Frequency Spectrum 50Hz')

            plt.plot(signal_50Hz + signal_60Hz)
            plt.title('Sine 50Hz + 60Hz')
            plt.show()
            plot_frequency_spectrum(signal_50Hz + signal_60Hz, sampling_freq, 'Frequency Spectrum 50Hz + 60Hz')

            inv_result = plot_inverse_fft(np.fft.fft(signal_50Hz + signal_60Hz), sampling_freq, "Inverted Fourier transform")
            diff = signal_50Hz + signal_60Hz - inv_result
            plt.plot(diff)
            plt.title("Signal diff")
            plt.show()
        elif answer == 4:
            sampling_freq = 1000
            ecg_signal = np.loadtxt('ekg100.txt')
            plt.plot(ecg_signal)
            plt.show()
            plot_frequency_spectrum(ecg_signal, sampling_freq, 'ekg100 Fourier transform')
            inv_result = plot_inverse_fft(np.fft.fft(ecg_signal), sampling_freq, "Inverted Fourier transform")

            diff = ecg_signal - inv_result
            plt.plot(diff)
            plt.title("Signal diff")
            plt.show()
        elif answer == 5:
            data = np.loadtxt('ekg_noise.txt')
            sampling_freq = 360
            # plot_ecg_time(data)
            plot_frequency_spectrum(data,sampling_freq, "Unfiltered Fourier transform")
            cutoff_freq_lp = 60
            filtered_data_lp = apply_filter(data, sampling_freq, cutoff_freq_lp, 'low')

            plot_frequency_spectrum(filtered_data_lp, sampling_freq, 'Filtered EKG (Low-Pass) Frequency Spectrum')

            plt.plot(data, label='Original EKG Signal')
            plt.show()
            plt.plot(filtered_data_lp, label='Filtered EKG Signal')
            plt.show()
            #sos = apply_filter(data, 360, 60, 'low')
            #sos = signal.butter(10, 60, 'hp', fs=360, output='sos')
            #filtered_data = signal.sosfilt(sos, data)
            #plot_ecg_time(filtered_data)
            pass
        elif answer == 6:
            break

