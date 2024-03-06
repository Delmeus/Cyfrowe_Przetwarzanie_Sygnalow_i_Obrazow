import numpy as np
import matplotlib.pyplot as plt
import os.path


def read_file(filename):
    matrix = np.loadtxt(filename)
    return matrix


def plot_small_ecg(matrix, frequency):
    freq_array = []
    freq_array.append(frequency)
    for i in range(1, len(matrix)):
        freq_array.append(freq_array[i - 1] + frequency)
    for measurement in matrix.transpose():
        plt.plot(freq_array, measurement)

    plt.xlabel('Time [s]')
    plt.ylabel('Value ')
    plt.show()


def plot_big_ecg(matrix, frequency):
    freq_array = []
    freq_array.append(frequency)
    for i in range(1, len(matrix)):
        freq_array.append(freq_array[i - 1] + frequency)
    plt.plot(freq_array, matrix)

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
    plt.plot(signal)
    plt.show()
    return signal


def plot_frequency_spectrum(signal, fs):
    n = len(signal)
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(n, 1/fs)

    plt.plot(freq[:n//2], np.abs(fft_result[:n//2]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum')
    plt.show()


if __name__ == '__main__':
    data = []
    filename = ''
    fs = 1/1000
    while True:
        print('Laboratorium 1 - EKG')
        print('1. Wczytaj plik\n2. Wyswietl plik\n3. Transformata Fouriera\n4. Odwrocona transformata Fouriera\n5. Widmo\n6. Zakoncz program')
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
                    fs = 1/1000
                    plot_small_ecg(data, fs)
                else:
                    fs = 1/360
                    plot_big_ecg(data, fs)
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
            fs = 1000  # Sampling frequency
            length = 256
            signal_50Hz = generate_sine_wave(50, length, fs)


            # Task 2: Plot frequency spectrum
            plot_frequency_spectrum(signal_50Hz, fs)

        elif answer == 6:
            break

