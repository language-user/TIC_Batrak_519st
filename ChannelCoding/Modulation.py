from random import randint
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sin, cos, pi
import scipy


def plot(x, y, axis_x="", axis_y="", title=""):
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(axis_x, fontsize=14)
    ax.set_ylabel(axis_y, fontsize=14)
    plt.title(title, fontsize=14)
    isdir = os.path.isdir('./figures/')
    if not isdir:
        os.mkdir('./figures/')
    fig.savefig('./figures/' + title + '.png', dpi=600)
    plt.close(fig)


def spectrum(sequence):
    y_spectrum = np.abs(scipy.fft.fftshift(scipy.fft.fft(sequence)))
    x_spectrum = scipy.fft.fftshift(scipy.fft.fftfreq(len(sequence), 1 / len(sequence)))
    return x_spectrum[round(len(x_spectrum)/2):], y_spectrum[round(len(y_spectrum)/2):]


def create_sequence():
    sequence = np.zeros(1000)
    sequence[0:100] = randint(0, 1)
    sequence[100:200] = randint(0, 1)
    sequence[200:300] = randint(0, 1)
    sequence[300:400] = randint(0, 1)
    sequence[400:500] = randint(0, 1)
    sequence[500:600] = randint(0, 1)
    sequence[600:700] = randint(0, 1)
    sequence[700:800] = randint(0, 1)
    sequence[800:900] = randint(0, 1)
    sequence[900:999] = randint(0, 1)
    return sequence


def ask_modulation(frequency, sequence):
    sequence_ask = np.zeros(1000)
    for i in range(0, len(sequence)):
        sequence_ask[i] = sequence[i] * cos(2 * pi * frequency * i / 1000)
    return sequence_ask


def ask_demodulation(frequency, sequence):
    prod = np.zeros(1000)
    demod_signal = np.zeros(1000)
    threshold = np.ones(1000) * 25
    demodulated_sequence = np.zeros(1000)

    for i, value in enumerate(sequence):
        prod[i] = value * cos(2 * pi * frequency * i / 1000)

    for i in range(0, 10):
        S = 0
        for t in range(0, 100):
            S = S + prod[t + 100 * (i - 1)]
            demod_signal[t + 100 * (i - 1)] = S

    demodulated = 1 / 2 * (np.sign(demod_signal - threshold) + 1)

    for i in range(0, 10):
        for t in range(0, 100):
            demodulated_sequence[t + 100 * (i - 1)] = demodulated[100 * i - 1]

    return demod_signal, demodulated_sequence



def psk_modulation(frequency, sequence):
    modulated_sequence = np.zeros(1000)
    for i, value in enumerate(sequence):
        modulated_sequence[i] = sin(2 * pi * frequency * i/1000 + value * pi + pi)
    return modulated_sequence


def psk_demodulation(frequency, sequence):
    product = np.zeros(1000)
    demodulated_signal = np.zeros(1000)
    threshold = np.ones(1000) * 25
    demodulated_sequence = np.zeros(1000)

    for i, value in enumerate(sequence):
        product[i] = value * sin(2 * pi * frequency * i / 1000)

    for i in range(0, 10):
        S = 0
        for t in range(0, 100):
            S = S + product[t + 100 * (i - 1)]
            demodulated_signal[t + 100 * (i - 1)] = S

    demodulated = 1 / 2 * (np.sign(demodulated_signal - threshold) + 1)

    for i in range(0, 10):
        for t in range(0, 100):
            demodulated_sequence[t + 100 * (i - 1)] = demodulated[100 * i - 1]

    return demodulated_signal, demodulated_sequence


def fsk_modulation(frequency1, frequency2, sequence):
    modulated_sequence = np.zeros(1000)
    for i, value in enumerate(sequence):
        modulated_sequence[i] = value * sin(2 * pi * frequency1 * i / 1000) + (abs(value - 1)) * sin(2 * pi * frequency2 * i / 1000)
    return modulated_sequence


def fsk_demodulation(frequency1, frequency2, sequence):
    product1 = np.zeros(1000)
    product2 = np.zeros(1000)
    demodulated_signal1 = np.zeros(1000)
    demodulated_signal2 = np.zeros(1000)
    demodulated_sequence = np.zeros(1000)

    for i, value in enumerate(sequence):
        product1[i] = value * sin(2 * pi * frequency1 * i / 1000)
        product2[i] = value * sin(2 * pi * frequency2 * i / 1000)

    for i in range(0, 10):
        S1 = 0
        S2 = 0
        for t in range(0, 100):
            S1 = S1 + product1[t + 100 * (i - 1)]
            demodulated_signal1[t + 100 * (i - 1)] = S1
            S2 = S2 + product2[t + 100 * (i - 1)]
            demodulated_signal2[t + 100 * (i - 1)] = S2

    demodulated = 1 / 2 * (np.sign(demodulated_signal1 - demodulated_signal2) + 1)

    for i in range(0, 10):
        for t in range(0, 100):
            demodulated_sequence[t + 100 * (i - 1)] = demodulated[100 * i - 1]

    return demodulated_signal1, demodulated_signal2, demodulated_sequence



def create_noise(mean, standard_deviation, length):
    return np.random.normal(mean, standard_deviation, length)


def noise_stress(sequence, sequence_modulated, modulation, frequency):
    error_modulated = []
    demodulated_sequence = np.zeros(1000)

    for i in range(0, 20):
        p = 0

        for m in range(0, 200):
            noise = create_noise(0, 1, 1000)
            sequence_noise = sequence_modulated + i * noise

            if modulation == "ASK":
                demodulated_signal, demodulated_sequence = ask_demodulation(frequency[0], sequence_noise)
            elif modulation == "PSK":
                demodulated_signal, demodulated_sequence = psk_demodulation(frequency[0], sequence_noise)
            elif modulation == "FSK":
                demodulated_signal1, demodulated_signal2, demodulated_sequence = fsk_demodulation(frequency[0],
                                                                                                  frequency[1],
                                                                                                  sequence_noise)

            summa = abs(sum(sequence - demodulated_sequence))
            p += summa / 1000

        error_modulated.append(p / 200)

    return error_modulated


def main(ask, psk, fsk1, fsk2):
    sequence = create_sequence()
    x = np.arange(len(sequence)) / 1000
    plot(x, sequence, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Згенерована випадкова послідовність")
    sequence_ask = ask_modulation(ask, sequence)
    plot(x, sequence_ask, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Амплітудна модуляція")
    x_spectrum, spectrum_sequence_ask = spectrum(sequence_ask)
    plot(x_spectrum, spectrum_sequence_ask, axis_x="Частота, Гц", axis_y="Амплітуда спектру",
         title="Спектр при амплітудніймодуляції")
    sequence_psk = psk_modulation(psk, sequence)
    plot(x, sequence_psk, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Фазова модуляція")
    x_spectrum, spectrum_sequence_psk = spectrum(sequence_psk)
    plot(x_spectrum, spectrum_sequence_psk, axis_x="Частота, Гц", axis_y="Амплітуда спектру",
         title="Спектр при фазовій модуляції")
    sequence_fsk = fsk_modulation(fsk1, fsk2, sequence)
    plot(x, sequence_fsk, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Частотна модуляція")
    x_spectrum, spectrum_sequence_fsk = spectrum(sequence_fsk)
    plot(x_spectrum, spectrum_sequence_fsk, axis_x="Частота, Гц",
         axis_y="Амплітуда спектру", title="Спектр при частотній модуляціїмодуляції")
    noise = create_noise(0, 1, 1000)
    sequence_ask_noise = sequence_ask + noise
    sequence_psk_noise = sequence_psk + noise
    sequence_fsk_noise = sequence_fsk + noise
    plot(x, sequence_ask_noise, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Амплітудна модуляція з шумом")
    plot(x, sequence_psk_noise, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Фазова модуляція з шумом")
    plot(x, sequence_fsk_noise, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Частотна модуляція з шумом")
    ask_demodulated_signal, sequence_demodulated = ask_demodulation(ask, sequence_ask_noise)
    plot(x, ask_demodulated_signal, axis_x="Час, c", axis_y="Амплітудасигналу",
         title="Демодульований сигнал з амплітудною модуляцією")
    plot(x, sequence_demodulated, axis_x="Час, c", axis_y="Амплітуда сигналу",
         title="Демодульована послідовність з амплітудною модуляцією")
    psk_demodulated_signal, sequence_demodulated = psk_demodulation(psk, sequence_psk_noise)
    plot(x, psk_demodulated_signal, axis_x="Час, c", axis_y="Амплітуда сигналу",
         title="Демодульований сигнал з фазовою модуляцією")
    plot(x, sequence_demodulated, axis_x="Час, c", axis_y="Амплітуда сигналу",
         title="Демодульована послідовність з фазовою модуляцією")
    fsk_demodulated_signal1, fsk_demodulated_signal2, sequence_demodulated = fsk_demodulation(fsk1, fsk2, sequence_fsk_noise)
    plot(x, fsk_demodulated_signal1, axis_x="Час, c", axis_y="Амплітуда сигналу",
         title="Демодульований сигнал 1 з частотною модуляцією")
    plot(x, fsk_demodulated_signal2, axis_x="Час, c", axis_y="Амплітуда сигналу",
         title="Демодульований сигнал 2 з частотною модуляцією")
    plot(x, sequence_demodulated, axis_x="Час, c", axis_y="Амплітуда сигналу",
         title="Демодульована послідовність з частотною модуляцією")
    error_ask = noise_stress(sequence, sequence_ask, "ASK", [ask])
    error_psk = noise_stress(sequence, sequence_psk, "PSK", [psk])
    error_fsk = noise_stress(sequence, sequence_fsk, "FSK", [fsk1, fsk2])
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.plot(np.arange(0, 20), error_ask, linewidth=1)
    ax.plot(np.arange(0, 20), error_psk, linewidth=1)
    ax.plot(np.arange(0, 20), error_fsk, linewidth=1)
    ax.set_xlabel('Діапазон змін шуму', fontsize=14)
    ax.set_ylabel('Ймовірність помилки', fontsize=14)
    ax.legend(['ASK', 'PSK', 'FSK'], loc=2)
    plt.title('Оцінка завадостійкості трьох видів модуляції', fontsize=14)
    isdir = os.path.isdir('./figures/')
    if not isdir:
        os.mkdir('./figures/')
    fig.savefig('./figures/' + 'Оцінка завадостійкості трьох видів модуляції' + '.png', dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main(25, 25, 30, 15)