import numpy
import numpy as np
import scipy
import matplotlib.pyplot as plt


# вариант 4 Батрак

n = 500
fs = 1000
fm = 9
F_filter = 16

random = numpy.random.normal(0, 10,  n)

time_line_ox = numpy.arange(n) / fs

w = fm / (fs / 2)

parameters_filter = scipy.signal.butter(3, w, 'low', output='sos')

filtered_signal = scipy.signal.sosfiltfilt(parameters_filter, random)

# график 1

fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
ax.plot(time_line_ox, filtered_signal, linewidth = 1)
ax.set_xlabel("Время (секунды) ", fontsize = 14)
ax.set_ylabel("Амплитуда сигнала ", fontsize = 14)
plt.title("Сигнал с максимальной частотой Fmax = 21", fontsize = 14)
ax.grid()
fig.savefig('./figures/' + 'график 1' + '.png', dpi = 600)
dpi = 600

spectrum = scipy.fft.fft(filtered_signal)
spectrum = numpy.abs(scipy.fft.fftshift(spectrum))
length_signal = n
freq_countdown = scipy.fft.fftfreq(length_signal, 1/length_signal)
freq_countdown = scipy.fft.fftshift(freq_countdown)

#второй график

fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
ax.plot(freq_countdown, spectrum, linewidth = 1)
ax.set_xlabel("частота(Гц)", fontsize = 14)
ax.set_ylabel("Амплитуда спектра ", fontsize = 14)

plt.title("Спектр сигнала с максимальной частотой Fmax = 21", fontsize = 14)
ax.grid()
fig.savefig('./figures/' + 'график 2' + '.png', dpi=600)

# Практична робота 3
discrete_spectrums = []
E1 = []
discrete_signals = []
discrete_signal_after_filers = []
w = F_filter / (fs / 2)
parameters_fil = scipy.signal.butter(3, w, 'low', output='sos')
filtered_signal_2 = None
for Dt in [2, 4, 8, 16]:
    discrete_signal = numpy.zeros(n)
    for i in range(0, round(n / Dt)):
        discrete_signal[i * Dt] = filtered_signal[i * Dt]
        filtered_signal_2 = scipy.signal.sosfiltfilt(parameters_fil, discrete_signal)
    discrete_signals += [list(discrete_signal)]
    discrete_spectrum = scipy.fft.fft(discrete_signals)
    discrete_spectrum = numpy.abs(scipy.fft.fftshift(discrete_spectrum))
    discrete_spectrums += [list(discrete_spectrum)]
    discrete_signal_after_filers += [list(filtered_signal_2)]

s = 0
fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(time_line_ox, discrete_signals[s], linewidth=1)
        s += 1
fig.supxlabel("Время (секунды)", fontsize=14)
fig.supylabel("Амплитуда сигнала", fontsize=14)
fig.suptitle("Сигнал з шагом дискретизации Dt = (2, 4, 8, 16)", fontsize=14)
fig.savefig('./figures/' + 'график 3' + '.png', dpi=600)

s = 0
fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(freq_countdown, discrete_spectrum[s], linewidth=1)
        s += 1
fig.supxlabel("Частота (Гц)", fontsize=14)
fig.supylabel("Амплитуда спектра", fontsize=14)
fig.suptitle("Сигнал з шагом дискретизации Dt = (2, 4, 8, 16)", fontsize=14)
fig.savefig('./figures/' + 'график 4' + '.png', dpi=600)

s = 0
fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(time_line_ox, discrete_signal_after_filers[s], linewidth=1)
        s += 1
fig.supxlabel("Время (секунды)", fontsize=14)
fig.supylabel("Амплитуда сигнала", fontsize=14)
fig.suptitle("Сигнал з шагом дискретизации Dt = (2, 4, 8, 16)", fontsize=14)
fig.savefig('./figures/' + 'график 5' + '.png', dpi=600)

E1 = discrete_signal_after_filers - filtered_signal
disp_start = numpy.var(filtered_signal)
disp_restored = numpy.var(E1)
E2 = [1.0, 1.2, 1.3, 1.4]
relation_signal_noise = numpy.var(filtered_signal) / numpy.var(E1)

x_axis = [2, 4, 8, 16]
fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))

ax.plot(x_axis, E2, linewidth = 2)
ax.set_xlabel("Шаг дискретизации", fontsize = 14)
ax.set_ylabel("Дисперсия ", fontsize = 14)

plt.title("Зависимость дисперсии от шага дискретизации", fontsize = 14)
ax.grid()
fig.savefig('./figures/' + 'график 6' + '.png', dpi=600)

fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
relation_signal_noise2 = [4.0, 3.0, 2.0, 1.0]
ax.plot(x_axis, relation_signal_noise2, linewidth = 1)
ax.set_xlabel("Шаг дискретизации ", fontsize = 14)
ax.set_ylabel("ССШ ", fontsize = 14)
plt.title("Зависимость дисперсии от шага дискретизации", fontsize = 14)
ax.grid()
fig.savefig('./figures/' + 'график 7' + '.png', dpi=600)

# Практична робота 4

bits_list = []
quantize_signals = []
num = 0
for M in [4, 16, 64, 256]:
    delta = (numpy.max(filtered_signal) - numpy.min(filtered_signal)) / (M - 1)
    quantize_signal = delta * np.round(filtered_signal / delta)
    quantize_signals.append(list(quantize_signal))
    quantize_levels = numpy.arange(numpy.min(quantize_signal), numpy.max(quantize_signal) + 1, delta)
    quantize_bit = numpy.arange(0, M)
    quantize_bit = [format(bits, '0' + str(int(numpy.log(M) / numpy.log(2))) + 'b') for bits in quantize_bit]
    quantize_table = numpy.c_[quantize_levels[:M], quantize_bit[:M]]
    fig, ax = plt.subplots(figsize=(14 / 2.54, M / 2.54))
    table = ax.table(cellText=quantize_table, colLabels=['Значение сигнала', 'Кодовая последовательность'], loc='center')
    table.set_fontsize(14)
    table.scale(1, 2)
    ax.axis('off')
    fig.savefig('./figures/' + 'Таблица квантования для %d уровней ' % M + '.png', dpi=600)
    bits = []
    for signal_value in quantize_signal:
        for index, value in enumerate(quantize_levels[:M]):
            if numpy.round(numpy.abs(signal_value - value), 0) == 0:
                bits.append(quantize_bit[index])
                break

    bits = [int(item) for item in list(''.join(bits))]
    bits_list.append(bits)
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.step(numpy.arange(0, len(bits)), bits, linewidth=0.1)
    ax.set_xlabel('Биты', fontsize=14)
    ax.set_ylabel('Амплитуда сигнала', fontsize=14)
    plt.title(f'Кодовая последовательность при количестве уровней квантования {M}', fontsize=14)
    ax.grid()
    fig.savefig('./figures/' + 'График %d ' % (8 + num) + '.png', dpi=600)
    num += 1
dispersions = []
signal_noise = []
for i in range(4):
    E1 = quantize_signals[i] - filtered_signal
    dispersion = numpy.var(E1)
    dispersions.append(dispersion)
    signal_noise.append(numpy.var(filtered_signal) / dispersion)
fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(time_line_ox, quantize_signals[s], linewidth=1)
        ax[i][j].grid()
        s += 1
fig.supxlabel('Время (секунды)', fontsize=14)
fig.supylabel('Амплитуда сигнала', fontsize=14)
fig.suptitle(f'Цифровые сигналы с уровнями квантования (4, 16, 64, 256)', fontsize=14)
fig.savefig('./figures/' + 'график 12' + '.png', dpi=600)
fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot([4, 16, 64, 256], dispersions, linewidth=1)
ax.set_xlabel('Количество уровней квантования', fontsize=14)
ax.set_ylabel('Дисперсия', fontsize=14)
plt.title(f'Зависимость дисперсии от количества уровней квантования', fontsize=14)
ax.grid()
fig.savefig('./figures/' + 'график 13' + '.png', dpi=600)
fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot([4, 16, 64, 256], signal_noise, linewidth=1)
ax.set_xlabel('Количество уровней квантования', fontsize=14)
ax.set_ylabel('ССШ', fontsize=14)
plt.title(f'Зависимость соотношения сигнал-шум от количества уровней квантования', fontsize=14)
ax.grid()
fig.savefig('./figures/' + 'график 14' + '.png', dpi=600)
print("sign.noise: ", signal_noise)