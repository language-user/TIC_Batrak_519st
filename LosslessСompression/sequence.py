import random
import string
import collections
import math

from matplotlib import pyplot as plt


def lossless_comp():

    global aray

    def parameters(s):
        counts = collections.Counter(s)
        probability = {symbol: count / N_sequence for symbol, count in counts.items()}
        probability_string = ", ".join(
            [f"{symbol}={prob:.4f}" for symbol, prob in probability.items()]
        )
        main_probability = sum(probability.values()) / len(probability)
        equality = all(
            abs(prob - main_probability) < 0.05 * main_probability
            for prob in probability.values()
        )
        if equality == main_probability:
            uniformity = "рівна"
        else:
            uniformity = "нерівна"
        entropy = -sum(p * math.log2(p) for p in probability.values())
        if sequence_size_alphabet > 1:
            source_excess = 1 - entropy / math.log2(sequence_size_alphabet)
        else:
            source_excess = 1
        return probability_string, main_probability, uniformity, entropy, source_excess

    text = open("results_sequence.txt", "w")
    # 1 Задание
    N_sequence = 100
    N1 = 4  # варіант
    arr1 = [1] * N1
    N0 = N_sequence - N1
    arr0 = [0] * N0
    results = []
    original_sequence_1 = arr1 + arr0
    random.shuffle(original_sequence_1)
    original_sequence_1 = "".join(map(str, original_sequence_1))
    text.write(f"Варіант: 10\n")
    text.write(f"Завдання 1\n")
    text.write("Послідовність: " + str(original_sequence_1) + "\n")
    original_sequence_size = len(original_sequence_1)
    text.write("Розмір послідовності: " + str(original_sequence_size) + " byte" + "\n")
    unique_chars = set(original_sequence_1)
    sequence_size_alphabet = len(unique_chars)
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")
    probability_str, mean_probability, uniformity, entropy, source_excess = parameters(
        original_sequence_1
    )
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_excess, 2), uniformity]
    )
    text = open("results_sequence.txt", "a")
    text.write(f"Ймовірність появи символів: {probability_str}\n")
    text.write(f"Середнє арифметичне ймовірності: {round(mean_probability, 3)}\n")
    text.write(f"Ймовірність розподілу символів: {uniformity}\n")
    text.write(f"Ентропія: {round(mean_probability, 3)}\n")
    text.write(f"Надмірність джерела: {round(mean_probability, 3)}\n")
    text.write("\n")

    # 2 Задание
    N2 = 6
    list2 = ["Б", "а", "т", "р", "а", "к"]
    N0_2 = N_sequence - N2
    list0_2 = [0] * N0_2
    original_sequence_2 = list2 + list0_2
    original_sequence_2 = "".join(map(str, original_sequence_2))
    text.write(f"Завдання 2\n")
    text.write("Послідовність: " + str(original_sequence_2) + "\n")
    text.write(
        "Розмір послідовності: " + str(len(original_sequence_1)) + " byte" + "\n"
    )
    unique_chars_2 = set(original_sequence_2)
    sequence_size_alphabet = len(unique_chars)
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")

    probability_str, mean_probability, uniformity, entropy, source_excess = parameters(
        original_sequence_2
    )
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_excess, 2), uniformity]
    )

    text = open("results_sequence.txt", "a")
    text.write(f"Ймовірність появи символів: {probability_str}\n")
    text.write(f"Середнє арифметичне ймовірності: {round(mean_probability, 3)}\n")
    text.write(f"Ймовірність розподілу символів: {uniformity}\n")
    text.write(f"Ентропія: {round(mean_probability, 3)}\n")
    text.write(f"Надмірність джерела: {round(mean_probability, 3)}\n")
    text.write("\n")

    # 3 задание
    original_sequence_3 = list(original_sequence_2)
    random.shuffle(original_sequence_3)
    original_sequence_3 = "".join(map(str, original_sequence_3))
    text.write(f"Завдання 3\n")
    text.write("Послідовність: " + str(original_sequence_3) + "\n")
    text.write(
        "Розмір послідовності: " + str(len(original_sequence_3)) + " byte" + "\n"
    )
    unique_chars = set(original_sequence_3)
    sequence_size_alphabet = len(unique_chars)
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")

    probability_str, mean_probability, uniformity, entropy, source_excess = parameters(
        original_sequence_3
    )
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_excess, 2), uniformity]
    )

    text = open("results_sequence.txt", "a")
    text.write(f"Ймовірність появи символів: {probability_str}\n")
    text.write(f"Середнє арифметичне ймовірності: {round(mean_probability, 3)}\n")
    text.write(f"Ймовірність розподілу символів: {uniformity}\n")
    text.write(f"Ентропія: {round(mean_probability, 3)}\n")
    text.write(f"Надмірність джерела: {round(mean_probability, 3)}\n")
    text.write("\n")

    # 4 задание
    aray = []
    letters = ["Б", "а", "т", "р", "а", "к", "5", "1", "9", "c", "т"]
    n_letters = len(letters)
    n_repeats = N_sequence / n_letters
    remainder = N_sequence * (N_sequence % n_letters)
    aray += letters * int(n_repeats)
    aray += letters[:remainder]
    original_sequence_4 = "".join(map(str, aray))
    text.write(f"Завдання 4\n")
    text.write("Послідовність: " + original_sequence_4 + "\n")
    text.write(
        "Розмір послідовності: " + str(len(original_sequence_4)) + " byte" + "\n"
    )
    unique_chars = set(original_sequence_4)
    sequence_size_alphabet = len(unique_chars)
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")

    probability_str, mean_probability, uniformity, entropy, source_excess = parameters(
        original_sequence_4
    )
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_excess, 2), uniformity]
    )

    text = open("results_sequence.txt", "a")
    text.write(f"Ймовірність появи символів: {probability_str}\n")
    text.write(f"Середнє арифметичне ймовірності: {round(mean_probability, 3)}\n")
    text.write(f"Ймовірність розподілу символів: {uniformity}\n")
    text.write(f"Ентропія: {round(mean_probability, 3)}\n")
    text.write(f"Надмірність джерела: {round(mean_probability, 3)}\n")
    text.write("\n")

    # 5 задание
    alphabet = ["б", "а", "5", "1", "9"]
    Pi = 0.2
    length = Pi * N_sequence
    original_sequence_5 = alphabet * int(length)
    random.shuffle(original_sequence_5)
    original_sequence_5 = "".join(map(str, original_sequence_5))
    text.write(f"Завдання 5\n")
    text.write("Послідовність: " + str(original_sequence_5) + "\n")
    text.write(
        "Розмір послідовності: " + str(len(original_sequence_5)) + " byte" + "\n"
    )
    unique_chars = set(original_sequence_5)
    sequence_size_alphabet = len(unique_chars)
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")

    counts = collections.Counter(original_sequence_5)
    probability = {symbol: count / N_sequence for symbol, count in counts.items()}
    probability_str = ", ".join(
        [f"{symbol}={prob:.4f}" for symbol, prob in probability.items()]
    )
    mean_probability = sum(probability.values()) / len(probability)
    equal = all(
        abs(prob - mean_probability) < 0.05 * mean_probability
        for prob in probability.values()
    )
    if equal:
        uniformity = "рівна"
    else:
        uniformity = "нерівна"
    entropy = -sum(p * math.log2(p) for p in probability.values())
    if sequence_size_alphabet > 1:
        source_excess = 1 - entropy / math.log2(sequence_size_alphabet)
    else:
        source_excess = 1
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_excess, 2), uniformity]
    )

    text = open("results_sequence.txt", "a")
    text.write(f"Ймовірність появи символів: {probability_str}\n")
    text.write(f"Середнє арифметичне ймовірності: {round(mean_probability, 3)}\n")
    text.write(f"Ймовірність розподілу символів: {uniformity}\n")
    text.write(f"Ентропія: {round(mean_probability, 3)}\n")
    text.write(f"Надмірність джерела: {round(mean_probability, 3)}\n")
    text.write("\n")

    # 6 задание
    list_letters = ["б", "а"]
    list_digits = ["5", "1", "9"]
    P_letters = 0.7
    P_digits = 0.3
    n_letters6 = int(P_letters * N_sequence) / len(list_letters)
    n_digits6 = int(P_digits * N_sequence) / len(list_digits)
    list_l = list_letters * int(n_letters6)
    list_d = list_digits * int(n_digits6)
    original_sequence_6 = list_l + list_d
    random.shuffle(original_sequence_6)
    original_sequence_6 = "".join(map(str, original_sequence_6))
    text.write(f"Завдання 6\n")
    text.write("Послідовність: " + str(original_sequence_6) + "\n")
    text.write(
        "Розмір послідовності: " + str(len(original_sequence_6)) + " byte" + "\n"
    )
    unique_chars = set(original_sequence_6)
    sequence_size_alphabet = len(unique_chars)
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")

    probability_str, mean_probability, uniformity, entropy, source_excess = parameters(
        original_sequence_6
    )
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_excess, 2), uniformity]
    )

    text = open("results_sequence.txt", "a")
    text.write(f"Ймовірність появи символів: {probability_str}\n")
    text.write(f"Середнє арифметичне ймовірності: {round(mean_probability, 3)}\n")
    text.write(f"Ймовірність розподілу символів: {uniformity}\n")
    text.write(f"Ентропія: {round(mean_probability, 3)}\n")
    text.write(f"Надмірність джерела: {round(mean_probability, 3)}\n")
    text.write("\n")

    # 7 задание
    elements = string.ascii_lowercase + string.digits
    original_sequence_7 = [random.choice(elements) for _ in range(N_sequence)]
    original_sequence_7 = "".join(map(str, original_sequence_7))
    text.write("Послідовність: " + str(original_sequence_7) + "\n")
    text.write(
        "Розмір послідовності: " + str(len(original_sequence_7)) + " byte" + "\n"
    )
    unique_chars = set(original_sequence_7)
    sequence_size_alphabet = len(unique_chars)
    text.write(f"Завдання 7\n")
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")

    probability_str, mean_probability, uniformity, entropy, source_excess = parameters(
        original_sequence_7
    )
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_excess, 2), uniformity]
    )

    text = open("results_sequence.txt", "a")
    text.write(f"Ймовірність появи символів: {probability_str}\n")
    text.write(f"Середнє арифметичне ймовірності: {round(mean_probability, 3)}\n")
    text.write(f"Ймовірність розподілу символів: {uniformity}\n")
    text.write(f"Ентропія: {round(mean_probability, 3)}\n")
    text.write(f"Надмірність джерела: {round(mean_probability, 3)}\n")
    text.write("\n")

    # 8 Задание
    original_sequence_8 = ["1"] * N_sequence
    original_sequence_8 = "".join(map(str, original_sequence_8))
    text.write(f"Завдання 8\n")
    text.write("Послідовність: " + str(original_sequence_8) + "\n")
    text.write(
        "Розмір послідовності: " + str(len(original_sequence_8)) + " byte" + "\n"
    )
    unique_chars = set(original_sequence_8)
    sequence_size_alphabet = len(unique_chars)
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")

    probability_str, mean_probability, uniformity, entropy, source_excess = parameters(
        original_sequence_8
    )
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_excess, 2), uniformity]
    )

    text = open("results_sequence.txt", "a")
    text.write(f"Ймовірність появи символів: {probability_str}\n")
    text.write(f"Середнє арифметичне ймовірності: {round(mean_probability, 3)}\n")
    text.write(f"Ймовірність розподілу символів: {uniformity}\n")
    text.write(f"Ентропія: {round(mean_probability, 3)}\n")
    text.write(f"Надмірність джерела: {round(mean_probability, 3)}\n")
    text.write("\n")

    text.close()

    seq = open("sequence.txt", "w")
    original_sequences = [
        original_sequence_1,
        original_sequence_2,
        original_sequence_3,
        original_sequence_4,
        original_sequence_5,
        original_sequence_6,
        original_sequence_7,
        original_sequence_8,
    ]
    seq.write(str(original_sequences))
    seq.close()

    text.close()

    fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))
    headers = ["Розмір алфавіту", "Ентропія", "Надмірність", "Ймовірність"]
    row = [
        "Послідовність 1",
        "Послідовність 2",
        "Послідовність 3",
        "Послідовність 4",
        "Послідовність 5",
        "Послідовність 6",
        "Послідовність 7",
        "Послідовність 8",
    ]
    ax.axis("off")
    table = ax.table(
        cellText=results,
        colLabels=headers,
        rowLabels=row,
        loc="center",
        cellLoc="center",
    )
    table.set_fontsize(14)
    table.scale(0.8, 2)
    fig.savefig("Таблиця" + ".png")


lossless_comp()
