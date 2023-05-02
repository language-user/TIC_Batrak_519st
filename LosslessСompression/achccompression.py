import ast
import collections
import math
import matplotlib.pyplot as plt



filename = "results_AC_CH.txt"


def write_to_file(file_name, message):
    with open(file_name, "a") as file:
        file.write(message + '\n')
        file.close()



def main():
    write_to_file("results_AC_CH.txt", "")
    results = []
    with open("sequence.txt", "r") as file:
        original_sequences = ast.literal_eval(file.read())
        file.close()

    for sequence in original_sequences:
        sequence = sequence[:10]
        sequence_length = len(sequence)
        unique_chars = set(sequence)
        sequence_alphabet_size = len(unique_chars)
        counts = collections.Counter(sequence)
        probability = {symbol: count / sequence_length for symbol, count in counts.items()}
        entropy = -sum(p * math.log2(p) for p in probability.values())
        write_to_file('results_AC_CH.txt', f'Оригінальна послідовність: {sequence}')
        write_to_file('results_AC_CH.txt', f'Ентропія: {entropy}')
        write_to_file('results_AC_CH.txt', "")
        encoded_data_ac, encoded_sequence_ac = enc_ac(unique_chars, probability, sequence_alphabet_size, sequence)
        bps_ac = len(encoded_sequence_ac) / sequence_length
        decoded_sequence_ac = dec_ac(encoded_data_ac, sequence_length)
        encoded_data_hc, encoded_sequence_hc = enc_ch(unique_chars, probability, sequence)
        bps_hc = len(encoded_sequence_hc) / sequence_length
        decoded_sequence_hc = dec_ch(encoded_data_hc, sequence)
        write_to_file('results_AC_CH.txt', '_'*40 +'Арифметичне кодування' + '_'*40)
        write_to_file('results_AC_CH.txt', f'Дані закодованої АС послідовності: {encoded_data_ac}')
        write_to_file('results_AC_CH.txt', f'Закодована АС послідовність: {encoded_sequence_ac}')
        write_to_file('results_AC_CH.txt', f'Значення bps при кодуванні АС: {bps_ac}')
        write_to_file('results_AC_CH.txt', f'Декодована АС послідовність: {decoded_sequence_ac}')
        write_to_file('results_AC_CH.txt', "")
        write_to_file('results_AC_CH.txt', '_'*40 + 'Кодування Хаффмана' + '_'*40)
        write_to_file('results_AC_CH.txt', f'Дані закодованої HС послідовності: {encoded_data_hc}')
        write_to_file('results_AC_CH.txt', f'Закодована HС послідовність: {encoded_sequence_hc}')
        write_to_file('results_AC_CH.txt', f'Значення bps при кодуванні HС: {bps_hc}')
        write_to_file('results_AC_CH.txt', f'Декодована HС послідовність: {decoded_sequence_hc}')
        write_to_file('results_AC_CH.txt', '_'*80)
        write_to_file('results_AC_CH.txt', "")
        results.append(
            [round((entropy), 2),
             bps_ac,
             bps_hc])

    fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))
    headers = ['Ентропія', 'bps AC', 'bps CH']
    row = ['Послідовність 1', 'Послідовність 2', 'Послідовність 3', 'Послідовність 4', 'Послідовність 5','Послідовність 6', 'Послідовність 7', 'Послідовність 8']
    ax.axis('off')
    table = ax.table(cellText=results, colLabels=headers, rowLabels=row,loc='center', cellLoc='center')
    table.auto_set_font_size(True)
    table.set_fontsize(14)
    table.scale(0.6, 2.2)
    fig.savefig('Результати стиснення методами AC та CH' + '.jpg', dpi=300)



def fl_bins(point, size_cod):
    binary_cod = '{:0{}b}'.format(int(point * (2 ** size_cod)), size_cod)
    return binary_cod



def dec_ac(encoded_data_ac, sequence_length):
    point, alphabet_size, alphabet, probability = encoded_data_ac
    unity = [[alphabet[i], sum(probability[:i]), sum(probability[:i+1])] for i in range(alphabet_size)]
    decoded_sequence = ""
    for i in range(int(sequence_length)):
        for symbol, prob_low, prob_high in unity:
            if point > prob_low and point < prob_high:
                diff = prob_high - prob_low
                decoded_sequence += symbol
                for j in range(alphabet_size):
                    _, prob_l, prob_h = unity[j]
                    unity[j][1], unity[j][2] = prob_low, probability[j] * diff + prob_low
                    prob_low = unity[j][2]
                break
    return decoded_sequence



def enc_ch(uniq_chars, probabilitys, sequence):
    alphabet = list(uniq_chars)
    probability = [probabilitys[symbol] for symbol in alphabet]
    final = [[alphabet[i], probability[i]] for i in range(len(alphabet))]
    final.sort(key=lambda x: x[1])

    if 1 in probability and len(set(probability)) == 1:
        symbol_code = [[alphabet[i], "1" * i + "0"] for i in range(len(alphabet))]
        encode = "".join([symbol_code[alphabet.index(c)][1] for c in sequence])
    else:
        tree = []
        for _ in range(len(final) - 1):
            left = final.pop(0)
            right = final.pop(0)
            tot = left[1] + right[1]
            tree.append([left[0], right[0]])
            final.append([left[0] + right[0], tot])
            final.sort(key=lambda x: x[1])

        symbol_code = []
        tree.reverse()
        alphabet.sort()
        for i in range(len(alphabet)):
            code = ""
            for j in range(len(tree)):
                if alphabet[i] in tree[j][0]:
                    code += '0'
                    if alphabet[i] == tree[j][0]:
                        break
                else:
                    code += '1'
                    if alphabet[i] == tree[j][1]:
                        break
            symbol_code.append([alphabet[i], code])

        encode = "".join([symbol_code[i][1] for i in range(len(alphabet)) if symbol_code[i][0] == c][0] for c in sequence)

    return [encode, symbol_code], encode


def dec_ch(encoded_sequence, sequence):
    encode = list(encoded_sequence[0])
    symbol_code = encoded_sequence[1]
    count = 0
    flag = 0

    for i in range(len(encode)):
        for j in range(len(symbol_code)):
            if encode[i] == symbol_code[j][1]:
                sequence += str(symbol_code)
                flag = 1

        if flag == 1:
            flag = 0
        else:
            count += 1

            if count == len(encode):
                break
            else:
                encode.insert(i + 1, str(encode[i] + encode[i + 1]))
                encode.pop(i + 2)
    return sequence
def enc_ac(uniq_chars, probabilitys, alphabet_size, sequence):
    alphabet = list(uniq_chars)
    probability = [probabilitys[symbol] for symbol in alphabet]
    unity = []

    probability_dict = {alphabet[i]: probability[i] for i in range(alphabet_size)}

    probability_range = 0.0
    for i in range(alphabet_size):
        l = probability_range
        probability_range = probability_range + probability[i]
        u = probability_range
        unity.append([alphabet[i], l, u])
    for symbol in sequence[:-1]:
        for j in range(len(unity)):
            if symbol == unity[j][0]:
                probability_low = unity[j][1]
                probability_high = unity[j][2]
                diff = probability_high - probability_low
                for k in range(len(unity)):
                    unity[k][1] = probability_low
                    unity[k][2] = probability_dict[unity[k][0]] * diff + probability_low
                    probability_low = unity[k][2]
                break
    low = 0
    high = 0
    for i in range(len(unity)):
        if unity[i][0] == sequence[-1]:
            low = unity[i][1]
            high = unity[i][2]
    point = (low + high) / 2
    size_cod = math.ceil(math.log((1 / (high - low)), 2) + 1)
    bin_code = fl_bins(point, size_cod)
    return [point, alphabet_size, alphabet, probability], bin_code



main()