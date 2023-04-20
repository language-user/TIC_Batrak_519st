import ast
import collections
import math
from matplotlib import pyplot as plt

open("results_rle_lzw.txt", "w")
list_of_results = []
N_sequence = 100


def encode_rle(sequence):

    result = []
    count = 1


    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            count += 1
        else:
            result.append((sequence[i-1], count))
            count = 1
    result.append((sequence[-1], count))


    output = []
    for item in result:
        output.append(f"{item[1]}{item[0]}")

    return "".join(output), result



def encode_lzw(sequence):
    d = {}
    for i in range(65536):
        d[chr(i)] = i
    c = ""
    r = []
    s = 0
    for char in sequence:
        n = c + char
        if n in d:
            c = n
        else:
            r.append(d[c])
            d[n] = len(d)
            b = 16 if d[c] < 65536 else math.ceil(math.log2(len(d)))
            c = char
            with open("results_rle_lzw.txt", "a") as f:
                f.write(f"Code: {d[c]}, Element: {c}, Bits: {b}\n")
                f.close()
            s = s + b
    l = 16 if d[c] < 65536 else math.ceil(math.log2(len(d)))
    s = s + l
    with open("results_rle_lzw.txt", "a") as f:
        f.write(f"Code: {d[c]}, Element: {c}, Bits: {l}\n")
    r.append(d[c])
    return r, s

def decode_rle(sequence):
    r = []
    for item in sequence:
        r.append(item[0] * item[1])
    return "".join(r)



def decode_lzw(sequence):

    d = {i: chr(i) for i in range(65536)}
    result = ""
    prev = None
    curr = ""


    for code in sequence:
        if code in d:
            curr = d[code]
        else:
            curr = prev + prev[0]
        result += curr


        if prev is not None:
            d[len(d)] = prev + curr[0]
        prev = curr

    return result


def main():
    with open("sequence.txt", "r") as file:
        first_sequence = ast.literal_eval(file.read())
        first_sequence = [i.strip("[]',") for i in first_sequence]
        file.close()

    for sequence in first_sequence:
        counter = collections.Counter(sequence)
        probability = {symbol: count / N_sequence for symbol, count in counter.items()}
        entropy = -sum(p * math.log2(p) for p in probability.values())

        file = open("results_rle_lzw.txt", "a")
        file.write('Оригінальна послідовність: {0}\n'.format(str(sequence)))
        file.write('Розмір оригінальної послідовності: {0} bits\n'.format(str(len(sequence) * 16)))
        file.write('Ентропія: {0}\n'.format(str(entropy)))
        file.write('\n')
        file.close()

        encoded_sequence, encoded = encode_rle(sequence)
        decoded_sequence = decode_rle(encoded)
        total_rle = len(encoded_sequence) * 16
        compression_ratio_RLE = round((len(sequence) / len(encoded_sequence)), 2)

        if compression_ratio_RLE < 1:
            compression_ratio_RLE = '-'
        else:
            compression_ratio_RLE = compression_ratio_RLE
        file = open("results_rle_lzw.txt", "a")
        file.write(
            '_________________________________________Кодування_RLE_______________________________________' + '\n')
        file.write('Закодована RLE послідовність: ' + str(encoded_sequence) + '\n')
        file.write('Розмір закодованої RLE послідовності: ' + str(total_rle) + ' bits' + '\n')
        file.write('Коефіцієнт стиснення RLE: ' + str(compression_ratio_RLE) + '\n')
        file.close()

        file = open("results_rle_lzw.txt", "a")
        file.write('Декодована RLE послідовність: ' + str(decoded_sequence) + '\n')
        file.write('Розмір декодованої RLE послідовності: ' + str(len(decoded_sequence) * 16) + ' bits' + '\n')
        file.close()

        with open("results_rle_lzw.txt", "a") as file:
            file.write(
                '_________________________________________Кодування_LZW_________________________________________' + '\n')
            file.write(
                '_________________________________________Поетапне кодування_________________________________________' + '\n')

        encoded_result, size = encode_lzw(sequence)
        with open("results_rle_lzw.txt", "a") as file:
            file.write('________________________________________________________________________________' + '\n')
            file.write(f"Закодована LZW послідовність:{''.join(map(str, encoded_result))} \n")
            file.write(f"Розмір закодованої LZW послідовності: {size} bits \n")
            compression_ratio_lzw = round((len(sequence) * 16 / size), 2)

            if compression_ratio_lzw < 1:
                compression_ratio_lzw = '-'
            else:
                compression_ratio_lzw = compression_ratio_lzw

            file.write(f"Коефіціент стиснення LZW: {compression_ratio_lzw} \n")
            file.close()


        decoded_result_lzw = decode_lzw(encoded_result)
        with open("results_rle_lzw.txt", "a") as file:
            file.write(f"Декодована LZW послідовність:{''.join(map(str, decoded_result_lzw))} \n")
            file.write(f"Розмір декодованої LZW послідовності: {len(decoded_result_lzw) * 16} bits \n ")
            file.write('\n' + '\n' + '\n' + '\n' + '\n')
            file.close()

        list_of_results.append([round(entropy, 2), compression_ratio_RLE, compression_ratio_lzw])

    fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))
    headers = ['Ентропія', 'КС RLE', 'КС LZW']
    row = ['Послідовність 1', 'Послідовність 2', 'Послідовність 3', 'Послідовність 4', 'Послідовність 5',
           'Послідовність 6', 'Послідовність 7', 'Послідовність 8']
    ax.axis('off')
    table = ax.table(cellText=list_of_results, colLabels=headers, rowLabels=row,
                     loc='center', cellLoc='center')

    table.set_fontsize(14)
    table.scale(0.8, 2)
    ax.text(0.5, 0.95, 'Результати стиснення методами RLE та LZW', transform=ax.transAxes, ha='center', va='top', fontsize=14)

    fig.savefig('Результати стиснення методами RLE та LZW' + '.jpg', dpi=600)

if "__main__" == __name__:
    main()





