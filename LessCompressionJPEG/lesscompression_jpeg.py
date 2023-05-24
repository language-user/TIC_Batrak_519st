import os
from LessCompressionJPEG.huffman import HuffmanTree
import math
import numpy as np
from scipy import fftpack
from PIL import Image


class JPEGFileReader:
    def __init__(self, filepath):
        self.file = open(filepath, 'r')

    def read_int(self, size):
        if size == 0:
            return 0
        bin_num = self._read_str(size)
        if bin_num[0] == '1':
            return self._int2(bin_num)
        else:
            return self._int2(self._binstr_flip(bin_num)) * -1

    def read_dc_table(self):
        table = dict()
        table_size = self._read_uint(16)
        for _ in range(table_size):
            category = self._read_uint(4)
            code_length = self._read_uint(4)
            code = self._read_str(code_length)
            table[code] = category
        return table

    def read_ac_table(self):
        table = dict()
        table_size = self._read_uint(16)
        for _ in range(table_size):
            run_length = self._read_uint(4)
            size = self._read_uint(4)
            code_length = self._read_uint(8)
            code = self._read_str(code_length)
            table[code] = (run_length, size)
        return table

    def read_blocks_count(self):
        return self._read_uint(32)

    def read_huffman_code(self, table):
        prefix = ''
        while prefix not in table:
            prefix += self._read_char()
        return table[prefix]

    def _read_uint(self, size):
        if size <= 0:
            raise ValueError("Size must be greater than 0")
        return self._int2(self._read_str(size))

    def _read_str(self, length):
        return self.file.read(length)

    def _read_char(self):
        return self._read_str(1)

    def _int2(self, bin_num):
        return int(bin_num, 2)


import os
import numpy as np
from PIL import Image


def encode(output_f: str, input_f: str, table_num: int):
	input_file = f"bmp4/{input_f}bmp"
	output_file = f"{output_f}.asf"
	image = Image.open(input_file)
	ycbcr = image.convert('YCbCr')
	npmat = np.array(ycbcr, dtype=np.uint8)
	rows, cols = npmat.shape[0], npmat.shape[1]

	if rows % 8 == cols % 8 == 0:
		blocks_count = rows // 8 * cols // 8
	else:
		raise ValueError("Ширина і висота зображення мають бути кратними 8")

	dc = np.empty((blocks_count, 3), dtype=np.int32)
	ac = np.empty((blocks_count, 63, 3), dtype=np.int32)

	for i in range(0, rows, 8):
		for j in range(0, cols, 8):
			try:
				block_index += 1
			except NameError:
				block_index = 0

			for k in range(3):
				block = npmat[i:i + 8, j:j + 8, k] - 128
				dct_matrix = dct_2d(block)
				quant_matrix = quantize(dct_matrix, 'lum' if k == 0 else 'chrom', table_num)
				zigzag = block_to_zigzag(quant_matrix)
				dc[block_index, k] = zigzag[0]
				ac[block_index, :, k] = zigzag[1:]

	H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
	H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
	H_AC_Y = HuffmanTree(flatten(run_length_encode(ac[i, :, 0])[0] for i in range(blocks_count)))
	H_AC_C = HuffmanTree(flatten(run_length_encode(ac[i, :, j])[0] for i in range(blocks_count) for j in [1, 2]))

	tables = {
		'dc_y': H_DC_Y.value_to_bitstring_table(),
		'ac_y': H_AC_Y.value_to_bitstring_table(),
		'dc_c': H_DC_C.value_to_bitstring_table(),
		'ac_c': H_AC_C.value_to_bitstring_table()
	}

	size_vyhsdnogo = os.path.getsize(input_file)

	with open("results_jpeg.txt", "a", encoding="utf8") as file:
		print(f'Таблиця квантування - {table_num}, зображення - {input_f}', file=file)
		print(f'Розмір вихідного файла: {size_vyhsdnogo} байт', file=file)

	write_to_file(f"Result/{output_file}", dc, ac, blocks_count, tables)

	return size_vyhsdnogo


def dequantize(block, component, table_num):
	q = load_quantization_table(component, table_num)
	return block * q


def idct_2d(image):
	return fftpack.idct(fftpack.idct(image.T, norm='ortho').T, norm='ortho')


def zigzag_to_block(zigzag):
	rows = cols = int(math.sqrt(len(zigzag)))

	if rows * cols != len(zigzag):
		raise ValueError("Довжина зіг-зага повинна бути ідеальним квадратом")

	block = np.empty((rows, cols), np.int32)

	for i, point in enumerate(zigzag_points(rows, cols)):
		block[point] = zigzag[i]

	return block


def read_image_file(filepath):
	file_reader = JPEGFileReader(filepath)
	huffman_tables = dict()

	for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
		if 'dc' in table_name:
			huffman_tables[table_name] = file_reader.read_dc_table()
		else:
			huffman_tables[table_name] = file_reader.read_ac_table()

	blocks_count = file_reader.read_blocks_count()
	dc = np.empty((blocks_count, 3), dtype=np.int32)
	ac = np.empty((blocks_count, 63, 3), dtype=np.int32)

	for block_index in range(blocks_count):
		for component in range(3):
			dc_table = huffman_tables['dc_y'] if component == 0 else huffman_tables['dc_c']
			ac_table = huffman_tables['ac_y'] if component == 0 else huffman_tables['ac_c']

			category = file_reader.read_huffman_code(dc_table)
			dc[block_index, component] = file_reader.read_int(category)

			cells_count = 0
			while cells_count < 63:
				run_length, size = file_reader.read_huffman_code(ac_table)

				if (run_length, size) == (0, 0):
					while cells_count < 63:
						ac[block_index, cells_count, component] = 0
						cells_count += 1
				else:
					for _ in range(run_length):
						ac[block_index, cells_count, component] = 0
						cells_count += 1

					if size == 0:
						ac[block_index, cells_count, component] = 0
					else:
						value = file_reader.read_int(size)
						ac[block_index, cells_count, component] = value

					cells_count += 1

	return dc, ac, huffman_tables, blocks_count


def binstr_flip(binstr):
    if not set(binstr).issubset('01'):
        raise ValueError("binstr should contain only '0' and '1'")
    return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))


def int_to_binstr(n):
    if n == 0:
        return ''
    binstr = bin(abs(n))[2:]
    return binstr if n > 0 else binstr_flip(binstr)


def uint_to_binstr(number, size):
    return bin(number)[2:][-size:].zfill(size)



def write_to_file(filepath, dc, ac, blocks_count, tables):
    f = open(filepath, 'w')
    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
        f.write(uint_to_binstr(len(tables[table_name]), 16))
        for key, value in tables[table_name].items():
            if table_name in {'dc_y', 'dc_c'}:
                f.write(uint_to_binstr(key, 4))
                f.write(uint_to_binstr(len(value), 4))
                f.write(value)
            else:
                f.write(uint_to_binstr(key[0], 4))
                f.write(uint_to_binstr(key[1], 4))
                f.write(uint_to_binstr(len(value), 8))
                f.write(value)

    f.write(uint_to_binstr(blocks_count, 32))
    for b in range(blocks_count):
        for c in range(3):
            category = bits_required(dc[b, c])
            symbols, values = run_length_encode(ac[b, :, c])
            dc_table = tables['dc_y'] if c == 0 else tables['dc_c']
            ac_table = tables['ac_y'] if c == 0 else tables['ac_c']
            f.write(dc_table[category])
            f.write(int_to_binstr(dc[b, c]))
            for i in range(len(symbols)):
                f.write(ac_table[tuple(symbols[i])])
                f.write(values[i])
    f.close()



def bits_required(n):
    n, result = abs(n), 0
    while n > 0:
        n >>= 1
        result += 1
    return result


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def run_length_encode(arr):
    last_nonzero, run_length = -1, 0
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i
    symbols, values = [], []
    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            values.append(int_to_binstr(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_required(elem)
            symbols.append((run_length, size))
            values.append(int_to_binstr(elem))
            run_length = 0
    return symbols, values



def dct_2d(image):
    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')


def load_quantization_table(component, table_num: int):
    if component == 'lum':
        if table_num == 1:
            q = np.array([
                [2, 2, 2, 2, 3, 4, 5, 6],
                [2, 2, 2, 2, 3, 4, 5, 6],
                [2, 2, 2, 2, 4, 5, 7, 9],
                [2, 2, 2, 4, 5, 7, 9, 12],
                [3, 3, 4, 5, 8, 10, 12, 12],
                [4, 4, 5, 7, 10, 12, 12, 12],
                [5, 5, 7, 9, 12, 12, 12, 12],
                [6, 6, 9, 12, 12, 12, 12, 12]])
        elif table_num == 2:
            q = np.array([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 48, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])
    elif component == 'chrom':
        if table_num == 1:
            q = np.array([
                [3, 3, 5, 9, 13, 15, 15, 15],
                [3, 4, 6, 11, 14, 12, 12, 12],
                [5, 6, 9, 14, 12, 12, 12, 12],
                [9, 11, 14, 12, 12, 12, 12, 12],
                [13, 14, 12, 12, 12, 12, 12, 12],
                [15, 12, 12, 12, 12, 12, 12, 12],
                [15, 12, 12, 12, 12, 12, 12, 12],
                [15, 12, 12, 12, 12, 12, 12, 12]])
        elif table_num == 2:
            q = np.array([
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]])

    else:
        raise ValueError(f"Component must be 'lum' or 'chrom', but found {component}")

    return q



def quantize(block, component, table_num):
    return (block / load_quantization_table(component, table_num)).round().astype(np.int32)


def block_to_zigzag(block):
    return np.array([block[point] for point in zigzag_points(*block.shape)])


def zigzag_points(rows, cols):
    UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)

    def move(direction, point):
        return {
            UP: lambda p: (p[0] - 1, p[1]),
            DOWN: lambda p: (p[0] + 1, p[1]),
            LEFT: lambda p: (p[0], p[1] - 1),
            RIGHT: lambda p: (p[0], p[1] + 1),
            UP_RIGHT: lambda p: move(UP, move(RIGHT, p)),
            DOWN_LEFT: lambda p: move(DOWN, move(LEFT, p))
        }[direction](point)

    def in_bounds(point):
        return 0 <= point[0] < rows and 0 <= point[1] < cols

    point, move_up = (0, 0), True
    for i in range(rows * cols):
        yield point
        if move_up:
            if in_bounds(move(UP_RIGHT, point)):
                point = move(UP_RIGHT, point)
            else:
                move_up = False
                if in_bounds(move(RIGHT, point)):
                    point = move(RIGHT, point)
                else:
                    point = move(DOWN, point)
        else:
            if in_bounds(move(DOWN_LEFT, point)):
                point = move(DOWN_LEFT, point)
            else:
                move_up = True
                if in_bounds(move(DOWN, point)):
                    point = move(DOWN, point)
                else:
                    point = move(RIGHT, point)



def decoder(output_filename: str, input_filename: str, table_num: int, target_size: int):
    input_file = f"Result/{input_filename}.asf"
    output_file = f"Result/{output_filename}.jpeg"
    dc, ac, tables, blocks_count = read_image_file(input_file)
    block_side = 8
    image_side = int(math.sqrt(blocks_count)) * block_side
    blocks_per_line = image_side // block_side
    npmat = np.empty((image_side, image_side, 3), dtype=np.uint8)
    for block_index in range(blocks_count):
        i = block_index // blocks_per_line * block_side
        j = block_index % blocks_per_line * block_side
        for channel in range(3):
            zigzag = [dc[block_index, channel]] + list(ac[block_index, :, channel])
            quant_matrix = zigzag_to_block(zigzag)
            dct_matrix = dequantize(quant_matrix, 'lum' if channel == 0 else 'chrom', table_num)
            block = idct_2d(dct_matrix)
            npmat[i:i + 8, j:j + 8, channel] = block + 128
    image = Image.fromarray(npmat, 'YCbCr')
    image = image.convert('RGB')
    image.save(output_file)
    size_jpeg = os.path.getsize(output_file)
    width, height = image.size
    compression_ratio = target_size / size_jpeg
    with open("results_jpeg.txt", "a", encoding="utf8") as file:
        print(f'Розмір файла JPEG: {size_jpeg} байт', file=file)
        print(f'Розмір зображення JPEG: {width}x{height}', file=file)
        print(f'Коефіцієнт стиснення = {compression_ratio}\n', file=file)


def main():
    files = os.listdir("bmp4")
    file_names = [el[:4] for el in files]
    for i in range(1, 3):
        for file_name in file_names:
            size = encode(f"{file_name}_{i}", file_name, i)
            decoder(f"{file_name}_{i}", f"{file_name}_{i}", i, size)


if __name__ == "__main__":
    main()
