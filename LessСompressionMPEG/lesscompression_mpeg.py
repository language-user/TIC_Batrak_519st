import math
import os
import random

import numpy as np
from matplotlib import pyplot as plt


import cv2

def get_frames(filename, first_frame, second_frame):
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame - 1)
    success1, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, second_frame - 1)
    success2, frame2 = cap.read()
    cap.release()
    return frame1, frame2

frame_number = 10
video_filename = "sample.avi"
first_frame, second_frame = get_frames(video_filename, frame_number, frame_number + 1)
print("First Frame Shape:", first_frame.shape)
print("Second Frame Shape:", second_frame.shape)



import math

def get_bits_per_pixel(image):
    height, width = image.shape
    image_list = image.tolist()
    total_bits = 0
    for row in image_list:
        for pixel in row:
            total_bits += math.log2(abs(pixel) + 1)
    bits_per_pixel = total_bits / (height * width)
    return bits_per_pixel

image_data = [[100, 200, 150], [50, 75, 120], [80, 160, 240]]
image = np.array(image_data)
bpp = get_bits_per_pixel(image)
print("Bits per Pixel:", bpp)



def getReconstructTarget(residual, predicted):
    return np.add(residual, predicted)


def getResidual(target, predicted):
    return np.subtract(target, predicted)


def block_search_body(input_anchor, input_target, block_size, search_area=7):
    height, width = input_anchor.shape
    segments_height, segments_width = segmentImage(input_anchor, block_size)
    predicted = np.ones((height, width)) * 255
    block_count = 0
    for y in range(0, int(segments_height * block_size), block_size):
        for x in range(0, int(segments_width * block_size), block_size):
            block_count += 1
            target_block = input_target[y:y + block_size, x:x + block_size]
            anchor_search_area = getAnchorSearchArea(x, y, input_anchor, block_size, search_area)
            anchor_block = getBestMatch(target_block, anchor_search_area, block_size)
            predicted[y:y + block_size, x:x + block_size] = anchor_block
    assert block_count == int(segments_height * segments_width)
    return predicted



def segmentImage(anchor, blockSize=16):
    h, w = anchor.shape
    hSegments = int(h/blockSize)
    wSegments = int(w/blockSize)
    return hSegments, wSegments


def getAnchorSearchArea(x, y, anchor, blockSize, searchArea):
    h, w = anchor.shape
    cx, cy = getCenter(x, y, blockSize)
    sx = max(0, cx-int(blockSize/2)-searchArea)
    sy = max(0, cy-int(blockSize/2)-searchArea)
    anchorSearch = anchor[sy:min(sy+searchArea*2+blockSize, h), sx:min(sx+searchArea*2+blockSize, w)]
    return anchorSearch


def getCenter(x, y, blockSize):
    return int(x + blockSize/2), int(y + blockSize/2)


def getBestMatch(tBlock, aSearch, blockSize):
    step = 4
    ah, aw = aSearch.shape
    acy, acx = int(ah/2), int(aw/2)
    minMAD = float("+inf")
    minP = None
    while step >= 1:
        p1 = (acx, acy)
        p2 = (acx+step, acy)
        p3 = (acx, acy+step)
        p4 = (acx+step, acy+step)
        p5 = (acx-step, acy)
        p6 = (acx, acy-step)
        p7 = (acx-step, acy-step)
        p8 = (acx+step, acy-step)
        p9 = (acx-step, acy+step)
        pointList = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
        for p in range(len(pointList)):
            aBlock = getBlockZone(pointList[p], aSearch, tBlock, blockSize)
            MAD = getMAD(tBlock, aBlock)
            if MAD < minMAD:
                minMAD = MAD
                minP = pointList[p]
        step = int(step/2)
    px, py = minP
    px, py = px - int(blockSize / 2), py - int(blockSize / 2)
    px, py = max(0, px), max(0, py)
    matchBlock = aSearch[py:py + blockSize, px:px + blockSize]
    return matchBlock


def getBlockZone(p, aSearch, tBlock, blockSize):
    px, py = p
    px, py = px-int(blockSize/2), py-int(blockSize/2)
    px, py = max(0,px), max(0, py)
    aBlock = aSearch[py:py+blockSize, px:px+blockSize]
    try:
        assert aBlock.shape == tBlock.shape
    except Exception as e:
        print(e)
    return aBlock


def getMAD(tBlock, aBlock):
    return np.sum(np.abs(np.subtract(tBlock, aBlock))) / (tBlock.shape[0] * tBlock.shape[1])


def main(input_anchor_frame, input_target_frame, block_size, save_output=True):
    bits_anchor = []
    bits_diff = []
    bits_predicted = []
    height, width, channels = input_anchor_frame.shape
    print(height, width, channels)
    diff_frame_rgb = np.zeros((height, width, channels))
    predicted_frame_rgb = np.zeros((height, width, channels))
    residual_frame_rgb = np.zeros((height, width, channels))
    restore_frame_rgb = np.zeros((height, width, channels))
    for i in range(0, 3):
        anchor_frame_c = input_anchor_frame[:, :, i]
        target_frame_c = input_target_frame[:, :, i]
        diff_frame = cv2.absdiff(anchor_frame_c, target_frame_c)
        predicted_frame = getBlockZone(anchor_frame_c, target_frame_c, block_size)
        residual_frame = getResidual(target_frame_c, predicted_frame)
        reconstruct_target_frame = getReconstructTarget(residual_frame, predicted_frame)
        bits_anchor += [get_bits_per_pixel(anchor_frame_c)]
        bits_diff += [get_bits_per_pixel(diff_frame)]
        bits_predicted += [get_bits_per_pixel(residual_frame)]
        diff_frame_rgb[:, :, i] = diff_frame
        predicted_frame_rgb[:, :, i] = predicted_frame
        residual_frame_rgb[:, :, i] = residual_frame
        restore_frame_rgb[:, :, i] = reconstruct_target_frame
    output_directory = "Results"
    is_directory = os.path.isdir(output_directory)
    if not is_directory:
        os.mkdir(output_directory)
    if save_output:
        cv2.imwrite(f"{output_directory}/First frame.png", input_anchor_frame)
        cv2.imwrite(f"{output_directory}/Second frame.png", input_target_frame)
        cv2.imwrite(f"{output_directory}/Difference between frames.png", diff_frame_rgb)
        cv2.imwrite(f"{output_directory}/Prediction frame.png", predicted_frame_rgb)
        cv2.imwrite(f"{output_directory}/Residual frame.png", residual_frame_rgb)
        cv2.imwrite(f"{output_directory}/Restore frame.png", restore_frame_rgb)
        bar_width = 0.25
        fig = plt.subplots(figsize=(12, 8))
        p1 = [sum(bits_anchor), bits_anchor[0], bits_anchor[1], bits_anchor[2]]
        diff = [sum(bits_diff), bits_diff[0], bits_diff[1], bits_diff[2]]
        mpeg = [sum(bits_predicted), bits_predicted[0], bits_predicted[1], bits_predicted[2]]
        br1 = np.arange(len(p1))
        br2 = [x + bar_width for x in br1]
        br3 = [x + bar_width for x in br2]
        br4 = [x + bar_width for x in br3]
        plt.bar(br1, p1, color='r', width=bar_width, edgecolor='grey', label='Size for initial frame')
        plt.bar(br2, diff, color='g', width=bar_width, edgecolor='grey', label='Size for frame difference')
        plt.bar(br3, mpeg, color='b', width=bar_width, edgecolor='grey', label='Size for motion-compensated difference')
        plt.title(f'Compression Ratio = {round(sum(bits_anchor) / sum(bits_predicted), 2)}', fontweight='bold', fontsize=15)
        plt.ylabel('Bits per pixel', fontweight='bold', fontsize=15)
        plt.xticks([r + bar_width for r in range(len(p1))],
                   ['Bits/Pixel RGB', 'Bits/Pixel R', 'Bits/Pixel G', 'Bits/Pixel B'])
        plt.legend()
        plt.savefig(f'{output_directory}/Histogram of bits per pixel for different encodings.png', dpi=600)



if __name__ == "__main__":
    fr = random.randint(0, 3000)
    frame1, frame2 = get_frames('sample4.avi', fr, fr + 1)
    main(frame1, frame2, 32, saveOutput=True)