import os

import cv2
import numpy as np

image_dir = r"D:\Dataset\MLFW\MLFW_112\aligned_maskParsing"
mask_dir = r"D:\Dataset\MLFW\MLFW_112\aligned_maskMask"


pair_dir = r"D:\Dataset\MLFW\MLFW_112/pairs.txt"
pair_list = [line.split() for line in open(pair_dir).readlines()]

# print(pair_list)

save_dir = r"D:\Dataset\MLFW\MLFW_112/aligned_pair_mask"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i, (input1_name, input2_name, check) in enumerate(pair_list):
    # input 1 read
    input1 = cv2.imread(os.path.join(image_dir, input1_name))
    input1_mask = cv2.imread(os.path.join(mask_dir, input1_name))

    # input 2 read
    input2 = cv2.imread(os.path.join(image_dir, input2_name))
    input2_mask = cv2.imread(os.path.join(mask_dir, input2_name))

    # 마스크 유무 판단, 임계값 1000
    input1_after = input1.copy()
    input2_after = input2.copy()
    if len(np.where(input1_mask == 0)) < 1000:
        # print(len(input1_mask[np.where(input1_mask==0)]))
        input1_after[np.where(input2_mask <= 10)] = 0
    if len(np.where(input2_mask == 0)) < 1000:
        # print(len(input2_mask[np.where(input2_mask == 0)]))
        input2_after[np.where(input1_mask <= 10)] = 0

    if int(len(np.where(input1_mask == 0)) < 1000) or int(len(np.where(input2_mask == 0)) < 1000):
        # print(i+1)
        # cv2.imshow("input1", input1)
        # cv2.imshow("input2", input2)
        # cv2.imshow("input1_after", input1_after)
        # cv2.imshow("input2_after", input2_after)
        # if cv2.waitKey(1000) & 0xFF == ord('q'):
        #     break

        cv2.imwrite(os.path.join(save_dir, input1_name), input1_after)
        cv2.imwrite(os.path.join(save_dir, input2_name), input2_after)
