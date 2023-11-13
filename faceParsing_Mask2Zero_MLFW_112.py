from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import time

import argparse
import cv2
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from networks import dml_csr_danet_2
from visualization import visualization
from faceParsing_Onefile import _box2cs
from utils import transforms_forRealOcc

from skimage import transform as trans

torch.multiprocessing.set_start_method("spawn", force=True)

IGNORE_LABEL = 255
NUM_CLASSES = 20  # 20
SNAPSHOT_DIR = './snapshots/best_CAECAM_112_bn_for_test.pth' #
INPUT_SIZE = [112, 112]
DATA_DIR = r"D:\Dataset\MLFW\aligned"
data_list = os.listdir(DATA_DIR)

SAVE_DIR = r"D:\Dataset\MLFW\aligned_maskParsing"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DML_CSR Network")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=SNAPSHOT_DIR,
                        help="Where restore models parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="choose gpu numbers")
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument("--model_type", type=int, default=0,
                        help="choose models type")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    crop_size = INPUT_SIZE

    cudnn.benchmark = True
    cudnn.enabled = True

    model = dml_csr_danet_2.DML_CSR(args.num_classes, trained=False)

    restore_from = args.restore_from
    print(restore_from)
    state_dict = torch.load(restore_from, map_location='cuda:0')
    model.load_state_dict(state_dict)

    model.cuda()
    model.eval()


    for image_name in data_list:
        print(image_name)
        image = cv2.imread(os.path.join(DATA_DIR, image_name), cv2.IMREAD_COLOR)

        h, w, _ = image.shape
        center, s = _box2cs([0, 0, w - 1, h - 1], INPUT_SIZE)
        r = 0
        trans = transforms_forRealOcc.get_affine_transform(center, s, r, INPUT_SIZE)
        image_warp = cv2.warpAffine(
            image,
            trans,
            (int(INPUT_SIZE[1]), int(INPUT_SIZE[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        image_trans = transform(image_warp)

        image_trans = image_trans.unsqueeze(0)

        tic = time.time()
        results = model(image_trans.cuda())
        # print(results.shape)
        interp = torch.nn.Upsample(size=(INPUT_SIZE[0], INPUT_SIZE[1]), mode='bilinear', align_corners=True)
        parsing = interp(results).data.cpu().numpy()
        parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC

        parsing_preds = np.asarray(np.argmax(parsing, axis=3))
        parsing_preds = parsing_preds.squeeze(0)
        parsing_preds = np.uint8(parsing_preds)

        index = np.where(parsing_preds == 19)
        # mask_seg = np.full(parsing_preds.shape, 255)
        mask_seg = image.copy()
        for x, y in zip(index[0], index[1]):
            # mask_seg[x][y] = face_align[x][y]
            mask_seg[x][y] = 0
        mask_seg = np.uint8(mask_seg)

        color_img = visualization(image, parsing_preds, num_of_class=args.num_classes)  # , save_im=True, save_path=save_path

        # cv2.imwrite(os.path.join(SAVE_DIR, image_name), color_img)
        cv2.imwrite(os.path.join(SAVE_DIR, image_name), mask_seg)


        # cv2.imshow("mask_seg", mask_seg)
        # cv2.imshow("color_img", color_img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break





