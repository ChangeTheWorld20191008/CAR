import os
import argparse
import numpy as np
from glob import glob
from PIL import Image

import torch
import torch.nn as nn

import utils
from EDSR.edsr import EDSR


parser = argparse.ArgumentParser(
    description='Content Adaptive Resampler for Image downscaling')
parser.add_argument(
    '--model', type=str,
    default='/home/zhuhao/myModel/super_resolution/CAR/models/4x/usn.pth',
    help='path to the pre-trained model')
parser.add_argument(
    '--img_dir', type=str, help='path to the images to be restore')
parser.add_argument(
    '--scale', type=int, default=4, help='downscale factor')
parser.add_argument(
    '--output_dir', type=str, help='path to store results')
parser.add_argument(
    '--benchmark', type=bool, default=True, help='report benchmark results')
args = parser.parse_args()


SCALE = args.scale
KSIZE = 3 * SCALE + 1
OFFSET_UNIT = SCALE
BENCHMARK = args.benchmark

upscale_net = EDSR(32, 256, scale=SCALE).cuda()

upscale_net = nn.DataParallel(upscale_net, [0])

upscale_net.load_state_dict(torch.load(args.model))
torch.set_grad_enabled(False)


def run(downscaled_img, name, save_imgs=False, save_dir=None):
    upscale_net.eval()

    downscaled_img = torch.clamp(downscaled_img, 0, 1)
    downscaled_img = torch.round(downscaled_img * 255)
    reconstructed_img = upscale_net(downscaled_img / 255.0)

    reconstructed_img = torch.clamp(reconstructed_img, 0, 1) * 255
    reconstructed_img = \
        reconstructed_img.data.cpu().numpy().transpose(0, 2, 3, 1)
    reconstructed_img = np.uint8(reconstructed_img)

    recon_img = reconstructed_img[0, ...].squeeze()

    if save_imgs and save_dir:
        img = Image.fromarray(recon_img)
        img.save(os.path.join(save_dir, name + '.jpg'))


if __name__ == '__main__':
    img_list = glob(os.path.join(args.img_dir, '**', '*.png'), recursive=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    img_num = len(img_list)
    for i in range(img_num):
        img_file = img_list[i]

        name = os.path.basename(img_file)
        name = os.path.splitext(name)[0]

        img = utils.load_img(img_file)

        run(img, name, save_imgs=True, save_dir=args.output_dir)

        i += 1
        left_num = img_num - i
        print(f"[INFO]: finish {name}, left {left_num}")
