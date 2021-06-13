# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import sys
import glob
import math
import argparse
import numpy as np
import pandas as pd
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric de instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')

    return parser.parse_args()


def calculate(npy_values, time_diff):
    # todo
    h = 8 # 8m
    # angle = 0
    # middle = h / math.cos(angle)
    middle_ratio = pd.DataFrame(np.load('../output/210612204000/0_204000_origin_disp.npy'))[1170][1086]
    middle = 300 # 300m

    depth_frame1 = npy_values[0] * middle / middle_ratio
    depth_frame2 = npy_values[1] * middle / middle_ratio

    horizontal_frame1 = math.sqrt(depth_frame1**2 - h**2)
    horizontal_frame2 = math.sqrt(depth_frame2**2 - h**2)

    speed = (horizontal_frame1 - horizontal_frame2) / time_diff
    speed = speed / 1000 * 3600

    return round(speed, 1)


def distance_estimation(args, output_dir, frame1, frame2):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "de.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    output_directory = os.path.dirname(output_dir)
    paths = [frame1[0], frame2[0]]

    print("-> Predicting on {:d} test images".format(len(paths)))

    bbox_info = [frame1[1:], frame2[1:]]
    npy_values = []

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            # scaled_disp, de = disp_to_depth(disp, 0.1, 100)
            # if args.pred_metric_depth:
            #     name_dest_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
            #     metric_depth = STEREO_SCALE_FACTOR * de.cpu().numpy()
            #     np.save(name_dest_npy, metric_depth)
            # else:
            #     name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            #     np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped de image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            name_dest_origin_npy = os.path.join(output_directory, "{}_origin_disp.npy".format(output_name))
            np.save(name_dest_origin_npy, disp_resized_np)

            df = pd.DataFrame(disp_resized_np)
            min_npy_frame = df.iloc[int(bbox_info[idx][0]):int(bbox_info[idx][2]), int(bbox_info[idx][1]):int(bbox_info[idx][3])].values.min()
            npy_values.append(min_npy_frame)

            # vmax = np.percentile(disp_resized_np, 95)
            # normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            # im = pil.fromarray(colormapped_im)
            #
            # name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            # im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))

    #todo
    # time diff : 1
    speed = calculate(npy_values, 1)

    print('-> Done!')
    print("speed : " + str(speed) + "km/h")


if __name__ == '__main__':
    args = parse_args()
    frame1 = [
        "../output/210612204000/origin/0_204000.jpg",
        "781", "858", "1021", "1078"
    ]
    frame2 = [
        "../output/210612204000/origin/1_204001.jpg",
        "687", "987", "1117", "1395"
    ]
    output_dir = "../output/210612204000/"
    distance_estimation(args, output_dir, frame1, frame2)
