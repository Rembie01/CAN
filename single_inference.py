import os
import cv2
import argparse
import torch
import json
import pickle as pkl
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from utils import load_config, load_checkpoint, compute_edit_distance
from models.infer_model_single import Inference
from dataset2 import Words
from torchvision.transforms.functional import to_tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model testing')
    parser.add_argument('--image_path', default='datasets/CROHME/14_test_images.pkl', type=str, help='测试image路径')

    args = parser.parse_args()

    config_file = 'config_televic_desktop_test.yaml'

    """Config"""
    params = load_config(config_file)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    words = Words(params['word_path'])
    print(len(words))
    params['word_num'] = len(words)

    if 'use_label_mask' not in params:
        params['use_label_mask'] = False
    print(params['decoder']['net'])
    model = Inference(params)
    model = model.to(device)

    load_checkpoint(model, None, params['checkpoint'])
    model.eval()
    try:
        im = Image.open(args.image_path).convert('L')
    except:
        raise IOError
    img = to_tensor(im)
    im.close()
    img = 1 - img
    plt.imshow(img.permute(1, 2, 0))
    img = img.unsqueeze(0)
    print(img.shape)
    
    with torch.no_grad():
        img = img.to(device)

        probs, _ = model(img, os.path.join(params['decoder']['net']))

        prediction = words.decode(probs)

    print(f'Prediction: {prediction}')
