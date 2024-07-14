import os
import cv2
import argparse
import torch
import json
import pickle as pkl
from tqdm import tqdm
import time
import numpy as np

from utils import load_config, load_checkpoint, compute_edit_distance
from models.infer_model import Inference
from dataset2 import Words, MLHMEDataset, TelevicDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model testing')
    parser.add_argument('--dataset', default='CROHME', type=str, help='the dataset to run the model on, dataloaders work depending on the specific dataset')
    parser.add_argument('--image_path', default='datasets/CROHME/14_test_images.pkl', type=str, help='测试image路径')
    parser.add_argument('--label_path', default='datasets/CROHME/14_test_labels.txt', type=str, help='测试label路径')

    parser.add_argument('--draw_map', action='store_true', default=False)
    args = parser.parse_args()

    if not args.dataset:
        print('No dataset specified')
        exit(-1)

    if args.dataset == 'MLHME':
        config_file = 'config_mlhme_test.yaml'

    elif args.dataset =='MLHMED':
        config_file = 'config_mlhme_desktop_test.yaml'
    
    elif args.dataset =='Televic':
        config_file = 'config_televic_desktop.yaml'
    
    else:
        print('Dataset not recognized')
        exit(-1)

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
    model = Inference(params, draw_map=args.draw_map)
    model = model.to(device)

    load_checkpoint(model, None, params['checkpoint'])
    model.eval()
    
    if args.dataset == 'MLHME' or args.dataset =='MLHMED':
        words = Words(params['word_path'])
        test_dataset = MLHMEDataset(params, params['test_label_path'], words, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=1)
    
    elif args.dataset == 'Televic':
        words = Words(params['word_path'])
        test_dataset = TelevicDataset(params, params['test_label_path'], params['question_file'], words)
        test_loader = DataLoader(test_dataset, batch_size=1)
    
    else:
        print('Dataset not recognized')
        exit(-1)
    
    line_right = 0
    e1, e2, e3 = 0, 0, 0
    bad_case = {}
    model_time = 0
    mae_sum, mse_sum = 0, 0
    
    with tqdm(test_loader) as pbar, torch.no_grad():
        for idx, (image, label, _) in enumerate(pbar):
            input_labels = label
            # print(input_labels.tolist()[0])
            labels = ' '.join(str(label.tolist()[0]))
            name = str(idx)
            name = name.split('.')[0] if name.endswith('jpg') else name
            img = image.to(device)
            a = time.time()
            
            input_labels = input_labels.unsqueeze(0).to(device)

            probs, _, mae, mse = model(img, input_labels, os.path.join(params['decoder']['net'], name))
            mae_sum += mae
            mse_sum += mse
            model_time += (time.time() - a)

            prediction = words.decode(probs)
            ground_truth = input_labels[0][0][:-1]
            ground_truth = words.decode(input_labels[0][0][:-1]) #remove the last element (eos token)
            if prediction == ground_truth:
                line_right += 1
            else:
                bad_case[name] = {
                    'label': ground_truth,
                    'predi': prediction
                }
                # print(name, prediction, labels)

            distance = compute_edit_distance(prediction, ground_truth)
            if distance <= 1:
                e1 += 1
            if distance <= 2:
                e2 += 1
            if distance <= 3:
                e3 += 1

    print(f'model time: {model_time}')
    print(f'ExpRate: {line_right / len(test_loader)}')
    print(f'mae: {mae_sum / len(test_loader)}')
    print(f'mse: {mse_sum / len(test_loader)}')
    print(f'e1: {e1 / len(test_loader)}')
    print(f'e2: {e2 / len(test_loader)}')
    print(f'e3: {e3 / len(test_loader)}')

    with open(f'{params["decoder"]["net"]}_bad_case.json','w', encoding='utf-8') as f:
        json.dump(bad_case,f,ensure_ascii=False)
