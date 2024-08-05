import os
import time
import argparse
import random
import torch
import numpy as np
from tensorboardX import SummaryWriter

from utils import load_config, save_checkpoint, load_checkpoint
from dataset import get_mlhme_dataset, get_televic_dataset
from models.can import CAN
from training import train, eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('--dataset', default='CROHME', type=str, help='the dataset to run the model on, dataloaders work depending on the specific dataset')
    parser.add_argument('--no_check', action='store_true', help='if true, does not store checkpoints')
    args = parser.parse_args()

    if not args.dataset:
        print('No dataset specified')
        exit(-1)

    if args.dataset == 'MLHME':
        config_file = 'config_mlhme_.yaml'
    
    elif args.dataset =='MLHMED':
        config_file = 'config_mlhme_desktop.yaml'
    
    elif args.dataset =='MLHMEVSC':
        config_file = 'config_mlhme_vsc.yaml'
    
    elif args.dataset == 'Televic':
        config_file = 'config_televic.yaml'
    
    elif args.dataset == 'TelevicD':
        config_file = 'config_televic_desktop.yaml'
    
    elif args.dataset == 'TelevicVSC':
        config_file = 'config_televic_vsc.yaml'
    
    else:
        print('Dataset not recognized')
        exit(-1)
    
    """Config"""
    params = load_config(config_file)

    """Seed"""
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    params['device1'] = device1
    params['device2'] = device2

    print('Using', device1, 'for DenseNet')
    print('Using', device2, 'for rest of network')

    if args.dataset == 'MLHME' or args.dataset == 'MLHMED' or args.dataset == 'MLHMEVSC':
        train_loader, eval_loader, params = get_mlhme_dataset(params)
    
    elif args.dataset == 'Televic' or args.dataset == 'TelevicD' or args.dataset == 'TelevicVSC':
        train_loader, eval_loader, params = get_televic_dataset(params)

    model = CAN(params)
    now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    model.name = f'{params["experiment"]}_{now}_decoder-{params["decoder"]["net"]}'

    print('Model name:', model.name)

    if args.no_check:
        writer = None
    else:
        writer = SummaryWriter(f'{params["log_dir"]}/{model.name}')

    optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                        eps=float(params['eps']), weight_decay=float(params['weight_decay']))

    if params['finetune']:
        print('Finetuning')
        print(f'Loading model: {params["checkpoint"]}')
        load_checkpoint(model, optimizer, params['checkpoint'])

    if not args.no_check:
        from sys import platform
        if not os.path.exists(os.path.join(params['checkpoint_dir'], model.name)):
            os.makedirs(os.path.join(params['checkpoint_dir'], model.name), exist_ok=True)
        
        print('Copying config to checkpoints...')
        if platform == "linux" or platform == "linux2":
            os.system(f'cp {config_file} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml')
        elif platform == "win32":
            os.system(f'copy {config_file} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml')
        else:
            print('Could not copy config')

    # Train
    if args.dataset == 'MLHME' or args.dataset == 'MLHMED' or args.dataset == 'Televic'or args.dataset == 'TelevicVSC':
        best_score, init_epoch = 0, 0

        print('Start training')

        for epoch in range(init_epoch, params['epochs']):
            train_loss, train_word_score, train_exprate = train(params, model, optimizer, epoch, train_loader, writer=writer)

            if epoch >= params['valid_start']:
                eval_loss, eval_word_score, eval_exprate = eval(params, model, epoch, eval_loader, writer=writer)
                print(f'(Epoch: {epoch+1}) loss: {eval_loss:.4f} word score: {eval_word_score:.4f} ExpRate: {eval_exprate:.4f}')
                if (eval_exprate + eval_word_score * 0.1) > best_score and not args.no_check and epoch >= params['save_start']:
                    best_score = eval_exprate + eval_word_score * 0.1
                    save_checkpoint(model, optimizer, eval_word_score, eval_exprate, epoch+1,
                                    optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'])
