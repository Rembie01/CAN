import os
import time
import argparse
import random
import torch
import numpy as np
from tensorboardX import SummaryWriter

from utils import load_config, save_checkpoint, load_checkpoint
from dataset import get_mlhme_dataset, get_crohme_dataset
from models.can import CAN
from training import train, eval

path = 'C:/Users/REM/OneDrive - Televic Group NV/Documents/CAN_test'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('--dataset', default='CROHME', type=str, help='数据集名称')
    parser.add_argument('--no_check', action='store_true', help='测试代码选项')
    args = parser.parse_args()

    if not args.dataset:
        print('No dataset specified')
        exit(-1)

    if args.dataset == 'CROHME':
        config_file = 'config.yaml'

    if args.dataset == 'MLHME':
        config_file = 'config_mlhme.yaml'

    """Config"""
    params = load_config(config_file)

    """Seed"""
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device

    print('Using', device)

    if args.dataset == 'CROHME':
        train_loader, eval_loader = get_crohme_dataset(params)

    if args.dataset == 'MLHME':
        train_loader, eval_loader = get_mlhme_dataset(params)

    model = CAN(params)
    now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    model.name = f'{params["experiment"]}_{now}_decoder-{params["decoder"]["net"]}'

    print('Model name:', model.name)
    model = model.to(device)

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
        if not os.path.exists(os.path.join(params['checkpoint_dir'], model.name)):
            os.makedirs(os.path.join(params['checkpoint_dir'], model.name), exist_ok=True)
        
        print('Copying config to checkpoints...')
        os.system(f'copy {config_file} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml')

    # Train
    if args.dataset == 'CROHME' or args.dataset == 'MLHME':
        best_score, init_epoch = -1, 0

        print('Start training')

        for epoch in range(init_epoch, params['epochs']):
            train_loss, train_word_score, train_exprate = train(params, model, optimizer, epoch, train_loader, writer=writer)

            if epoch >= params['valid_start']:
                eval_loss, eval_word_score, eval_exprate = eval(params, model, epoch, eval_loader, writer=writer)
                print(f'(Epoch: {epoch+1}) loss: {eval_loss:.4f} word score: {eval_word_score:.4f} ExpRate: {eval_exprate:.4f}')
                if eval_exprate > best_score and not args.no_check and epoch >= params['save_start']:
                    best_score = eval_exprate
                    save_checkpoint(model, optimizer, eval_word_score, eval_exprate, epoch+1,
                                    optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'])
