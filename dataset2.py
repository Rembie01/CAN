import json
import re
import os
import torch
import time
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, RandomSampler
from PIL import Image
from pathlib import Path
from torchvision.transforms.functional import to_tensor


class HMERDataset(Dataset):
    def __init__(self, params, image_path, label_path, words, is_train=True):
        super(HMERDataset, self).__init__()
        if image_path.endswith('.pkl'):
            with open(image_path, 'rb') as f:
                self.images = pkl.load(f)
        elif image_path.endswith('.list'):
            with open(image_path, 'r') as f:
                lines = f.readlines()
            self.images = {}
            print(f'data files: {lines}')
            for line in lines:
                name = line.strip()
                print(f'loading data file: {name}')
                start = time.time()
                with open(name, 'rb') as f:
                    images = pkl.load(f)
                self.images.update(images)
                print(f'loading {name} cost: {time.time() - start:.2f} seconds!')

        with open(label_path, 'r') as f:
            self.labels = f.readlines()

        self.words = words
        self.is_train = is_train
        self.params = params

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        image = self.images[name]
        image = torch.Tensor(255-image) / 255
        image = image.unsqueeze(0)
        labels.append('eos')
        words = self.words.encode(labels)
        words = torch.LongTensor(words)
        return image, words


class MLHMEDataset(Dataset):
    # als istrain = true, append eos, anders niet
    def __init__(self, params, labels_path, words, debug=False):
        self.params = params
        self.words = words
        self.image_paths = []
        self.image_labels = []
        self.image_root = Path(labels_path).parent / 'train_images'
        self.debug = debug

        with open(labels_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
        
        for line in lines:
            image_name, *labels = line.strip().split()
            labels.append('eos')
            self.image_paths.append(self.image_root / image_name)
            self.image_labels.append(labels)
    
        # print(self.image_labels[0], self.image_paths[0])

    def __len__(self):
        assert len(self.image_paths) == len(self.image_labels)
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, labels = self.image_paths[idx], self.image_labels[idx]
        im = Image.open(image_path).convert('L')
        img = to_tensor(im)
        im.close()
        img = 1 - img
        if self.debug:
            words, _ = self.words.encode(labels)
        else:
            words = self.words.encode(labels)
        words = torch.LongTensor(words)
        return img, words


class TelevicDataset(Dataset):
    def __init__(self, params, json_dir, words, debug=False):
        self.params = params
        self.words = words
        self.image_paths = []
        self.image_labels = []
        self.image_root = Path(json_dir).parent / 'png_files'
        self.debug = debug

        for json_file in os.listdir(json_dir):
            if json_file.endswith('.json'):
                with open(Path(json_dir) / json_file, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if 'latex_styled' in data:
                        self.image_labels.append(data['latex_styled'])
                image_name = json_file.replace('.json', '.png')
                self.image_paths.append(self.image_root / image_name)
        
        if debug:
            print(str(self.image_labels[0]).replace(' ',''), self.image_paths[0])

        combined_pattern = r'(\\sqrt)|(\\leftrightarrow)|(\\left\\{)|(\\left[\(\)\{\}\[\]\|.])|(\\left(\\rvert)?(\\lvert)?)|(\\right[\(\)\{\}\[\]\|.]?(\\rvert)?(\\lvert)?)|(\\text {([\w .?!,\)\()]+)})|(\\frac)|((\\[a-zA-Z]+)({[a-zA-Z~]+})?({[a-zA-Z\|]+})?)|([+-=*^_{}])|([\d\w])|(\\{2})|([\[\]\(\)\|\\&<>?!~%\"\'\&])'

        temp = []
        for label in self.image_labels:
            parts = re.findall(combined_pattern, label)
            comb_parts = []
            for tuple in parts:
                nonzero_elements = list(x for x in tuple if x != '')
                comb_parts.append(nonzero_elements[0])
            temp.append(comb_parts)
        
        for i in range(len(self.image_labels)):
            one = ''.join(temp[i]).replace(' ', '')
            two = self.image_labels[i].replace(' ', '')
            assert one == two, f'index {i} : {self.image_paths[i]} : {one} is not equal to {two}'
            temp[i].append('eos')
        self.image_labels = temp

        assert len(self.image_paths) == len(self.image_labels)

        """
        for line in lines:
            image_name, *labels = line.strip().split()
            labels.append('eos')
        """

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, labels = self.image_paths[idx], self.image_labels[idx]
        if self.debug:
            print(image_path)
        im = Image.open(image_path).convert('L')
        img = to_tensor(im)
        im.close()
        img = 1 - img

        # max, min = torch.max(img), torch.min(img)
        # img  = (img-min)/(max-min) 

        if self.debug:
            words, altered = self.words.encode(labels)
            print(f'Altered {altered}')
            words = torch.LongTensor(words)
            return img, words, altered
        else:
            words = self.words.encode(labels)
            words = torch.LongTensor(words)
            return img, words


def get_crohme_dataset(params):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    print(f"训练数据路径 images: {params['train_image_path']} labels: {params['train_label_path']}")
    print(f"验证数据路径 images: {params['eval_image_path']} labels: {params['eval_label_path']}")

    train_dataset = HMERDataset(params, params['train_image_path'], params['train_label_path'], words, is_train=True)
    eval_dataset = HMERDataset(params, params['eval_image_path'], params['eval_label_path'], words, is_train=False)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
          f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
    return train_loader, eval_loader


def get_mlhme_dataset(params):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    print(f"Train labels: {params['train_label_path']}")
    print(f"Eval labels: {params['eval_label_path']}")

    train_dataset = MLHMEDataset(params, params['train_label_path'], words)
    eval_dataset = MLHMEDataset(params, params['eval_label_path'], words)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)

    print(f'Train dataset: {len(train_dataset)}, Train steps: {len(train_loader)} \n'
          f'Eval dataset: {len(eval_dataset)}, Eval steps: {len(eval_loader)}')
    return train_loader, eval_loader

def get_televic_dataset(params):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    print(f"Train labels: {params['train_label_path']}")
    print(f"Eval labels: {params['eval_label_path']}")

    train_dataset = MLHMEDataset(params, params['train_label_path'], words)
    eval_dataset = MLHMEDataset(params, params['eval_label_path'], words)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)

    print(f'Train dataset: {len(train_dataset)}, Train steps: {len(train_loader)} \n'
          f'Eval dataset: {len(eval_dataset)}, Eval steps: {len(eval_loader)}')
    return train_loader, eval_loader


def collate_fn(batch_images):
    max_width, max_height, max_length = 0, 0, 0
    _, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
    return images, image_masks, labels, labels_masks


class Words:
    def __init__(self, words_path, debug=False):
        with open(words_path, encoding='utf8') as f:
            words = f.readlines()
        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}
        self.debug = debug

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        label_index = []
        altered = False
        for item in labels:
            try:
                label_index.append(self.words_dict[item])
            except KeyError:
                altered = True
                if item.startswith('\\begin{array}') or item == '\\begin{aligned}':
                    label_index.append(self.words_dict['\\begin{matrix}'])
                elif item == '\\end{array}' or item == '\\end{aligned}':
                    label_index.append(self.words_dict['\\end{matrix}'])
                elif item.startswith('\\text'):
                    if self.debug:
                        print(item)
                elif item.startswith('\\vec'):
                    if self.debug:
                        print(item)
                elif item =='\\top':
                    label_index.append(self.words_dict['T'])
                elif item == '\\quad' or item == '\\qquad' or item == '\\':
                    pass
                    # label_index.append(self.words_dict['\'])
                elif item == '\\Leftrightarrow':
                    label_index.append(self.words_dict['\\Rightarrow'])
                elif item == '\\left(' or item == '\\right(':
                    label_index.append(self.words_dict['('])
                elif item == '\\left)' or item == '\\right)':
                    label_index.append(self.words_dict[')'])
                elif item == '\\left|' or item == '\\right|' or item == '\\left\\rvert' or item == '\\right\\rvert' or item == '\\left\\lvert' or item == '\\right\\lvert':
                    label_index.append(self.words_dict['|'])
                elif item == '\\left.' or item == '\\right.':
                    label_index.append(self.words_dict['.'])
                elif item.startswith('\\operatorname') or item.startswith('\\mathrm') or item == '\\operatorname{dom}' or item == '\\operatorname{asin}' or item == '\\operatorname{amplitude}' or item == '\\operatorname{dem}' or item == '\\operatorname{Nan}' or item == '\\mathrm{man}' or item == '\\operatorname{cs}' or item == '\\operatorname{sen}' or item == '\\operatorname{dam}' or item == '\\operatorname{yin}' or item == '\\operatorname{set}':
                    if self.debug:
                        print(item)
                elif item == '&':
                    pass
                elif item == '%':
                    pass
                elif item == '\\underbrace':
                    pass
                elif item == '\\hat{i}':
                    label_index.append(self.words_dict['\\uparrow'])
                elif item == '\\hat{x}':
                    label_index.append(self.words_dict['X'])
                elif item == '\\hat{y}':
                    label_index.append(self.words_dict['Y'])
                elif item == '\\hat{z}':
                    label_index.append(self.words_dict['Z'])
                elif item == '\\backslash':
                    label_index.append(self.words_dict['|'])
                elif item == '\\mathrm{I}':
                    label_index.append(self.words_dict['\\left['])
                elif item == '\\mathbb{Z}':
                    label_index.append(self.words_dict['Z'])
                elif item == '\\mathbb{R}':
                    label_index.append(self.words_dict['R'])
                elif item == '\\mathbb{k}':
                    label_index.append(self.words_dict['K'])
                elif item == '\\mathbb{C}':
                    label_index.append(self.words_dict['C'])
                elif item == '\\ldots':
                    label_index.append(self.words_dict['\\cdots'])
                elif item == '\\right':
                    pass
                elif item == '\\rangle' or item == '\\langle':
                    pass
                elif item == '\\simeq':
                    label_index.append(self.words_dict['='])
                elif item == '\\hat':
                    pass
                elif item == '\\longrightarrow' or item == '\\hookleftarrow' or item == '\\leftrightarrow' or item == '\\mapsto' or item == '\\longleftrightarrow' or '\\ll':
                    label_index.append(self.words_dict['\\rightarrow'])
                elif item == '\\Downarrow':
                    label_index.append(self.words_dict['\\downarrow'])
                elif item == '\\wedge':
                    pass
                elif item == '\\vee':
                    pass
                elif item == '\\Gamma':
                    label_index.append(self.words_dict['V'])
                
                elif item == '\\searrow':
                    label_index.append(self.words_dict['\\downarrow'])
                elif item == '\\hline':
                    pass
                elif item == '\\Phi':
                    pass
                elif item == '\\dot{B}':
                    label_index.append(self.words_dict['B'])
                else:
                    raise KeyError(f'\'{item}\' is not a key, comes from {labels}')
        if self.debug:
            return label_index, altered
        else:
            return label_index

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label


collate_fn_dict = {
    'collate_fn': collate_fn
}

def test():
    words = Words(r'D:\Masterproef\echmer\words.txt', debug=True)
    test1 = MLHMEDataset(0, r'D:\Masterproef\echmer\data\MLHME-38K\train_set\all_labels.txt', words, debug=True)
    test2 = TelevicDataset(0, r'D:\Masterproef\echmer\data\Televic\mathpix_processed_data\mathpix_output_adjusted', words, debug=True)

    start_time = time.time()
    for i in range(len(test1)):
        item = test1.__getitem__(i)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.30f} seconds")

    start_time = time.time()
    altered = 0
    for i in range(len(test2)):
        im, lab, alt = test2.__getitem__(i)
        if not alt:
            altered += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Altered: {altered}/{len(test2)}')
    print(f"Elapsed time: {elapsed_time:.3f} seconds")

if __name__ == "__main__":
    test()