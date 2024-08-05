import torch
import time
import pickle as pkl
import json
import re
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler
from PIL import Image
from pathlib import Path
from torchvision.transforms.functional import to_tensor

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def mean_pooling(model_output: Tensor, attention_mask: Tensor) -> Tensor:
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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
    def __init__(self, params, labels_path, words):
        self.params = params
        self.words = words
        self.image_paths = []
        self.image_labels = []
        self.image_root = Path(labels_path).parent / 'train_images'

        with open(labels_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
        
        for line in lines:
            image_name, *labels = line.strip().split()
            labels.append('eos')
            self.image_paths.append(self.image_root / image_name)
            self.image_labels.append(labels)

    def __len__(self):
        assert len(self.image_paths) == len(self.image_labels)
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, labels = self.image_paths[idx], self.image_labels[idx]
        im = Image.open(image_path).convert('L')
        img = to_tensor(im)
        im.close()
        img = 1 - img
        words, _ = self.words.encode(labels)
        words = torch.LongTensor(words)
        return img, words


class TelevicDataset(Dataset):
    def __init__(self, params, jsons_file, words, questions_file=None):
        self.params = params
        self.invert = params['invert_image']
        self.words = words
        self.image_paths = []
        self.image_labels = []
        self.question_ids = []
        self.question_embeddings = []
        self.image_root = Path(jsons_file).parent / 'jpg_files_resized'
        self.json_root = Path(jsons_file).parent / 'mathpix_output_adjusted'
        self.questions_file = questions_file

        if questions_file is not None:
            import os
            from transformers import AutoTokenizer, AutoModel
            import gc
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            tokenizer = AutoTokenizer.from_pretrained("witiko/mathberta")
            model = AutoModel.from_pretrained("witiko/mathberta")

            with open(questions_file, 'r', encoding='utf-8') as questions:
                lines = questions.readlines()

            for line in lines:
                question_id, *content = line.strip().split()
                self.question_ids.append(question_id)
                question = ' '.join(content)
                tokenized = tokenizer(question, max_length=512, padding=True, truncation=True, return_tensors='pt')
                outputs = model(**tokenized)
                embedding = average_pool(outputs.last_hidden_state, tokenized['attention_mask']) #average pool or mean pool?
                self.question_embeddings.append(embedding.clone().detach())
            
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        with open(jsons_file, 'r', encoding='utf-8') as jsons:
            lines = jsons.readlines()

        for json_file in lines:
            if json_file.endswith('.json\n'):
                json_file = json_file.strip('\n')
                with open(Path(self.json_root) / json_file, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if 'latex_styled' in data:
                        self.image_labels.append(data['latex_styled'])
                    else:
                        print('not found')
                image_name = json_file.replace('.json', '.jpg')
                self.image_paths.append(self.image_root / image_name)
        
        # combined_pattern = r'(\\{2})|(\\[{}&%|]))|(\\sqrt)|(\\leftrightarrow)|(\\left\\{)|(\\left[\(\)\{\}\[\]\|.])|(\\left(\\rvert)?(\\lvert)?)|(\\right[\(\)\{\}\[\]\|.]?(\\rvert)?(\\lvert)?)|(\\text ?{([\w .?!,\)\()]+) ?})|(\\frac)|((\\[a-zA-Z]+)({[a-zA-Z~]+})?({[a-zA-Z\|]+})?)|([+-=*^_{}])|([\d\w])|([\[\]\(\)\|\\&<>?!~%\"\'\&])'
        combined_pattern = r'(\\ ?,)|(\\ ?{)|(\\ ?})|(\\ ?%)|(\\ ?\|)|(\\sqrt)|(\\leftrightarrow)|(\\left\\{)|(\\left[\(\)\{\}\[\]\|.]?(\\rvert)?(\\lvert)?)|(\\right[\(\)\{\}\[\]\|.]?(\\rvert)?(\\lvert)?)|(\\text ?{([\w .?!,\)\()]+) ?})|(\\frac)|((\\[a-zA-Z]+)({[a-zA-Z~]+})?({[a-zA-Z\|]+})?)|([+-=*^_{}])|([\d\w])|(\\{2})|([\[\]\(\)\|\\&<>?!~%\"\'\&])'
        
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


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, labels = self.image_paths[idx], self.image_labels[idx]
        im = Image.open(image_path).convert('L')
        img = to_tensor(im)
        im.close()
        if self.invert:
            img = 1 - img
        
        if self.questions_file is not None:
            combined_pattern = r'canvas_(\d+)_(\d+)'
            parts = re.findall(combined_pattern, image_path.__str__())
            embedding = self.question_embeddings[self.question_ids.index(parts[0][1])].flatten()
            words, _ = self.words.encode(labels)
            words = torch.LongTensor(words)
            return img, words, embedding
        else:
            words, _ = self.words.encode(labels)
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
    return train_loader, eval_loader, params


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
    return train_loader, eval_loader, params


def get_televic_dataset(params):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    print(f"Train labels: {params['train_label_path']}")
    print(f"Eval labels: {params['eval_label_path']}")

    train_dataset = TelevicDataset(params, params['train_label_path'], words)
    eval_dataset = TelevicDataset(params, params['eval_label_path'], words)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)

    print(f'Train dataset: {len(train_dataset)}, Train steps: {len(train_loader)} \n'
          f'Eval dataset: {len(eval_dataset)}, Eval steps: {len(eval_loader)}')
    return train_loader, eval_loader, params


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

def collate_fn_context(batch_images):
    max_width, max_height, max_length = 0, 0, 0
    context_len = len(batch_images[0][2])
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
    context = torch.zeros((len(proper_items), context_len))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
        context[i] = proper_items[i][2]
    return images, image_masks, labels, labels_masks, context

class Words:
    def __init__(self, words_path):
        with open(words_path, encoding='utf8') as f:
            words = f.readlines()
        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}
        self.text_expression = r'\\text ?{ ?([\w .?!,\)\():;]+) ?}'

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
                    text = re.findall(self.text_expression, item)
                    for character in text[0]:
                        try:
                            label_index.append(self.words_dict[character])
                        except KeyError:
                            if character == ' ':
                                pass
                            else:
                                raise KeyError
                elif item.startswith('\\vec'):
                    expression = r'\\vec ?{ ?([\w]+) ?}'
                    text = re.findall(expression, item)
                    label_index.append(self.words_dict['\\overrightarrow'])
                    label_index.append(self.words_dict['{'])
                    for character in text:
                        try:
                            label_index.append(self.words_dict[character])
                        except KeyError:
                            print(character)
                            raise KeyError
                    label_index.append(self.words_dict['}'])
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
                elif item.startswith('\\operatorname') or item.startswith('\\mathrm'):
                    expression = r'{ ?([\w~]+) ?}'
                    text = re.findall(expression, item)
                    for character in text[0]:
                        try:
                            label_index.append(self.words_dict[character])
                        except KeyError:
                            if character == '~':
                                pass
                            else:
                                raise KeyError
                elif item == '&':
                    pass
                elif item == '%':
                    label_index.append(self.words_dict['\\%'])
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
                elif item == '\\rangle':
                    label_index.append(self.words_dict['>'])
                elif item == '\\langle':
                    label_index.append(self.words_dict['<'])
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
        return label_index, altered
    
    def encode2(self, labels):
        label_index = []
        altered = False
        for item in labels:
            try:
                label_index.append(self.words_dict[item])
            except KeyError:
                if item.startswith('\\begin{array}') or item == '\\begin{aligned}':
                    label_index.append(self.words_dict['\\begin{matrix}'])

                elif item == '\\end{array}' or item == '\\end{aligned}':
                    label_index.append(self.words_dict['\\end{matrix}'])
                
                elif item.startswith('\\text'):
                    text = re.findall(self.text_expression, item)
                    for character in text[0]:
                        try:
                            _ = self.words_dict[character]
                            label_index.append(self.words_dict[character])
                        except KeyError:
                            if character == ' ':
                                pass
                            else:
                                raise KeyError
                elif item.startswith('\\operatorname'):
                    expression = r'{ ?([\w~]+) ?}'
                    text = re.findall(expression, item)
                    for character in text[0]:
                        try:
                            _ = self.words_dict[character]
                            label_index.append(self.words_dict[character])
                        except KeyError:
                            if character == '~':
                                pass
                            else:
                                raise KeyError
                elif item.startswith('\\mathrm'):
                    expression = r'{ ?([\w~]+) ?}'
                    text = re.findall(expression, item)
                    for character in text[0]:
                        try:
                            label_index.append(self.words_dict[character])
                        except KeyError:
                            if character == '~':
                                pass
                            else:
                                raise KeyError
                elif item.startswith('\\vec'):
                    expression = r'\\vec ?{ ?([\w]+) ?}'
                    text = re.findall(expression, item)
                    label_index.append(self.words_dict['\\vec'])
                    label_index.append(self.words_dict['{'])
                    for character in text:
                        try:
                            label_index.append(self.words_dict[character])
                        except KeyError:
                            raise KeyError
                    label_index.append(self.words_dict['}'])
                elif item.startswith('\\hat'):
                    expression = r'\\hat ?{ ?([\w]+) ?}'
                    text = re.findall(expression, item)
                    label_index.append(self.words_dict['\\hat'])
                    label_index.append(self.words_dict['{'])
                    for character in text:
                        try:
                            label_index.append(self.words_dict[character])
                        except KeyError:
                            print(character)
                            raise KeyError
                    label_index.append(self.words_dict['}'])
                elif item == '\\ldots':
                    label_index.append(self.words_dict['\\cdots'])
                elif item == '\\left(' or item == '\\right(':
                    label_index.append(self.words_dict['('])
                elif item == '\\left)' or item == '\\right)':
                    label_index.append(self.words_dict[')'])
                elif item == '\\left|' or item == '\\right|' or item == '\\left\\rvert' or item == '\\right\\rvert' or item == '\\left\\lvert' or item == '\\right\\lvert':
                    label_index.append(self.words_dict['|'])
                elif item == '\\left.' or item == '\\right.':
                    label_index.append(self.words_dict['.'])
                elif item == '\\left[' or item == '\\right[':
                    label_index.append(self.words_dict['['])
                elif item == '\\left]' or item == '\\right]':
                    label_index.append(self.words_dict[']'])
                elif item == '\\left\\{':
                    label_index.append(self.words_dict['\\{'])

                elif item == '\\right' or item == '\\left':
                    pass #formatting

                elif item == '\\left\\lvert' or item == '\\right\\lvert':
                    label_index.append(self.words_dict['\\lvert'])
                elif item == '\\left\\rvert' or item == '\\right\\rvert':
                    label_index.append(self.words_dict['\\rvert'])

                elif item == '\\leqslant':
                    label_index.append(self.words_dict['\\leq'])
                elif item == '\\geqslant':
                    label_index.append(self.words_dict['\\geq'])
                else:
                    raise ValueError(item, 'is not recognized in the dictionary', labels)
                
        return label_index, altered
    
    def encode3(self, labels):
        label_index = []
        altered = False
        for item in labels:
            try:
                _ = self.words_dict[item]
                label_index.append(item)
            except KeyError:
                label_index.append(item)
                
        return label_index, altered

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label
    
    def decode2(self, label_index):
        label = [self.words_index_dict[int(item)] for item in label_index]
        return label


collate_fn_dict = {
    'collate_fn': collate_fn,
    'collate_fn_context': collate_fn_context
}

def test1():
    words = Words(r'/data/leuven/351/vsc35170/echmer/data/words.txt')
    test1 = TelevicDataset(0, r'/data/leuven/351/vsc35170/echmer/data/Televic/mathpix_processed_data/train.txt', r'/data/leuven/351/vsc35170/echmer/data/Televic/formatted_questions.txt', words)
    
    start_time = time.time()
    altered = 0
    for i in range(len(test1)):
        print(i,'/',len(test1))
        im, lab, emb = test1.__getitem__(i)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Altered: {altered}/{len(test1)}')
    print(f"Elapsed time: {elapsed_time:.3f} seconds")


def test2():
    from utils import load_config
    config_file = r'/data/leuven/351/vsc35170/echmer/config_televic_desktop_inference.yaml'
    params = load_config(config_file)
    train_loader, _ = get_televic_dataset(params)
    for i, batch in enumerate(train_loader):
        print(str(i+1),'/',len(train_loader))
        images, image_masks, labels, labels_masks, contexts = batch
    print('Images:', images[0])
    print('Context:', contexts[0])


if __name__ == '__main__':
    test2()
