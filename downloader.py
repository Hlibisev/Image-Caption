import os
import os.path
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from torch.utils.data import Subset
from torchvision import transforms
from PIL import Image
from pytorch_pretrained_bert import BertTokenizer
from pycocotools.coco import COCO
from collections import Counter
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


class CoCoDataset(data.Dataset):
    def __init__(self, transform=None,
                 start_word="[CLS] ",
                 end_word=" [SEP]",
                 unk_word="[unused2]",
                 pad_word="[PAD]",
                 img_folder='/content/train2014/Users/hlibisev/Downloads/train2014',
                 annotations_file="/content/annotations/captions_train2014.json",
                 vocab_threshold=5):

        """ Create dataset to work with COCO """

        self.transform = transform
        self.vocab_threshold = vocab_threshold
        self.img_folder = img_folder
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_word = pad_word

        self.coco = COCO(annotations_file)
        self.ids = list(self.coco.anns.keys())

        self.get_id_to_token()
        self.get_transform()

    def __getitem__(self, index):
        """ Get image and caption """
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        # Convert image to tensor and pre-process using transform
        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        image = self.transform(image)

        # Convert caption to tensor of word ids.
        tokens = self.tokenizer.tokenize(self.start_word + str(caption).lower() + self.end_word)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens = torch.Tensor(tokens).long()

        # return pre-processed image and caption tensors
        return image, tokens

    def __len__(self):
        return len(self.ids)

    def get_id_to_token(self):

        """ Creating array to associate the index of a word with index of token for bert """
        # Counting using word
        counter = Counter()
        for i, id in enumerate(tqdm(self.ids)):
            caption = str(self.coco.anns[id]['caption'])
            tokens = self.tokenizer.tokenize(caption.lower())
            counter.update(tokens)

        # Throwing away rare words
        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        self.id_to_token = np.zeros(len(words))
        self.token_to_id = {}

        # Filling array in
        for i in range(len(words)):
            key = self.tokenizer.convert_tokens_to_ids([words[i]])[0]
            self.token_to_id[key] = i
            self.id_to_token[i] = key

    def get_transform(self):

        # If None, apply standard transforms
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),  # smaller edge of image resized to 256
                transforms.RandomCrop(224),  # get 224x224 crop from random location
                transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
                transforms.ToTensor(),  # convert the PIL Image to a tensor
                transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                                     (0.229, 0.224, 0.225))])


class PadCollate:
    """ To make sequence same length """

    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def train_val_dataset(dataset, val_split=0.1, random_state=42):
    """ Get train val split """
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, random_state=random_state)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


def get_coco_loader(transform=None,
                    batch_size=32,
                    vocab_threshold=5,
                    start_word="[CLS]",
                    end_word="[SEP]",
                    unk_word="[unused2]",
                    pad_word="[PAD]",
                    num_workers=0,
                    img_folder='/content/train2014/Users/hlibisev/Downloads/train2014',
                    annotations_file="/content/annotations/captions_train2014.json"):

    """ Creating train_loader and val_loader for COCO dataset"""

    dataset = CoCoDataset(transform=transform,
                          vocab_threshold=vocab_threshold,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          pad_word=pad_word,
                          annotations_file=annotations_file,
                          img_folder=img_folder)
    # train val split
    datasets = train_val_dataset(dataset)

    train_loader = data.DataLoader(dataset=datasets['train'],
                                   num_workers=num_workers,
                                   batch_size=batch_size,
                                   collate_fn=PadCollate(pad_idx=0))

    val_loader = data.DataLoader(dataset=datasets['val'],
                                 num_workers=num_workers,
                                 batch_size=batch_size,
                                 collate_fn=PadCollate(pad_idx=0))

    return train_loader, val_loader, dataset
