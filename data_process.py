from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random

class TextDataset(Dataset):
    def __init__(self, labels, sentences, masks, types):
        super(TextDataset, self).__init__()
        self.labels = labels
        self.sentences = sentences
        self.masks = masks
        self.types = types

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        sentences = torch.LongTensor(self.sentences[item])
        masks = torch.LongTensor(self.masks[item])
        types = torch.LongTensor(self.types[item])
        label = torch.LongTensor([self.labels[item]])
        return sentences, masks, types, label

def get_dataloader(args):
    train_labels, train_sentences, train_masks, train_types = read_corpus(args.train_data_path, args.vocab_path, args.max_len)
    train_dataset = TextDataset(train_labels, train_sentences, train_masks, train_types)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)#, collate_fn=collate_fn)

    dev_labels, dev_sentences, dev_masks, dev_types = read_corpus(args.dev_data_path, args.vocab_path, args.max_len)
    dev_dataset = TextDataset(dev_labels, dev_sentences, dev_masks, dev_types)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)  # , collate_fn=collate_fn)

    test_labels, test_sentences, test_masks, test_types = read_corpus(args.test_data_path, args.vocab_path, args.max_len)
    test_dataset = TextDataset(test_labels, test_sentences, test_masks, test_types)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)#, collate_fn=collate_fn)

    return train_dataloader, dev_dataloader, test_dataloader

def collate_fn(data):
    return data[0], data[1], data[2], data[3]

def read_corpus(data_path, vocab_path, max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_path)
    sentences = []
    masks = []
    types = []
    labels = []
    with open(data_path, encoding='utf-8') as f:
        for l in f:
            sentence, label = l.strip().split('\t')
            labels.append(int(label))
            sentence = ["[CLS]"] + tokenizer.tokenize(sentence) + ["[SEP]"]
            sentence = tokenizer.convert_tokens_to_ids(sentence)
            type = [0] * len(sentence)
            mask = [1] * len(sentence)
            if len(sentence) < max_len:
                type += [1] * (max_len - len(sentence))
                mask += [0] * (max_len - len(sentence))
                sentence += [0] * (max_len - len(sentence))
            else:
                type = type[:max_len]
                mask = mask[:max_len]
                sentence = sentence[:max_len]
            sentences.append(sentence)
            types.append(type)
            masks.append(mask)
    return labels, sentences, masks, types
