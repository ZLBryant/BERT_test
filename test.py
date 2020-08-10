from data_process import get_dataloader
import argparse
import torch
import os
import numpy as np
import random
from model import BertForClassify, model_train, model_test

def args_init():
    parser = argparse.ArgumentParser(description="Bert For Text Classification")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--label_num", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_per_iter", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=int, default=0.01)
    parser.add_argument("--train_data_path", type=str, default='data/train.txt')
    parser.add_argument("--dev_data_path", type=str, default='data/dev.txt')
    parser.add_argument("--test_data_path", type=str, default='data/test.txt')
    parser.add_argument("--vocab_path", type=str, default="chinese_roberta_wwm_ext_pytorch/vocab.txt")
    parser.add_argument("--bert_path", type=str, default="chinese_roberta_wwm_ext_pytorch/")
    parser.add_argument("--model_save_path", type=str, default="output/")
    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = args_init()
    args.device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    args.model_save_path += 'model.pkl'
    if args.train:
        setup_seed(args.seed)
        train_dataloader, dev_dataloader, _ = get_dataloader(args)
        model = BertForClassify(args).to(args.device)
        model_train(model, train_dataloader, dev_dataloader, args)
    elif args.test:
        _, _, test_dataloader = get_dataloader(args)
        model = torch.load(args.model_save_path)
        acc = model_test(model, test_dataloader, args)
        print("test acc:", acc)
    else:
        print("illegal input!")