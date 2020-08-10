from pytorch_pretrained_bert import BertModel, BertAdam
from torch import nn
import torch

class BertForClassify(nn.Module):
    def __init__(self, args):
        super(BertForClassify, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_path)
        for p in self.bert.parameters():
            p.requires_grad = True
        self.predict = nn.Linear(768, args.label_num)

    def forward(self, sentences, masks, types):
        _, output = self.bert(sentences, attention_mask=masks, token_type_ids=types, output_all_encoded_layers=False)
        return self.predict(output)

def model_train(model, train_dataloader, dev_dataloader, args):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=0.05, t_total=len(train_dataloader) * args.epoch)
    criterion = nn.CrossEntropyLoss(size_average=True)
    best_acc = 0
    cur_loss = 0
    for epoch in range(args.epoch):
        model.train()
        train_acc = 0
        predict_num = 0
        for batch_idx, batch_data in enumerate(train_dataloader):
            sentences = batch_data[0].to(args.device)
            masks = batch_data[1].to(args.device)
            types = batch_data[2].to(args.device)
            label = batch_data[3].to(args.device)
            predict = model(sentences, masks, types)
            model.zero_grad()
            loss = criterion(predict, label.squeeze())
            cur_loss += loss.detach().cpu().numpy()
            loss.backward()
            optimizer.step()
            predict = predict.max(-1)[1]
            train_acc += (predict == label.squeeze()).sum()
            predict_num += label.shape[0]
            if batch_idx % args.eval_per_iter == 0:
                cur_loss /= args.eval_per_iter
                train_acc = float(train_acc) / predict_num
                print("train loss:", cur_loss, "train acc:", train_acc)
                cur_loss = 0
                predict_num = 0
                train_acc = 0
                acc = model_test(model, dev_dataloader, args)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model, args.model_save_path)
                print("eval acc:", acc)
                model.train()

def model_test(model, test_dataloader, args):
    model.eval()
    with torch.no_grad():
        acc = 0
        predict_num = 0
        for batch_idx, batch_data in enumerate(test_dataloader):
            sentences = batch_data[0].to(args.device)
            masks = batch_data[1].to(args.device)
            types = batch_data[2].to(args.device)
            label = batch_data[3].to(args.device)
            predict = model(sentences, masks, types)
            predict = predict.max(-1)[1]
            acc += (predict == label.squeeze()).sum()
            predict_num += label.shape[0]
        acc = float(acc) / predict_num
    return acc