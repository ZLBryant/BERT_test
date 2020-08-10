from pytorch_pretrained_bert import BertTokenizer, BertForNextSentencePrediction
import torch
import numpy as np

if __name__ == "__main__":
    vocab_path = "chinese_roberta_wwm_ext_pytorch/vocab.txt"
    tokenizer = BertTokenizer(vocab_file=vocab_path)
    samples = ["[CLS]今天天气怎么样？[SEP]今天天气很好。[SEP]", "[CLS]小明今年几岁了？[SEP]小明爱吃西瓜。[SEP]"]
    samples_tokens = [tokenizer.tokenize(each) for each in samples]
    samples_tokens_ids = [tokenizer.convert_tokens_to_ids(samples_token) for samples_token in samples_tokens]
    input = torch.LongTensor(samples_tokens_ids)
    bert_path = "chinese_roberta_wwm_ext_pytorch/"
    model = BertForNextSentencePrediction.from_pretrained(bert_path)
    model.eval()
    output = model(input)
    nsp_predict = output.max(dim=-1)[1].numpy()
    print(nsp_predict)
    #output:
    #[0,1]，0表示是上下句关系，1表示不是