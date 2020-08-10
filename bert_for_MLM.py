from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import torch
import numpy as np

def pad_sentences(sentences):
    sentence_lens = [len(sentence) for sentence in sentences]
    max_len = max(sentence_lens)
    for each in sentences:
        each += [0] * (max_len - len(each))

if __name__ == '__main__':
    samples = ['[CLS] 中国的[MASK]都是哪里？ [SEP] 北京是 [MASK] 国的首都。 [SEP]',
               '[CLS] 科比81分对阵哪支球队？ [SEP] 对阵猛[MASK]队。 [SEP]']
    vocab_path = "chinese_roberta_wwm_ext_pytorch/vocab.txt"
    tokenizer = BertTokenizer(vocab_file=vocab_path)
    samples_tokens = [tokenizer.tokenize(sample) for sample in samples]
    samples_ids = [tokenizer.convert_tokens_to_ids(samples_token) for samples_token in samples_tokens]
    pad_sentences(samples_ids)
    bert_path = "chinese_roberta_wwm_ext_pytorch/"
    model = BertForMaskedLM.from_pretrained(bert_path)
    model.eval()
    input_ids = torch.LongTensor(samples_ids)
    outputs = model(input_ids)#看一下outputs的结构
    predict_token_ids = outputs.max(dim=-1)[1].numpy()
    complete_sentences = [tokenizer.convert_ids_to_tokens(predict_token_id) for predict_token_id in predict_token_ids]
    print("complete sentence")
    for each in complete_sentences:
        print(''.join(each))
    # 输出：
    # complete sentence
    # 都中国的首都是哪里？都北京是中国的首都。都
    # 。科比81分对阵哪支球队？。对阵猛龙队。。。
    # 第一句中mask掉的内容在第一句都有出现过，第二句中则没有，不过效果不错