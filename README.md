# BERT_test

对BERT进行了一些简单的使用，参考了:
[https://zhuanlan.zhihu.com/p/112655246](https://zhuanlan.zhihu.com/p/112655246)和[https://www.cnblogs.com/wwj99/p/12283799.html](https://www.cnblogs.com/wwj99/p/12283799.html)

本次使用的是模型为chinese_roberta_wwm_ext_pytorch，下载链接详见[https://github.com/ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)

#### BERT用于文本分类，运行如下：
``` text
训练阶段：python test.py --train
测试模型：python test.py --test
结果：test acc: 0.9468

####BERTForMaskedLM简单使用
详见bert_for_MLM.py

####BertForNextSentencePrediction简单使用
详见bert_for_NSP.py

