#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/27 15:27
# @Author  : yingxiao zhang
# @File    : predict_tokenizer_custom.py


from transformers import T5ForConditionalGeneration
from tokenizer_custom import encode, decode
import torch


def load_model(model_path, tokenizer_path):
    """
    加载模型和tokenizer。
    """
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return model
def predict(model, vocab, vocab_dict, input_text, max_length=512):
    """
    使用模型进行预测。
    """
    # 对输入文本进行编码
    input_ids = encode(input_text, vocab_dict, max_length)
    # print(input_ids)
    input_ids = torch.tensor([input_ids])

    # 生成输出
    output_ids = model.generate(input_ids, max_length=max_length)
    output_ids = output_ids[0].squeeze().tolist()
    print(output_ids)
    print(len(output_ids))

    # 将输出ids转换为文字
    # output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_text = decode(output_ids, vocab, max_length)
    return output_text


# 加载训练好的模型
model = load_model("./out/base-0304/checkpoint-80000", "/workspace/yingxiao1/flan-t5-base")  # 请替换"out/checkpoint-xxx"为您的实际模型路径

vocab = ['cls', 'sep', 'unk']
custom_vocab_file="./data/custom_vocab_tokenizer_custom_10.txt"
with open(custom_vocab_file, 'r', encoding='utf-8') as f:
    custom_tokens = [line.strip() for line in f.readlines()]
vocab.extend(custom_tokens)
# 使用字典来存储vocab以及它们的索引，实现O(1)复杂度的查找
vocab_dict = {word: idx for idx, word in enumerate(vocab)}

# 准备一些输入
input_text = " "
# 进行预测
output_text = predict(model, vocab, vocab_dict, input_text)
print("预测输出：", output_text)
