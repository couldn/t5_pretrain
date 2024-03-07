#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/21 19:28
# @Author  : yingxiao zhang
# @File    : predict.py

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


def load_model(model_path, tokenizer_path):
    """
    加载模型和tokenizer。
    """
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    print("vocab_size:", tokenizer.vocab_size)
    print("vocab_size:", len(tokenizer.get_vocab()))
    custom_vocab_file = "./data/custom_vocab_large.txt"
    with open(custom_vocab_file, 'r', encoding='utf-8') as f:
        custom_tokens = [line.strip() for line in f.readlines()]
    num_added_toks = tokenizer.add_tokens(custom_tokens)
    print(f'Added {num_added_toks} tokens')
    print("vocab_size:", len(tokenizer.get_vocab()))
    return model, tokenizer


def predict(model, tokenizer, input_text, max_length=512):
    """
    使用模型进行预测。
    """
    # 对输入文本进行编码
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    print(input_ids)
    # 将input_ids张量转换为列表
    input_ids_list = input_ids.squeeze().tolist()  # 使用squeeze()移除多余的维度
    # 使用convert_ids_to_tokens方法将id转换为tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
    # 打印每个id及其对应的token
    for id, token in zip(input_ids_list, tokens):
        print(f"{id} -> {token}")

    # 生成输出
    output_ids = model.generate(input_ids, max_length=max_length)

    # 将输出ids转换为文字
    # output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_text = tokenizer.decode(output_ids[0])
    print(output_ids[0])
    # input_ids = tokenizer.encode(output_text, return_tensors="pt")
    # input_ids_list = input_ids.squeeze().tolist()
    # tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
    # # 打印每个id及其对应的token
    # for id, token in zip(input_ids_list, tokens):
    #     print(f"{id} -> {token}")
    return output_text


# 加载训练好的模型
model, tokenizer = load_model("/workspace/yingxiao1/flan-t5-base", "/workspace/yingxiao1/flan-t5-base")  # 请替换"out/checkpoint-xxx"为您的实际模型路径

# 准备一些输入
input_text = ""

# 进行预测
output_text = predict(model, tokenizer, input_text)

print("预测输出：", output_text)