#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/26 17:33
# @Author  : yingxiao zhang
# @File    : tokenizer_test.py
# from transformers import T5Tokenizer
#
# #<pad> </s> <unk> ▁ \t [ ] , " "
# tokenizer = T5Tokenizer("m.model")
#
# input_text = '''[
#             "3163693955\t社会民生\t0",
#             "1945570091\t内地明星\t0",
#             "2151481347\t内地明星\t0",
#             "1749964961\t内地明星,妆容点评\t0",
#             "2609400635\t内地明星,港台明星,甜品烘焙\t0"
#         ]'''
#
# input_ids = tokenizer.encode(input_text, return_tensors="pt")
# print(input_ids)
# # 将input_ids张量转换为列表
# input_ids_list = input_ids.squeeze().tolist()  # 使用squeeze()移除多余的维度
# # 使用convert_ids_to_tokens方法将id转换为tokens
# tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
# token_for_id_1 = tokenizer.convert_ids_to_tokens(1)
# print(f"The token for ID 1 is: {token_for_id_1}")
# # 打印每个id及其对应的token
# for id, token in zip(input_ids_list, tokens):
#     print(f"{id} -> {token}")


from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer_path = "/workspace/yingxiao1/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

input_text = "nice to meet you"

input_ids = tokenizer.encode(input_text, return_tensors="pt")
print(input_ids)
# 将input_ids张量转换为列表
input_ids_list = input_ids.squeeze().tolist()  # 使用squeeze()移除多余的维度
# 使用convert_ids_to_tokens方法将id转换为tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
# 打印每个id及其对应的token
for id, token in zip(input_ids_list, tokens):
    print(f"{id} -> {token}")
# for i in range(10):
#     token_for_id = tokenizer.convert_ids_to_tokens(i)
#     print(f"The token for ID {i} is: {token_for_id}")
