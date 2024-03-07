#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/27 14:28
# @Author  : yingxiao zhang
# @File    : tokenizer_custom.py

#cls:0号位，sep:1号位，unk：2号位
def encode(string, vocab_dict, maxlen):
    # 最多取maxlen-1个单元，这是因为我们后面还要append一个结束符
    tokens = string.split("#")[-(maxlen - 1):]

    # 列表推导处理主循环，直接查字典获取索引，若不存在则返回2（未知词）
    ids = [vocab_dict.get(token, 2) for token in tokens]
    # ids = [0] + ids
    # 在列表最后添加结束符1
    ids.append(1)

    return ids

def decode(ids, vocab, maxlen):
    string = []
    ids = ids[:maxlen]
    for id in ids:
        if id == 0:
            continue
        if id == 1:
            break
        else:
            string.append(vocab[id])
    return "#".join(string)