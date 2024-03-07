#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 14:33
# @Author  : yingxiao zhang
# @File    : pretrain_pipeline.py

import json
from transformers import T5ForConditionalGeneration, T5Config
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from tokenizer_custom import encode
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import glob

class CustomDataCollator:
    def __call__(self, examples):
        batch_size = len(examples)
        max_input_length = max(len(x["input_ids"]) for x in examples)
        max_label_length = max(len(x["labels"]) for x in examples)

        input_ids_padded = torch.full((batch_size, max_input_length), fill_value=0, dtype=torch.long)
        labels_padded = torch.full((batch_size, max_label_length), fill_value=-100, dtype=torch.long)

        for i, example in enumerate(examples):
            input_ids = example["input_ids"]
            labels = example["labels"]
            input_ids_padded[i, :len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
            labels_padded[i, :len(labels)] = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": input_ids_padded != 0
        }


class StreamJSONDataset(IterableDataset):
    def __init__(self, vocab_dict, data_pattern):
        super(StreamJSONDataset, self).__init__()
        self.vocab_dict = vocab_dict
        self.data_pattern = data_pattern

    def parse_file(self, data_pattern):
        #num:8280600
        file_list = glob.glob(data_pattern)
        for file_path in file_list:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    input_text = item['input']
                    target_text = item['output']
                    input_encodings = encode(input_text, self.vocab_dict, 512)
                    target_encodings = encode(target_text, self.vocab_dict, 512)
                    yield {
                        "input_ids": torch.tensor(input_encodings, dtype=torch.long),
                        "labels": torch.tensor(target_encodings, dtype=torch.long),
                    }

    def __len__(self):
        #你数据的总条数
        return 8000000

    def __iter__(self):
        return iter(self.parse_file(self.data_pattern))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_t5_with_custom_tokenizer(data_pattern, model_name="/workspace/yingxiao1/flan-t5-small", output_dir="out/small-0307", logging_dir="logs/small-0307"):
    vocab = ['cls', 'sep', 'unk', 'mask', "0", "1"]
    # your vocab, one word one line
    vocab_file_list = glob.glob("./data/t5_train_data_words/part-*")
    for file in vocab_file_list:
        with open(file, 'r', encoding='utf-8') as f:
            custom_tokens = [line.strip() for line in f.readlines()]
            vocab.extend(custom_tokens)
    # while(len(vocab)<30000000):
    #     vocab.append("test")
    # 使用字典来存储vocab以及它们的索引，实现O(1)复杂度的查找
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}

    # model = T5ForConditionalGeneration.from_pretrained(model_name)
    # 设定模型配置（例如T5 Small）
    # adjust embedding_dim
    config = T5Config.from_pretrained(model_name, d_model=72)
    # 使用随机初始化参数创建模型实例
    model = T5ForConditionalGeneration(config)
    # resize model vocab size
    model.resize_token_embeddings(len(vocab))
    print("vocab_size:", len(vocab))
    print(f"The model has {count_parameters(model):,} trainable parameters")

    # 创建数据集实例
    dataset = StreamJSONDataset(vocab_dict, data_pattern)

    data_collator = CustomDataCollator()

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,  # TensorBoard 日志目录
        logging_steps=10,  # 每10个训练步骤记录一次日志
        save_steps=40000,  # 每50个训练步骤保存一次模型
        per_device_train_batch_size=2,
        num_train_epochs=2,
        # learning_rate=5e-5,
        learning_rate=0.01,
        weight_decay=0.01, # 通常0.01或0.1
        # adam_epsilon=1e-8,
        warmup_steps=4000,
        # lr_scheduler_type='cosine',
        gradient_accumulation_steps=4,
        max_steps=8000000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

# 用例操作
# data_pattern is your dataset path
train_t5_with_custom_tokenizer(data_pattern="./data/*")