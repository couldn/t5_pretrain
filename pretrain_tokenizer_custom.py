#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/27 15:25
# @Author  : yingxiao zhang
# @File    : pretrain_tokenizer_custom.py

import json
from transformers import T5ForConditionalGeneration, T5Config
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from tokenizer_custom import encode

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

class CustomJSONDataset(Dataset):
    def __init__(self, vocab_dict, file_path):
        super(CustomJSONDataset, self).__init__()

        self.inputs = []
        self.labels = []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print("data nums:", len(data))
            for item in data:
                # input_text = f"instruction: {item['instruction']} input: {item['input']}"
                input_text = item['input']
                target_text = item['output']

                # Here, we separate the encoding for input and target texts.
                # For the input text, we add the EOS token at the end, which is typical for T5 training.
                input_encodings = encode(input_text, vocab_dict, 512)
                target_encodings = encode(target_text, vocab_dict, 512)

                self.inputs.append(input_encodings)
                self.labels.append(target_encodings)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = {"input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)}
        return item

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_t5_with_custom_tokenizer(data_file, model_name="/workspace/yingxiao1/flan-t5-base", custom_vocab_file="./data/custom_vocab_tokenizer_custom_10.txt",
                                   output_dir="out/base-0304", logging_dir="logs/base-0304"):
    vocab = ['cls', 'sep', 'unk']
    with open(custom_vocab_file, 'r', encoding='utf-8') as f:
        custom_tokens = [line.strip() for line in f.readlines()]
    vocab.extend(custom_tokens)
    # while(len(vocab)<30000000):
    #     vocab.append("test")
    # 使用字典来存储vocab以及它们的索引，实现O(1)复杂度的查找
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}

    # model = T5ForConditionalGeneration.from_pretrained(model_name)
    # 设定模型配置（例如T5 Small）
    config = T5Config.from_pretrained(model_name, d_model=72)
    # 使用随机初始化参数创建模型实例
    model = T5ForConditionalGeneration(config)

    model.resize_token_embeddings(len(vocab))

    print(f"The model has {count_parameters(model):,} trainable parameters")

    dataset = CustomJSONDataset(vocab_dict, file_path=data_file)

    data_collator = CustomDataCollator()

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,  # TensorBoard 日志目录
        logging_steps=10,  # 每10个训练步骤记录一次日志
        save_steps=40000,  # 每50个训练步骤保存一次模型
        per_device_train_batch_size=16,
        num_train_epochs=4,
        # learning_rate=5e-5,
        learning_rate=0.01,
        weight_decay=0.01, # 通常0.01或0.1
        # adam_epsilon=1e-8,
        warmup_steps=4000,
        # lr_scheduler_type='cosine',
        gradient_accumulation_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

# 用例操作
train_t5_with_custom_tokenizer("./data/t5_train_data_tokenizer_custom_10.json")