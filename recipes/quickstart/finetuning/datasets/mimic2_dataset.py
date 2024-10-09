# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import copy
import itertools
import re

import torch
from datasets import load_dataset
from PIL import Image

print("@@@ using mimic2_dataset.py")

def remove_image_tags(text):
    # Use regex to match one or more <image> tags at the beginning of the string
    return re.sub(r"^(<image>)+", "", text)


# check system prompt token seq or user prompt token seq is in the current token list
def check_header(targets, seq):
    for i in range(len(seq) - 3):
        if seq[i : i + 3] in targets:
            return True
    return False


def replace_target(target, seq):
    for i in range(len(seq) - 3):
        if seq[i : i + 3] == target:
            seq[i], seq[i + 1], seq[i + 2] = -100, -100, -100
    return seq


def tokenize_dialogs(dialogs, images, processor):
    text_prompt = processor.apply_chat_template(dialogs)
    batch = processor(
        images=images, text=text_prompt, padding=True, return_tensors="pt"
    )
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i, n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx : idx + 1]
            if check_header(prompt_header_seqs, current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx : idx + 1] = [-100] * (idx - last_idx + 1)
            else:
                last_idx = idx + 1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)
        # Mask the padding token and image token 128256
        for i in range(len(labels)):
            if (
                labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256
            ):  #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch


def get_custom_dataset(dataset_config, processor, split, split_ratio=0.9):
    train_jsonl_path = "/mnt/data/ruian/mimic2/gpt/train_dataset_gpt_labels.jsonl"
    test_jsonl_path = "/mnt/data/ruian/mimic2/gpt/test_dataset_gpt_labels.jsonl"

    data_files = {"train": train_jsonl_path, "test": test_jsonl_path}
    dataset_dict = load_dataset("json", data_files=data_files)

    if split == "train":
        dataset = dataset_dict["train"]
    elif split == "test":
        dataset = dataset_dict["test"]

    return dataset


class GPTDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = (
            "right"  # during training, one always uses padding on the right
        )

    def __call__(self, samples):
        dialogs, images_list = [], []
        for sample in samples:
            image_list, sample_list = sample["image"], sample["conversations"]

            images = [
                Image.open(each_image).convert("RGB") for each_image in image_list
            ]

            user_content = [{"type": "image"}] * len(image_list)
            user_text = remove_image_tags(sample_list[0]["value"])
            user_content.append(user_text)

            user_msg = {"role": "user", "content": user_content}

            assistant_msg = {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": sample_list[1]["value"],
                    }
                ],
            }

            dialog = [user_msg, assistant_msg]

            dialogs.append(dialog)
            images_list.append(images)
        return tokenize_dialogs(dialogs, images_list, self.processor)


def get_data_collator(processor):
    return GPTDataCollator(processor)
