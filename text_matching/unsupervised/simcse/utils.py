import random
import traceback

import torch
import numpy as np


def convert_example(
        examples: dict,
        tokenizer,
        max_seq_len: int,
        mode='train'
):
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 数据样本（不同mode下的数据集不一样）, e.g. -> {
                                                "text": '蛋黄吃多了有什么坏处',                               # train mode
                                                        or '蛋黄吃多了有什么坏处	吃鸡蛋白过多有什么坏处	0',  # evaluate mode
                                                        or '蛋黄吃多了有什么坏处	吃鸡蛋白过多有什么坏处',     # inference mode
                                            }
        mode (bool): 数据集格式 -> 'train': （无监督）训练集模式，一行只有一句话；
                                'evaluate': 验证集训练集模式，两句话 + 标签
                                'inference': 推理集模式，两句话。

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'query_input_ids': [[101, 3928, ...], [101, 4395, ...]],
                            'query_token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'doc_input_ids': [[101, 2648, ...], [101, 3342, ...]],
                            'doc_token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'labels': [1, 0, ...]
                        }
    """
    tokenized_output = {
        'query_input_ids': [],
        'query_token_type_ids': [],
        'doc_input_ids': [],
        'doc_token_type_ids': []
    }

    for example in examples['text']:
        try:
            if mode == 'train':
                query = doc = example.strip()
            elif mode == 'evaluate':
                query, doc, label = example.strip().split('\t')
            elif mode == 'inference':
                query, doc = example.strip().split('\t')
            else:
                raise ValueError(f'No mode called {mode}, expected in ["train", "evaluate", "inference"].')

            query_encoded_inputs = tokenizer(
                text=query,
                truncation=True,
                max_length=max_seq_len,
                padding='max_length')
            doc_encoded_inputs = tokenizer(
                text=doc,
                truncation=True,
                max_length=max_seq_len,
                padding='max_length')
        except:
            print(f'{examples["text"]} -> {traceback.format_exc()}')
            exit()

        tokenized_output['query_input_ids'].append(query_encoded_inputs["input_ids"])
        tokenized_output['query_token_type_ids'].append(query_encoded_inputs["token_type_ids"])
        tokenized_output['doc_input_ids'].append(doc_encoded_inputs["input_ids"])
        tokenized_output['doc_token_type_ids'].append(doc_encoded_inputs["token_type_ids"])
        if mode == 'evaluate':
            if 'labels' not in tokenized_output:
                tokenized_output['labels'] = []
            tokenized_output['labels'].append(int(label))

    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output


def word_repetition(
        input_ids,
        token_type_ids,
        dup_rate=0.32,
        min_dup_sentence_len_threshold=5,
        device='cpu',
        max_len=512
) -> torch.tensor:
    """
    随机重复单词策略，用于在正例样本中添加噪声，并强制截断到 max_len。
    """
    input_ids = input_ids.numpy().tolist()
    token_type_ids = token_type_ids.numpy().tolist()

    batch_size = len(input_ids)
    repetitied_input_ids = []
    repetitied_token_type_ids = []

    for batch_id in range(batch_size):
        cur_input_id = input_ids[batch_id]
        actual_len = np.count_nonzero(cur_input_id)
        dup_word_index = []

        if actual_len > min_dup_sentence_len_threshold:
            dup_len = random.randint(a=0, b=max(2, int(dup_rate * actual_len)))
            dup_word_index = random.sample(list(range(1, actual_len - 1)), k=dup_len)

        r_input_id = []
        r_token_type_id = []
        for idx, word_id in enumerate(cur_input_id):
            if idx in dup_word_index:
                r_input_id.append(word_id)
                r_token_type_id.append(token_type_ids[batch_id][idx])
            r_input_id.append(word_id)
            r_token_type_id.append(token_type_ids[batch_id][idx])

        # ✅ 强制截断到 max_len
        r_input_id = r_input_id[:max_len]
        r_token_type_id = r_token_type_id[:max_len]

        repetitied_input_ids.append(r_input_id)
        repetitied_token_type_ids.append(r_token_type_id)

    # ✅ padding 到 batch 内最大长度（BERT 支持自动 attention_mask 推断）
    max_batch_len = max(len(seq) for seq in repetitied_input_ids)
    for i in range(batch_size):
        pad_len = max_batch_len - len(repetitied_input_ids[i])
        repetitied_input_ids[i] += [0] * pad_len
        repetitied_token_type_ids[i] += [0] * pad_len

    return (
        torch.tensor(repetitied_input_ids).to(device),
        torch.tensor(repetitied_token_type_ids).to(device),
    )
