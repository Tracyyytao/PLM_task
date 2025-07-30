import os
import json

import torch
from tqdm import tqdm

from transformers import BertModel
from transformers import BertTokenizer

text_file = 'data/comment_classify/types_desc.txt'  # 候选文本存放地址
output_file = 'embeddings/comment_classify/sentence_transformer_type_embeddings.json'  # embedding存放地址

#device = 'cuda:0'  # 指定GPU设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_type = 'Sentence Transformer'  # 使用DSSM还是Sentence Transformer
saved_model_path = './check/comment_classify/sentence_transformer/model_best/model_best'  # 训练模型存放地址
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
model = BertModel.from_pretrained('../../Huggingface/bert-base-chinese')
model.to(device).eval()


'''
def forward_embedding(type_desc: str) -> torch.tensor:
    """
    将输入喂给encoder并得到对应的embedding向量。

    Args:
        type_desc (_type_): 输入文本

    Returns:
        torch.tensor: (768,)
    """
    encoded_inputs = tokenizer(
        text=type_desc,
        truncation=True,
        max_length=256,
        return_tensors='pt',
        padding='max_length')
    if model_type == 'dssm':
        embedding = model(input_ids=encoded_inputs['input_ids'].to(device),
                          token_type_ids=encoded_inputs['token_type_ids'].to(device),
                          attention_mask=encoded_inputs['attention_mask'].to(device))
    elif model_type == 'sentence_transformer':
        embedding = model.get_embedding(input_ids=encoded_inputs['input_ids'].to(device),
                                        token_type_ids=encoded_inputs['token_type_ids'].to(device),
                                        attention_mask=encoded_inputs['attention_mask'].to(device))
    else:
        raise ValueError('@param model_type must in ["dssm", "sentence_transformer"].')
    return embedding.detach().cpu().numpy()[0].tolist()
'''


def forward_embedding(type_desc: str) -> list:
    encoded_inputs = tokenizer(
        text=type_desc,
        truncation=True,
        max_length=256,
        return_tensors='pt',
        padding='max_length'
    )

    inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
    outputs = model(**inputs)  # BertModel 返回的是一个包含多个字段的对象

    # 取 [CLS] 向量（pooler_output 是对第一个 token 的线性变换 + tanh）
    embedding_tensor = outputs.pooler_output  # shape: (1, 768)

    return embedding_tensor.detach().cpu().numpy()[0].tolist()


def extract_embedding(use_embedding=True):
    """
    获得type_embedding文件中存放的所有文本的embedding并存放到本地。

    Args:
        use_embedding (bool, optional): _description_. Defaults to True.
        model_type (str): 使用哪种模型结构
    """
    type_embedding_dict = {}
    with open(text_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            res = line.strip().split('\t')
            if len(res) == 3:
                _type, label, desc = res
            elif len(res) == 2:
                _type, label = res
                desc = ''
            type_desc = f'{label}：{desc}'

            if use_embedding:
                type_embedding = forward_embedding(type_desc)
            else:
                type_embedding = []

            type_embedding_dict[_type] = {
                'label': label,
                'text': type_desc,
                'embedding': type_embedding
            }
    json.dump(type_embedding_dict, open(output_file, 'w', encoding='utf8'), ensure_ascii=False)


if __name__ == '__main__':
    extract_embedding(use_embedding=True)