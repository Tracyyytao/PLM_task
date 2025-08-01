from typing import List

import torch
from rich import print
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference(
        query_list: List[str],
        doc_list: List[str],
        model,
        tokenizer,
        device,
        max_seq_len=64,
        batch_size=8
):
    """
    推理函数。

    Args:
        query_list (List[str]): query列表
        doc_list (List[str]): doc列表
        batch_size (int, optional): _description_. Defaults to 8.
    """
    assert len(query_list) == len(doc_list), 'must have same length.'

    cos_sim_list = []
    for i in range(0, len(query_list), batch_size):
        query_inputs = tokenizer(
            query_list[i:i + batch_size],
            max_length=max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )
        doc_inputs = tokenizer(
            doc_list[i:i + batch_size],
            max_length=max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )
        query_embedding = model.get_pooled_embedding(
            query_inputs['input_ids'].to(device),
            query_inputs['token_type_ids'].to(device)
        )  # (batch, hidden_dim)
        doc_embedding = model.get_pooled_embedding(
            doc_inputs['input_ids'].to(device),
            doc_inputs['token_type_ids'].to(device)
        )  # (batch, hidden_dim)

        cur_cos_sim = torch.nn.functional.cosine_similarity(query_embedding, doc_embedding)
        cos_sim_list.extend(cur_cos_sim.cpu().tolist())

    return cos_sim_list


if __name__ == '__main__':
    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained('./check/model_best')
    model = torch.load('./check/model_best/model.pt')
    model.to(device).eval()

    sentence_pair = [
        ('男孩喝女孩的尿的故事', '怎样才知道是生男孩还是女孩'),
        ('这种图片是用什么软件制作的？', '这种图片制作是用什么软件呢？')
    ]
    query_list = [s[0] for s in sentence_pair]
    doc_list = [s[1] for s in sentence_pair]
    res = inference(
        query_list,
        doc_list,
        model,
        tokenizer,
        device
    )
    print(res)