from rich import print

import torch
from transformers import AutoTokenizer

device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained('./check/model_best/')
model = torch.load('./check/model_best/model.pt')
model.to(device).eval()


def test_inference(text1, text2, max_seq_len=128) -> torch.tensor:
    """
    预测函数，输入两句文本，返回这两个文本相似/不相似的概率。

    Args:
        text1 (str): 第一段文本
        text2 (_type_): 第二段文本
        max_seq_len (int, optional): 文本最大长度. Defaults to 128.

    Reuturns:
        torch.tensor: 相似/不相似的概率 -> (batch, 2)
    """
    encoded_inputs = tokenizer(
        text=[text1],
        text_pair=[text2],
        truncation=True,
        max_length=max_seq_len,
        return_tensors='pt',
        padding='max_length')

    with torch.no_grad():
        model.eval()
        logits = model(input_ids=encoded_inputs['input_ids'].to(device),
                       token_type_ids=encoded_inputs['token_type_ids'].to(device),
                       attention_mask=encoded_inputs['attention_mask'].to(device))
        print(logits)


if __name__ == '__main__':
    test_inference(
        '手机：一种可以在较广范围内使用的便携式电话终端。',
        '味道非常好，京东送货速度也非常快，特别满意。',
        max_seq_len=128
    )