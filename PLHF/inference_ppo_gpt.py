import torch

from transformers import AutoTokenizer
from trl.gpt2 import GPT2HeadWithValueModel

model_path = 'checkpoints/ppo_sentiment_gpt/model_10_0.87'  # 模型存放地址
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2HeadWithValueModel.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.eos_token = tokenizer.pad_token

gen_len = 16
gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}


def inference(prompt: str):
    """
    根据prompt生成内容。

    Args:
        prompt (str): _description_
    """
    inputs = tokenizer(prompt, return_tensors='pt')
    response = model.generate(inputs['input_ids'].to(device),
                              max_new_tokens=gen_len, **gen_kwargs)
    r = response.squeeze()[-gen_len:]
    return tokenizer.decode(r)


if __name__ == '__main__':
    from rich import print

    gen_times = 10

    prompt = '说实话真的很'
    print(f'prompt: {prompt}')

    for i in range(gen_times):  # 对同一个 prompt 连续生成 10 次答案
        res = inference(prompt)
        print(f'res {i}: ', res)