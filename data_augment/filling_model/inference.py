from rich import print
from transformers import AutoTokenizer, T5ForConditionalGeneration

device = 'cuda:0'
max_source_seq_len = 128
tokenizer = AutoTokenizer.from_pretrained('./check/model_best/')
model = T5ForConditionalGeneration.from_pretrained('./check/model_best/')
model.to(device).eval()


def inference(masked_texts: list):
    """
    inference函数。

    Args:
        masked_texts (list): 掩码后的文字列表
    """
    inputs = tokenizer(
        text=masked_texts,
        truncation=True,
        max_length=max_source_seq_len,
        padding='max_length',
        return_tensors='pt'
    )
    outputs = model.generate(input_ids=inputs["input_ids"].to(device))
    outputs = [tokenizer.decode(output.cpu().numpy(), skip_special_tokens=True).replace(" ", "") \
                    for output in outputs]
    print(f'maksed text: {masked_texts}')
    print(f'output: {outputs}')


if __name__ == '__main__':
    masked_texts = [
        '"《μVision2单片机应用程序开发指南》是2005年2月[MASK]图书，作者是李宇"中[MASK]位置的文本是：'
    ]
    inference(masked_texts)