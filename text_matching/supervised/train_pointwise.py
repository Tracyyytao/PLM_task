import os
import time
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, default_data_collator, get_scheduler

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model import PointwiseMatching
from utils import convert_pointwise_example
from iTrainingLogger import iSummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='../../Huggingface/bert-base-chinese', type=str)
parser.add_argument("--train_path", default='./data/comment_classify/train.txt', type=str)
parser.add_argument("--dev_path", default='./data/comment_classify/dev.txt', type=str)
parser.add_argument("--save_dir", default="./check", type=str)
parser.add_argument("--max_seq_len", default=512, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--learning_rate", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--num_train_epochs", default=10, type=int)
parser.add_argument("--warmup_ratio", default=0.06, type=float)
parser.add_argument("--valid_steps", default=200, type=int)
parser.add_argument("--logging_steps", default=10, type=int)
parser.add_argument("--img_log_dir", default='logs', type=str)
parser.add_argument("--img_log_name", default='Model Performance', type=str)
parser.add_argument('--device', default="cuda:0", help="cuda:0 or cpu")
args = parser.parse_args()

writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)

def evaluate_model(model, data_loader, global_step):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            logits = model(input_ids=batch['input_ids'].to(args.device),
                           token_type_ids=batch['token_type_ids'].to(args.device),
                           position_ids=batch['position_ids'].to(args.device),
                           attention_mask=batch['attention_mask'].to(args.device))
            preds = logits.argmax(dim=-1).cpu().tolist()
            labels = batch['labels'].cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    model.train()
    return acc, precision, recall, f1

def train():
    encoder = BertModel.from_pretrained(args.model)
    model = PointwiseMatching(encoder)
    tokenizer = BertTokenizer.from_pretrained(args.model)
    dataset = load_dataset('text', data_files={'train': args.train_path, 'dev': args.dev_path})
    convert_func = partial(convert_pointwise_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dataset = dataset.map(convert_func, batched=True)

    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator,
                                  batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    model.to(args.device)

    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    loss_list = []
    criterion = torch.nn.CrossEntropyLoss()
    tic_train = time.time()
    global_step, best_f1 = 0, 0
    for epoch in range(1, args.num_train_epochs + 1):
        for batch in train_dataloader:
            logits = model(input_ids=batch['input_ids'].to(args.device),
                           token_type_ids=batch['token_type_ids'].to(args.device),
                           position_ids=batch['position_ids'].to(args.device),
                           attention_mask=batch['attention_mask'].to(args.device))
            labels = batch['labels'].to(args.device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))

            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                writer.add_scalar('train/train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                      % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                os.makedirs(cur_save_dir, exist_ok=True)
                torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
                tokenizer.save_pretrained(cur_save_dir)

                acc, precision, recall, f1 = evaluate_model(model, eval_dataloader, global_step)
                writer.add_scalar('eval/accuracy', acc, global_step)
                writer.add_scalar('eval/precision', precision, global_step)
                writer.add_scalar('eval/recall', recall, global_step)
                writer.add_scalar('eval/f1', f1, global_step)
                writer.record()

                print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))
                if f1 > best_f1:
                    print(f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}")
                    best_f1 = f1
                    best_save_dir = os.path.join(args.save_dir, "model_best")
                    os.makedirs(best_save_dir, exist_ok=True)
                    torch.save(model, os.path.join(best_save_dir, 'model.pt'))
                    tokenizer.save_pretrained(best_save_dir)
                tic_train = time.time()

if __name__ == '__main__':
    train()
