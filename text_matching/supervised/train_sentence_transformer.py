import os
import time
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, default_data_collator, get_scheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import SentenceTransformer
from utils import convert_dssm_example
from iTrainingLogger import iSummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='../../Huggingface/bert-base-chinese', type=str, help="backbone of encoder.")
parser.add_argument("--train_path", default='./data/comment_classify/train.txt', type=str, help="The path of train set.")
parser.add_argument("--dev_path", default='./data/comment_classify/dev.txt', type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./check/comment_classify/sentence_transformer/model_best", type=str, help="Model saving directory.")
parser.add_argument("--max_seq_len", default=512, type=int, help="Maximum input sequence length.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total training epochs.")
parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Warmup ratio.")
parser.add_argument("--valid_steps", default=200, type=int, help="Evaluation frequency.")
parser.add_argument("--logging_steps", default=10, type=int, help="Logging frequency.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="TensorBoard log dir.")
parser.add_argument("--img_log_name", default='Model Performance2', type=str, help="TensorBoard log name.")
parser.add_argument('--device', default="cuda:0", help="Device to use.")
args = parser.parse_args()

writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def evaluate_model(model, data_loader, global_step):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            logits = model.get_similarity_label(
                query_input_ids=batch['query_input_ids'].to(args.device),
                query_token_type_ids=batch['query_token_type_ids'].to(args.device),
                query_attention_mask=batch['query_attention_mask'].to(args.device),
                doc_input_ids=batch['doc_input_ids'].to(args.device),
                doc_token_type_ids=batch['doc_token_type_ids'].to(args.device),
                doc_attention_mask=batch['doc_attention_mask'].to(args.device)
            )
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    model.train()
    return acc, precision, recall, f1


def train():
    encoder = BertModel.from_pretrained(args.model)
    model = SentenceTransformer(encoder)
    tokenizer = BertTokenizer.from_pretrained(args.model)

    dataset = load_dataset('text', data_files={'train': args.train_path, 'dev': args.dev_path})
    convert_func = partial(convert_dssm_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dataset = dataset.map(convert_func, batched=True)

    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator,
                                  batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator,
                                 batch_size=args.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    model.to(args.device)

    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warm_steps,
                                 num_training_steps=max_train_steps)

    loss_list = []
    criterion = torch.nn.CrossEntropyLoss()
    tic_train = time.time()
    global_step, best_f1 = 0, 0

    for epoch in range(1, args.num_train_epochs + 1):
        for batch in train_dataloader:
            logits = model.get_similarity_label(
                query_input_ids=batch['query_input_ids'].to(args.device),
                query_token_type_ids=batch['query_token_type_ids'].to(args.device),
                query_attention_mask=batch['query_attention_mask'].to(args.device),
                doc_input_ids=batch['doc_input_ids'].to(args.device),
                doc_token_type_ids=batch['doc_token_type_ids'].to(args.device),
                doc_attention_mask=batch['doc_attention_mask'].to(args.device)
            )
            labels = batch['labels'].to(args.device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())
            global_step += 1

            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                avg_loss = sum(loss_list) / len(loss_list)
                writer.add_scalar('train/train_loss', avg_loss, global_step)
                print(f"Step {global_step}, Epoch {epoch}, Loss: {avg_loss:.4f}, Speed: {args.logging_steps / time_diff:.2f} steps/s")
                tic_train = time.time()
                loss_list.clear()

            if global_step % args.valid_steps == 0:
                save_dir = os.path.join(args.save_dir, f"model_{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model, os.path.join(save_dir, 'model.pt'))
                tokenizer.save_pretrained(save_dir)

                acc, precision, recall, f1 = evaluate_model(model, eval_dataloader, global_step)
                writer.add_scalar('eval/accuracy', acc, global_step)
                writer.add_scalar('eval/precision', precision, global_step)
                writer.add_scalar('eval/recall', recall, global_step)
                writer.add_scalar('eval/f1', f1, global_step)
                writer.record()

                print(f"[Eval] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                if f1 > best_f1:
                    print(f"Best F1 updated: {best_f1:.4f} --> {f1:.4f}")
                    best_f1 = f1
                    best_save_dir = os.path.join(args.save_dir, "model_best")
                    os.makedirs(best_save_dir, exist_ok=True)
                    torch.save(model, os.path.join(best_save_dir, 'model.pt'))
                    tokenizer.save_pretrained(best_save_dir)
                tic_train = time.time()


if __name__ == '__main__':
    train()

