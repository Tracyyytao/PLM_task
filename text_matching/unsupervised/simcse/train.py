import os
import time
import argparse
from functools import partial

import torch
from scipy import stats
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, default_data_collator, get_scheduler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import SimCSE
from utils import convert_example, word_repetition
from iTrainingLogger import iSummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='bert-base-chinese', type=str)
parser.add_argument("--train_path", default='data/LCQMC/train.txt', type=str)
parser.add_argument("--dev_path", default='data/LCQMC/dev.tsv', type=str)
parser.add_argument("--save_dir", default="./check", type=str)
parser.add_argument("--max_seq_len", default=512, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--num_train_epochs", default=10, type=int)
parser.add_argument("--warmup_ratio", default=0.0, type=float)
parser.add_argument("--duplicate_ratio", default=0.32, type=float)
parser.add_argument("--valid_steps", default=200, type=int)
parser.add_argument("--logging_steps", default=10, type=int)
parser.add_argument("--img_log_dir", default='logs', type=str)
parser.add_argument("--img_log_name", default='Model Performance', type=str)
parser.add_argument('--device', default="cuda:0", help="Device for training.")
args = parser.parse_args()

writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def evaluate_model(model, data_loader, cosine_similarity_threshold=0.5):
    """
    使用 sklearn 评估指标替代 Hugging Face evaluate。

    Args:
        model: 当前模型
        data_loader: 测试集的 DataLoader
        cosine_similarity_threshold: 余弦相似度阈值
    """
    model.eval()
    sims, labels, predictions = [], [], []

    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            query_input_ids, query_token_type_ids, \
            doc_input_ids, doc_token_type_ids = batch["query_input_ids"], batch["query_token_type_ids"], \
                                                batch["doc_input_ids"], batch["doc_token_type_ids"]

            query_embedding = model.get_pooled_embedding(
                query_input_ids.to(args.device),
                query_token_type_ids.to(args.device)
            )
            doc_embedding = model.get_pooled_embedding(
                doc_input_ids.to(args.device),
                doc_token_type_ids.to(args.device)
            )
            cos_sim = torch.nn.functional.cosine_similarity(query_embedding, doc_embedding)
            batch_preds = [1 if p else 0 for p in cos_sim > cosine_similarity_threshold]
            batch_labels = batch["labels"].cpu().tolist()

            predictions.extend(batch_preds)
            labels.extend(batch_labels)
            sims.extend(cos_sim.tolist())

    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    spearman_corr = stats.spearmanr(labels, sims).correlation

    model.train()
    return acc, precision, recall, f1, spearman_corr


def train():
    encoder = AutoModel.from_pretrained(args.model)
    model = SimCSE(encoder, dropout=args.dropout)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = load_dataset('text', data_files={'train': args.train_path, 'dev': args.dev_path})
    print(dataset)

    train_dataset = dataset["train"].map(
        partial(convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len, mode='train'),
        batched=True
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size
    )

    eval_dataset = dataset["dev"].map(
        partial(convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len, mode='evaluate'),
        batched=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.batch_size
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
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
    tic_train = time.time()
    global_step, best_f1 = 0, 0

    for epoch in range(1, args.num_train_epochs + 1):
        for batch in train_dataloader:
            query_input_ids, query_token_type_ids, \
            doc_input_ids, doc_token_type_ids = batch["query_input_ids"], batch["query_token_type_ids"], \
                                                batch["doc_input_ids"], batch["doc_token_type_ids"]

            if args.duplicate_ratio > 0:
                query_input_ids, query_token_type_ids = word_repetition(
                    query_input_ids, query_token_type_ids, device=args.device)
                doc_input_ids, doc_token_type_ids = word_repetition(
                    doc_input_ids, doc_token_type_ids, device=args.device)

            loss = model(
                query_input_ids=query_input_ids,
                query_token_type_ids=query_token_type_ids,
                doc_input_ids=doc_input_ids,
                doc_token_type_ids=doc_token_type_ids,
                device=args.device
            )
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

                acc, precision, recall, f1, spearman_corr = evaluate_model(model, eval_dataloader)
                writer.add_scalar('eval/accuracy', acc, global_step)
                writer.add_scalar('eval/precision', precision, global_step)
                writer.add_scalar('eval/recall', recall, global_step)
                writer.add_scalar('eval/f1', f1, global_step)
                writer.add_scalar('eval/spearman_corr', spearman_corr, global_step)
                writer.record()

                print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f, spearman_corr: %.5f"
                      % (precision, recall, f1, spearman_corr))
                if f1 > best_f1:
                    print(f"best F1 performance updated: {best_f1:.5f} --> {f1:.5f}")
                    best_f1 = f1
                    best_dir = os.path.join(args.save_dir, "model_best")
                    os.makedirs(best_dir, exist_ok=True)
                    torch.save(model, os.path.join(best_dir, 'model.pt'))
                    tokenizer.save_pretrained(best_dir)
                tic_train = time.time()


if __name__ == '__main__':
    from rich import print
    train()
