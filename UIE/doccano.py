import os
import time
import argparse
import json
from decimal import Decimal
import numpy as np

from utils import convert_ext_examples, convert_cls_examples


def do_convert():
    tic_time = time.time()
    if not os.path.exists(args.doccano_file):
        raise ValueError("Please input the correct path of doccano file.")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if len(args.splits) != 0 and len(args.splits) != 3:
        raise ValueError("Only []/ len(splits)==3 accepted for splits.")

    def _check_sum(splits):
        return Decimal(str(splits[0])) + Decimal(str(splits[1])) + Decimal(
            str(splits[2])) == Decimal("1")

    if len(args.splits) == 3 and not _check_sum(args.splits):
        raise ValueError(
            "Please set correct splits, sum of elements in splits should be equal to 1."
        )

    with open(args.doccano_file, "r", encoding="utf-8") as f:
        raw_examples = f.readlines()

    def _create_ext_examples(examples,
                             negative_ratio,
                             prompt_prefix="情感倾向",
                             options=["正向", "负向"],
                             separator="##",
                             shuffle=False,
                             is_train=True):
        entities, relations, aspects = convert_ext_examples(
            examples, negative_ratio, prompt_prefix, options, separator,
            is_train)
        examples = entities + relations + aspects
        if shuffle:
            indexes = np.random.permutation(len(examples))
            examples = [examples[i] for i in indexes]
        return examples

    def _create_cls_examples(examples, prompt_prefix, options, shuffle=False):
        examples = convert_cls_examples(examples, prompt_prefix, options)
        if shuffle:
            indexes = np.random.permutation(len(examples))
            examples = [examples[i] for i in indexes]
        return examples

    def _save_examples(save_dir, file_name, examples):
        count = 0
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
        print("Save %d examples to %s." % (count, save_path))

    if len(args.splits) == 0:
        if args.task_type == "ext":
            examples = _create_ext_examples(raw_examples, args.negative_ratio,
                                            args.prompt_prefix, args.options,
                                            args.separator, args.is_shuffle)
        else:
            examples = _create_cls_examples(raw_examples, args.prompt_prefix,
                                            args.options, args.is_shuffle)
        _save_examples(args.save_dir, "train.txt", examples)
    else:
        if args.is_shuffle:
            indexes = np.random.permutation(len(raw_examples))
            index_list = indexes.tolist()
            raw_examples = [raw_examples[i] for i in indexes]

        i1, i2, _ = args.splits
        p1 = int(len(raw_examples) * i1)
        p2 = int(len(raw_examples) * (i1 + i2))

        train_ids = index_list[:p1]
        dev_ids = index_list[p1:p2]
        test_ids = index_list[p2:]

        with open(os.path.join(args.save_dir, "sample_index.json"), "w") as fp:
            maps = {
                "train_ids": train_ids,
                "dev_ids": dev_ids,
                "test_ids": test_ids
            }
            fp.write(json.dumps(maps))

        if args.task_type == "ext":
            train_examples = _create_ext_examples(raw_examples[:p1],
                                                  args.negative_ratio,
                                                  args.prompt_prefix,
                                                  args.options, args.separator,
                                                  args.is_shuffle)
            dev_examples = _create_ext_examples(raw_examples[p1:p2],
                                                -1,
                                                args.prompt_prefix,
                                                args.options,
                                                args.separator,
                                                is_train=False)
            test_examples = _create_ext_examples(raw_examples[p2:],
                                                 -1,
                                                 args.prompt_prefix,
                                                 args.options,
                                                 args.separator,
                                                 is_train=False)
        else:
            train_examples = _create_cls_examples(raw_examples[:p1],
                                                  args.prompt_prefix,
                                                  args.options)
            dev_examples = _create_cls_examples(raw_examples[p1:p2],
                                                args.prompt_prefix,
                                                args.options)
            test_examples = _create_cls_examples(raw_examples[p2:],
                                                 args.prompt_prefix,
                                                 args.options)

        _save_examples(args.save_dir, "train.txt", train_examples)
        _save_examples(args.save_dir, "dev.txt", dev_examples)
        _save_examples(args.save_dir, "test.txt", test_examples)

    print('Finished! It takes %.2f seconds' % (time.time() - tic_time))


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--doccano_file", default="data/simple_ner/doccano_labeled_data.jsonl", type=str, help="The doccano file exported from doccano platform.")
    parser.add_argument("--save_dir", default="data/simple_ner", type=str, help="The path of data that you wanna save.")
    parser.add_argument("--negative_ratio", default=5, type=int, help="Used only for the extraction task, the ratio of positive and negative samples, number of negtive samples = negative_ratio * number of positive samples")
    parser.add_argument("--splits", default=[0.8, 0.1, 0.1], type=float, nargs="*", help="The ratio of samples in datasets. [0.6, 0.2, 0.2] means 60% samples used for training, 20% for evaluation and 20% for test.")
    parser.add_argument("--task_type", choices=['ext', 'cls'], default="ext", type=str, help="Select task type, ext for the extraction task and cls for the classification task, defaults to ext.")
    parser.add_argument("--options", default=["正向", "负向"], type=str, nargs="+", help="Used only for the classification task, the options for classification")
    parser.add_argument("--prompt_prefix", default="情感倾向", type=str, help="Used only for the classification task, the prompt prefix for classification")
    parser.add_argument("--is_shuffle", default=True, type=bool, help="Whether to shuffle the labeled dataset, defaults to True.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization")
    parser.add_argument("--separator", type=str, default='##', help="Used only for entity/aspect-level classification task, separator for entity label and classification label")

    args = parser.parse_args()
    # yapf: enable

    do_convert()