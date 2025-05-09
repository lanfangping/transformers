#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for text classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
import warnings
import difflib
import datetime
import copy
from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict
from tqdm import tqdm

import datasets
import evaluate
import numpy as np
import pandas as pd
from datasets import Value, load_dataset
from sklearn.metrics import roc_curve, auc, classification_report
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from categories import EDIT_CATEGORIES
from utils import heatmap
from bertviz import model_view

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.38.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

os.environ["WANDB_PROJECT"] = "manualcmp"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    text_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "sentence" column for single/multi-label classifcation task.'
            )
        },
    )
    text_pair_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text pair column in the input dataset or a CSV/JSON file. "
                'If not specified, it will not use.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "THe delimiter to use to join text columns into a single sentence."}
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classifcation task'
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    enhance_attention_on_difference: Optional[bool] = field(default=False, metadata={"help": "Enhancing the attention weight on the difference part between a text pair."})

    def __post_init__(self):
        if self.dataset_name is None:
            if self.train_file is None or self.validation_file is None:
                raise ValueError(" training/validation file or a dataset name.")

            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

    output_attentions: bool  = field(
        default=False,
        metadata={"help": (
                        "Will enable to output attention weights for model", 
                        "output shape = tuple(Tensor(batch_size, num_heads, sequence_a, sequence_b)(one for each layer))"
                    )
        },
    )

    sample_index_for_output_attentions: Optional[int] = field(
        default=0,
        metadata={"help": (
                        "Will enable to see attention weights for a specific data"
                    )
        },
    )

    draw_attention_heatmap: Optional[bool] = field(
        default=False, 
        metadata={"help": "Draw attention heatmap for the sampled data in `sample_index_for_output_attentions`"}
    )

    inference_only: Optional[bool] = field(
        default=False, 
        metadata={"help": "Do inference only"}
    )


def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a mutli-label dataset"""
    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # print(training_args.device)
    # print("model_args", model_args)
    # print("data_args", data_args)
    # print("training_args", training_args)
    # exit()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files, or specify a dataset name
    # to load from huggingface/datasets. In either case, you can specify a the key of the column(s) containing the text and
    # the key of the column containing the label. If multiple columns are specified for the text, they will be joined togather
    # for the actual text value.
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
        # Try print some info about the dataset
        logger.info(f"Dataset loaded: {raw_datasets}")
        logger.info(raw_datasets)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file
        # if training_args.do_predict:
        if data_args.test_file is not None:
            train_extension = data_args.train_file.split(".")[-1]
            test_extension = data_args.test_file.split(".")[-1]
            assert (
                test_extension == train_extension
            ), "`test_file` should have the same extension (csv or json) as `train_file`."
            data_files["test"] = data_args.test_file
        elif training_args.do_predict:
            raise ValueError("Need either a dataset name or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    if data_args.remove_splits is not None:
        for split in data_args.remove_splits.split(","):
            logger.info(f"removing split {split}")
            raw_datasets.pop(split)

    if data_args.train_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["train"] = raw_datasets[data_args.train_split_name]
        raw_datasets.pop(data_args.train_split_name)

    if data_args.validation_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["validation"] = raw_datasets[data_args.validation_split_name]
        raw_datasets.pop(data_args.validation_split_name)

    if data_args.test_split_name is not None:
        logger.info(f"using {data_args.test_split_name} as test set")
        raw_datasets["test"] = raw_datasets[data_args.test_split_name]
        raw_datasets.pop(data_args.test_split_name)

    if data_args.remove_columns is not None:
        for split in raw_datasets.keys():
            for column in data_args.remove_columns.split(","):
                logger.info(f"removing column {column} from split {split}")
                raw_datasets[split].remove_columns(column)

    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    # convert data to an appropriate datatype 
    def convert_appropriate_datatype(examples):
        examples['label'] = eval(examples['label'])
        return examples
    raw_datasets = raw_datasets.map(convert_appropriate_datatype)

    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    is_regression = (
        raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if data_args.do_regression is None
        else data_args.do_regression
    )
    is_multi_label = False
    if is_regression:
        label_list = None
        num_labels = 1
        # regession requires float as label type, let's cast it if needed
        for split in raw_datasets.keys():
            if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
                logger.warning(
                    f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}"
                )
                features = raw_datasets[split].features
                features.update({"label": Value("float32")})
                try:
                    raw_datasets[split] = raw_datasets[split].cast(features)
                except TypeError as error:
                    logger.error(
                        f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                    )
                    raise error

    else:  # classification
        if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
            is_multi_label = True
            logger.info("Label type is list, doing multi-label classification")
        # Trying to find the number of labels in a multi-label classification task
        # We have to deal with common cases that labels appear in the training set but not in the validation/test set.
        # So we build the label list from the union of labels in train/val/test.
        label_list = get_label_list(raw_datasets, split="train")
        for split in ["validation", "test"]:
            if split in raw_datasets:
                val_or_test_labels = get_label_list(raw_datasets, split=split)
                diff = set(val_or_test_labels).difference(set(label_list))
                if len(diff) > 0:
                    # add the labels that appear in val/test but not in train, throw a warning
                    logger.warning(
                        f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                    )
                    label_list += list(diff)
        # if label is -1, we throw a warning and remove it from the label list
        for label in label_list:
            if label == -1:
                logger.warning("Label -1 found in label list, removing it.")
                label_list.remove(label)

        label_list.sort()
        num_labels = len(label_list)
        if num_labels <= 1:
            raise ValueError("You need more than one label to do classification.")

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        output_attentions=model_args.output_attentions
    )

    if is_regression:
        config.problem_type = "regression"
        logger.info("setting problem type to regression")
    elif is_multi_label:
        config.problem_type = "multi_label_classification"
        logger.info("setting problem type to multi label classification")
    else:
        config.problem_type = "single_label_classification"
        logger.info("setting problem type to single label classification")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # for training ,we will update the config with label infos,
    # if do_train is not set, we will use the label infos in the config
    if training_args.do_train and not is_regression:  # classification, training
        label_to_id = {v: i for i, v in enumerate(label_list)}
        # update config with label infos
        if model.config.label2id != label_to_id:
            logger.warning(
                "The label2id key in the model config.json is not equal to the label2id key of this "
                "run. You can ignore this if you are doing finetuning."
            )
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in label_to_id.items()}
    elif not is_regression:  # classification, but not training
        logger.info("using label infos in the model config")
        logger.info("label2id: {}".format(model.config.label2id))
        label_to_id = model.config.label2id
    else:  # regression
        label_to_id = None

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def multi_labels_to_ids(labels: List[str]) -> List[float]:
        ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
        for label in labels:
            ids[label_to_id[label]] = 1.0
        return ids
    
    def find_first_element_in_list(element, array_list, begin_idx=0):
        for idx in range(begin_idx, len(array_list)):
            if array_list[idx] == element:
                return idx
        return -1

    def find_diff_indexes_in_arrays(array, array_pair, item_real_begin_index=0, item_pair_real_begin_index=0):
        # convert int array to strin array
        array = [str(a) for a in array]
        array_pair = [str(a) for a in array_pair]

        # Creating a Differ object
        d = difflib.Differ()

        # Calculating the difference
        diff = list(d.compare(array, array_pair))
        # print(diff)
        # # Extracting the index of changing words
        # changes = [i for i, word in enumerate(diff) if word.startswith("+ ") or word.startswith("- ")]

        i, j = 0, 0
        old_diff = []
        new_diff = []
        for idx, word in enumerate(diff):
            if word.startswith("+"):
                new_diff.append(i)
                i += 1
            elif word.startswith("-"):
                old_diff.append(j)
                j += 1
            else:
                i += 1
                j += 1

        return old_diff, new_diff

    def process_attention_weight(input_ids):
        attetion_masks = []
        for i, ids in enumerate(input_ids):
            sep_idx1 = find_first_element_in_list(tokenizer.sep_token_id, ids, begin_idx=0)
            array = ids[1:sep_idx1]

            sep_idx2 = find_first_element_in_list(tokenizer.sep_token_id, ids, begin_idx=sep_idx1+1)
            array_pair = ids[sep_idx1+1:sep_idx2]
                
            temp_old_diff, temp_new_diff = find_diff_indexes_in_arrays(array, array_pair)
            old_diff = [idx+1 for idx in temp_old_diff]
            new_diff = [idx+sep_idx1+1 for idx in temp_new_diff]
            changes = old_diff + new_diff

            atte_mask = []
            for idx in range(len(ids)):
                if idx in changes:
                    atte_mask.append(1)
                elif ids[idx] == tokenizer.pad_token_id:
                    atte_mask.append(0)
                else:
                    atte_mask.append(0.5)
            attetion_masks.append(atte_mask)

        return attetion_masks

    def preprocess_function(examples):
        # if data_args.text_column_names is not None and data_args.text_pair_column_names:
        #     text_column_names = data_args.text_column_names.split(",")
        #     # join together text columns into "sentence" column
        #     examples["sentence"] = examples[text_column_names[0]]
        #     for column in text_column_names[1:]:
        #         for i in range(len(examples[column])):
        #             examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]
        if data_args.text_pair_column_name is not None:
            result = tokenizer(examples[data_args.text_column_name], examples[data_args.text_pair_column_name], padding=padding, max_length=max_seq_length, truncation=True)

            # changes the attention weight on difference part
            if data_args.enhance_attention_on_difference:
                result['attention_mask'] = process_attention_weight(result['input_ids'])
        else:
            # Tokenize the texts
            result = tokenizer(examples[data_args.text_pair_column_name], padding=padding, max_length=max_seq_length, truncation=True)
        
        if label_to_id is not None and "label" in examples:
            if is_multi_label:
                result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
            else:
                result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
        return result

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        print(raw_datasets)
        # print(np.shape(raw_datasets['test']['label']))
    # print(raw_datasets['train'][100])
    # print(tokenizer.convert_ids_to_tokens(raw_datasets['train'][100]['input_ids']))
    # exit()
    
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
            else:
                logger.warning("Validation dataset not found. Falling back to test dataset for validation.")
                eval_dataset = raw_datasets["test"]
        else:
            eval_dataset = raw_datasets["validation"]

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        # remove label column if it exists
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if data_args.metric_name is not None:
        metric = (
            evaluate.load(data_args.metric_name, config_name="multilabel", cache_dir=model_args.cache_dir)
            if is_multi_label
            else evaluate.load(data_args.metric_name, cache_dir=model_args.cache_dir)
        )
        logger.info(f"Using metric {data_args.metric_name} for evaluation.")
    else:
        if is_regression:
            metric = evaluate.load("mse", cache_dir=model_args.cache_dir)
            logger.info("Using mean squared error (mse) as regression score, you can use --metric_name to overwrite.")
        else:
            if is_multi_label:
                metric = evaluate.load("f1", config_name="multilabel", cache_dir=model_args.cache_dir)
                logger.info(
                    "Using multilabel F1 for multi-label classification task, you can use --metric_name to overwrite."
                )
            else:
                metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
                logger.info("Using accuracy as classification score, you can use --metric_name to overwrite.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        if is_regression:
            preds = np.squeeze(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
        elif is_multi_label:
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(torch.Tensor(preds)).numpy()
            # next, use threshold to turn them into integer predictions
            preds = np.zeros(probs.shape)
            preds[np.where(probs >= 0.3)] = 1

            # preds = np.array([np.where(p > 0, 1, 0) for p in preds])  # convert logits to multi-hot encoding
            # Micro F1 is commonly used in multi-label classification
            result = metric.compute(predictions=preds, references=p.label_ids, average="micro")
        else:
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # find the optimal threshold for test dataset
    def optimal_threshold(y_true, y_pred):
        y_true = np.concatenate(y_true)
        y_pred = y_pred.ravel()

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        # Calculate Youden's J statistic
        J = tpr - fpr

        # Find the optimal threshold
        optimal_idx = np.argmax(J)
        optimal_threshold = thresholds[optimal_idx]

        return optimal_threshold

    if training_args.do_predict:
        logger.info("*** Predict ***")
        # Removing the `label` columns if exists because it might contains -1 and Trainer won't like that.
        y_true = None
        predict_dataset_copy = copy.deepcopy(predict_dataset)
        if "label" in predict_dataset.features:
            y_true = predict_dataset['label']
            predict_dataset = predict_dataset.remove_columns("label")

        # print(predict_dataset)
        print("predict_dataset_copy", predict_dataset_copy)
        results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        # print("results", results)
        predictions = results.predictions
        # print("predictions", predictions)
        # predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        if is_regression:
            predictions = np.squeeze(predictions)
        elif is_multi_label:
            # # (deprecated)Convert logits to multi-hot encoding. We compare the logits to 0 instead of 0.5, because the sigmoid is not applied.
            # # (deprecated)You can also pass `preprocess_logits_for_metrics=lambda logits, labels: nn.functional.sigmoid(logits)` to the Trainer
            # # (deprecated)and set p > 0.5 below (less efficient in this case)
            # (deprecated)predictions = np.array([np.where(p > 0, 1, 0) for p in predictions])
            # Convert logits to multi-hot encoding and apply sigmod to logits. 
            # We compare the value to optimal threshold or default 0.3.
            sigmoid = torch.nn.Sigmoid()
            if model_args.output_attentions:
                predictions = predictions[0]
            probs = sigmoid(torch.Tensor(predictions)).numpy()

            # find the optimal threshold, other use 0.3 as default threshold
            threshold = 0.3
            if y_true is not None:
                threshold = optimal_threshold(y_true, probs)
            # print("optimal threshold:", threshold)
            
            # next, use threshold to turn them into integer predictions
            predictions = np.zeros(probs.shape)
            predictions[np.where(probs >= threshold)] = 1
        else:
            predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
        output_metrics_file = os.path.join(training_args.output_dir, "predict_metrics.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info("***** Predict results *****")
                writer.write("index\tprediction\n")
                predicted_labels = []
                true_labels = []
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    elif is_multi_label:
                        # recover from multi-hot encoding
                        item = [label_list[i] for i in range(len(item)) if item[i] == 1]
                        predicted_labels.append(item)
                        writer.write(f"{index}\t{item}\n")
                    else:
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")

                def convert_one_hot_labels(example):
                    item = example['label']
                    example['label'] = [label_list[i] for i in range(len(item)) if item[i] == 1]
                    return example

                def add_predicted_labels(example, idx):
                    example["predicted_labels"] = predicted_labels[idx]
                    return example
                
                dataset_with_new_column = predict_dataset_copy.map(add_predicted_labels, with_indices=True)
                dataset_with_new_column = dataset_with_new_column.remove_columns(['input_ids', 'token_type_ids', 'attention_mask'])
                print("dataset_with_new_column", dataset_with_new_column)

                dataset_with_new_column = dataset_with_new_column.map(convert_one_hot_labels)
                output_predict_csv_file = os.path.join(training_args.output_dir, "predict_results.csv")
                dataset_with_new_column.to_csv(output_predict_csv_file, index=False)

            logger.info("Predict results saved at {}".format(output_predict_file))

            if y_true is not None: 
                with open(output_metrics_file, "w") as writer:
                    logger.info("***** Predict metrics *****")
                    report = classification_report(y_true, predictions)
                    print(report) 
                    writer.write("{}".format(str(report)))
                logger.info("Predict metrics saved at {}".format(output_metrics_file))
    
    if model_args.output_attentions:
        model.eval()

        def draw_heatmaps(sample_index):
            predict_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
            record = predict_dataset[sample_index]
            
            record = {k: v.cuda().unsqueeze(0) for k,v in record.items() if k in ['input_ids', 'token_type_ids', 'attention_mask']}

            attentions = model(**record).attentions  
            
            layer = -1 # the last layer
            last_layer_attention = attentions[layer]
            # print("attentions shape", attentions[0].size())
            average_heads_attention = torch.mean(last_layer_attention, dim=1)  # Averaging across heads # For the last layer, shape: [batch_size, num_heads, seq_len, seq_len]
            tokens = tokenizer.convert_ids_to_tokens(record['input_ids'].tolist()[0])  # Convert input ids to token strings
            # print(tokens)
            words = tokenizer.decode(record['input_ids'].tolist()[0])
            # print(words)
            # Assuming you're working with the first item in the batch
            average_attention_np = average_heads_attention[0].cpu().detach().numpy()
            # print("average_attention_np", np.shape(average_attention_np))

            # attention only for CLS token
            attention_CLS = average_attention_np[:][0]
            # print("attention_CLS shape", np.shape(attention_CLS))
            attention_CLS = attention_CLS.reshape((np.shape(attention_CLS)[0], 1))
            # print("attention_CLS reshape", np.shape(attention_CLS))

            # find the [SEP] token, and remove the [PAD] after the [SEP]
            first_SEP_position = None
            second_SEP_position = None
            for i in range(len(tokens)):
                if tokens[i] == '[SEP]':
                    if first_SEP_position is None:
                        first_SEP_position = i
                    elif second_SEP_position is None:
                        second_SEP_position = i
                        break
            # print("second_SEP_position", second_SEP_position)
            attention_first_sentence = attention_CLS[:first_SEP_position+1]
            tokens_first_sentence = tokens[:first_SEP_position+1]
            # print("tokens_first_sentence =", tokens_first_sentence)
            # print("attention_first_sentence =", attention_first_sentence.tolist())
            attention_second_sentence = attention_CLS[first_SEP_position:second_SEP_position+1]
            tokens_second_sentence = tokens[first_SEP_position:second_SEP_position+1]
            # print("tokens_second_sentence =", tokens_second_sentence)
            # print("attention_second_sentence =", attention_second_sentence.tolist())

            # current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            output_heatmap_file = os.path.join(training_args.output_dir, "heatmaps/heatmap{}.png".format(sample_index))
            df1 = pd.DataFrame(attention_first_sentence.tolist())
            df2 = pd.DataFrame(attention_second_sentence.tolist())

            # Determine the global min and max of the data for both heatmaps to use the same color scale
            vmin = min(df1.values.min(), df2.values.min())
            vmax = max(df1.values.max(), df2.values.max())

            # Create a figure to hold the subplots
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4, 12))

            # Increase the space between the heatmaps
            plt.subplots_adjust(hspace=max(len(tokens_first_sentence), len(tokens_second_sentence)) / 1)  # Adjust the vertical space between plots

            # Plot the first heatmap
            sns.heatmap(df1, ax=axs[0], cmap="viridis", yticklabels=tokens_first_sentence, xticklabels=['[CLS]'],  vmin=vmin, vmax=vmax, cbar=False)
            axs[0].set_title('old_snippet')

            # Plot the second heatmap
            heatmap2 = sns.heatmap(df2, ax=axs[1], cmap="viridis", yticklabels=tokens_second_sentence, xticklabels=['[CLS]'], vmin=vmin, vmax=vmax, cbar=False)
            axs[1].set_title('new_snippet')

            # Adjust layout for the colorbar
            plt.subplots_adjust(right=0.85, hspace=0.9)

            plt.tight_layout(pad=4)

            # Create a colorbar
            cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7]) # [left, bottom, width, height]
            fig.colorbar(heatmap2.collections[0], cax=cbar_ax)
            
            # Show the plot
            plt.savefig(output_heatmap_file)

        for i in tqdm(range(len(predict_dataset))):
            draw_heatmaps(i)
        # sample_index = model_args.sample_index_for_output_attentions
        # draw_heatmaps(sample_index)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
