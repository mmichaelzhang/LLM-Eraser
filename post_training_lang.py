'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''

import os
import sys
import argparse
from typing import List

import torch
import transformers
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
import random
import json
import datasets
import numpy as np

from LLMPruner.peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from LLMPruner.utils.prompter import Prompter, ZeroPrompter
from LLMPruner.datasets.ppl_dataset import get_loaders

device = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import Trainer

def gen_data_dict(n_samples, neg_task = None):
    instructions = []
    outputs = []

    instructions_neg = []
    outputs_neg = []


    tasks = ['code', 'en', 'es', 'zh', 'ar']
    neg_task = neg_task.split(',')
    print(neg_task)

    all_datasets = [[json.loads(e) for e in open('./datasets/Lang_data/{}.jsonl'.format(task), 'r').readlines()] for task in tasks]
    pos_pool = []
    neg_pool = []
    for dataset in all_datasets:
        for e in dataset:
            if (tasks[all_datasets.index(dataset)] in neg_task):
                neg_pool.append((e['inputs'], e['outputs']))
            else:
                pos_pool.append((e['inputs'], e['outputs']))
    for _ in range(n_samples):
        i = random.randint(0, len(pos_pool) - 1)
        j = random.randint(0, len(neg_pool) - 1)
        instructions.append(pos_pool[i][0])
        instructions_neg.append(neg_pool[j][0])
        outputs.append(pos_pool[i][1])
        outputs_neg.append(neg_pool[j][1])
    datadict = {'instructions': instructions, 'outputs': outputs, 'instructions_neg': instructions_neg, 'outputs_neg': outputs_neg}
    return datadict
        

class selfcollator(transformers.DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        features1 = [{x:feature[x][0] for _,x in enumerate(feature)} for feature in features]
        features2 = [{x:feature[x][1] for _,x in enumerate(feature)} for feature in features]

        labels = [feature["labels"] for feature in features1] if "labels" in features1[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features1:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features1 = self.tokenizer.pad(
            features1,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        # print(features1)

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features1["labels"])
            features1["decoder_input_ids"] = decoder_input_ids


        labels = [feature["labels"] for feature in features2] if "labels" in features2[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features2:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features2 = self.tokenizer.pad(
            features2,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features2["labels"])
            features2["decoder_input_ids"] = decoder_input_ids

        features = features1
        for _,x in enumerate(features2):
            features[x+'_2'] = features2[x]
        # for i in range(len(features1)):
        #     ft = features1[i]
        #     for _,x in enumerate(features2[i]):
        #         ft[x+'_neg'] = features2[i][x]
        #     features.append(ft)

        return features

class CustomTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        inputs1 = {}
        inputs2 = {}
        inputs1['input_ids'] = inputs['input_ids']
        inputs1['labels'] = inputs['labels']
        inputs1['attention_mask'] = inputs['attention_mask']

        inputs2['input_ids'] = inputs['input_ids_2']
        inputs2['labels'] = inputs['labels_2']
        inputs2['attention_mask'] = inputs['attention_mask_2']

        if self.label_smoother is not None and "labels" in inputs1:
            labels1 = inputs1.pop("labels")
        else:
            labels1 = None
        outputs1 = model(**inputs1)

        # print(outputs1.keys(), outputs2.keys())
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs1[self.args.past_index]

        if labels1 is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss1 = self.label_smoother(outputs1, labels1, shift_labels=True)
            else:
                loss1 = self.label_smoother(outputs1, labels1)
        else:
            if isinstance(outputs1, dict) and "loss" not in outputs1:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs1.keys())}. For reference, the inputs it received are {','.join(inputs1.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss1 = outputs1["loss"] if isinstance(outputs1, dict) else outputs1[0]
        

        if (args.use_neg):
        
            if self.label_smoother is not None and "labels" in inputs2:
                labels2 = inputs2.pop("labels")
            else:
                labels2 = None
            outputs2 = model(**inputs2)

            # print(outputs1.keys(), outputs2.keys())
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs2[self.args.past_index]

            if labels2 is not None:
                if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss2 = self.label_smoother(outputs2, labels2, shift_labels=True)
                else:
                    loss2 = self.label_smoother(outputs2, labels2)
            else:
                if isinstance(outputs2, dict) and "loss" not in outputs2:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs2.keys())}. For reference, the inputs it received are {','.join(inputs2.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss2 = outputs2["loss"] if isinstance(outputs2, dict) else outputs2[0]
            loss = loss1 - 0.01 * loss2
            return (loss, outputs1) if return_outputs else loss
        else:
            return (loss1, outputs1) if return_outputs else loss1

def main(args):
    # Set WanDB
    os.environ["WANDB_PROJECT"] = args.wandb_project
    data = gen_data_dict(10000, args.neg_tasks)
    data = datasets.Dataset.from_dict(data)
    print(data)
    # Load Pruned Model
    pruned_dict = torch.load(args.prune_model, map_location='cpu')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    print("Here")
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    if not args.no_instruction:
        prompter = Prompter(args.prompt_template_name)
    else:
        prompter = ZeroPrompter()

    if device == 'cuda':
        model.half()

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt1, prompt2, add_eos_token=True):
        result = tokenizer(
            [prompt1,prompt2],
            truncation=True,
            max_length=args.cutoff_len,
            padding=True,
            return_tensors=None,
        )
        # print("Pos", prompt1, "Neg", prompt2)
        if (
            result["input_ids"][0][-1] != tokenizer.eos_token_id
            and len(result["input_ids"][0]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"][0].append(tokenizer.eos_token_id)
            result["attention_mask"][0].append(1)

            result["input_ids"][1].append(tokenizer.eos_token_id)
            result["attention_mask"][1].append(1)

        
        result["labels"] = result["input_ids"].copy()

        # result['input_ids'] = [result1['input_ids'], result2['input_ids']]
        # result['attention_mask'] = [result1['attention_mask'], result2['attention_mask']]
        # result["labels"] = [result1["labels"], result2["labels"]]
        
        return result

    def generate_and_tokenize_prompt(data_point):
        # full_prompt1 = prompter.generate_prompt(
        #     data_point["instruction"],
        #     data_point["input"],
        #     data_point["output"],
        # )
        # full_prompt2 = prompter.generate_prompt(
        #     data_point["instruction"][:10],
        #     data_point["input"],
        #     data_point["output"],
        # )
        full_prompt1 = data_point['instructions']+data_point['outputs']
        full_prompt2 = data_point['instructions_neg']+data_point['outputs_neg']

        tokenized_full_prompt = tokenize(full_prompt1, full_prompt2)
        if not args.train_on_inputs:
            # user_prompt1 = prompter.generate_prompt(
            #     data_point["instruction"], data_point["input"]
            # )

            # user_prompt2 = prompter.generate_prompt(
            #     data_point["instruction"], data_point["input"]
            # )
            
            tokenized_user_prompt = tokenize(
                data_point['instructions'], data_point['instructions_neg'], add_eos_token=args.add_eos_token
            )
            user_prompt_len1 = len(tokenized_user_prompt['input_ids'][0])
            user_prompt_len2 = len(tokenized_user_prompt['input_ids'][1])

            # user_prompt_len1 = tokenized_user_prompt['input_ids'][-1]
            # user_prompt_len2 = len(tokenized_user_prompt['input_ids']) - user_prompt_len1 - 1
            
            if args.add_eos_token:
                user_prompt_len1 -= 1
                user_prompt_len2 -= 1
            tokenized_full_prompt["labels"] = [
                [-100] * (user_prompt_len1) + tokenized_full_prompt["labels"][0][user_prompt_len1:],
                [-100] * (user_prompt_len2) + tokenized_full_prompt["labels"][1][user_prompt_len2:]
                ]

            # tokenized_full_prompt["labels"] = [
            #     -100
            # ] * (user_prompt_len1) + tokenized_full_prompt["labels"][
            #     user_prompt_len1:tokenized_full_prompt['labels'][-1]
            # ] + [-100] * (user_prompt_len2) + tokenized_full_prompt["labels"][
            #     tokenized_full_prompt['labels'][-1]+user_prompt_len2:
            # ]
            # could be sped up, probably

        return tokenized_full_prompt

    def split_and_tokenizer(test_data, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(test_data[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        test_set = []
        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_set.append({
                'input_ids': batch,
                'labels': batch
            })
        return test_set

    # Prepare For LoRA
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  

    # Load Train Dataset
    
    train_val = data.train_test_split(
        test_size=200, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = {
        args.data_path: train_val["test"].shuffle().map(generate_and_tokenize_prompt),
    }
    # for e in train_data['input_ids']:
    #     print(torch.tensor(e).size())
    
    # # Load Extra Validation Dataset
    if args.extra_val_dataset:
        from LLMPruner.datasets.ppl_dataset import get_wikitext2, get_ptb

        seq_len = 128
        for extra_dataset in args.extra_val_dataset.split(','):
            if 'wikitext2' in extra_dataset:
                _, test_data = get_wikitext2(seq_len, None)
                test_data = split_and_tokenizer(test_data, tokenizer, seq_len, field_name='text')
            if 'ptb' in extra_dataset:
                _, test_data = get_ptb(seq_len, None)
                test_data = split_and_tokenizer(test_data, tokenizer, seq_len, field_name='sentence')
            val_data[extra_dataset] = test_data

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=10,
            logging_first_step=True,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=200,
            output_dir=args.output_dir,
            save_total_limit=20,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None,
            group_by_length=args.group_by_length,
            report_to="wandb",
            run_name=args.output_dir.split('/')[-1],
            metric_for_best_model="{}_loss".format(args.data_path),
        ),
        data_collator=selfcollator(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    model.state_dict = old_state_dict
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--prune_model', type=str, help='prune model name')
    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=8, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca", help="The prompt template to use, will default to alpaca.")
    parser.add_argument('--no_instruction', action='store_true', default=False, help="Whether to use the instruction template or not.")

    # Lora Configuration
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')

    # llm hyperparameters
    parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")
   
    # wandb params
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")

    parser.add_argument('--neg_tasks', type=str, default='piqa', help='negative tasks')
    parser.add_argument('--use_neg', action='store_true', help='use negative tasks or not')
   
    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version

    main(args)
