from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoConfig, AutoTokenizer, EarlyStoppingCallback
from modeling_stack_roberta import StackRobertaForMaskedLM
from modeling_stack_gpt2 import StackGPT2LMHeadModel
import math
import argparse
from statistics import mean
import torch
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--use_stack', action="store_true")
parser.add_argument('--pretrained', action="store_true")
parser.add_argument('--output_dir', type=str, default='outputs/pre-trained/with_stack/')
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--seeds', nargs='+', type=int, required=True)
parser.add_argument('--dataset', type=str, default='wikitext-2')
parser.add_argument('--task', type=str, default='mlm')
args = parser.parse_args()
print(args)

if args.dataset == 'wikitext-2':
    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir='./cache/')
    text_column = "text"
elif args.dataset == 'ptb':
# datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
    datasets = load_dataset('ptb_text_only', cache_dir='./cache/')
    text_column = "sentence"
elif args.dataset == 'wikitext-103':
    datasets = load_dataset('wikitext', 'wikitext-103-raw-v1', cache_dir='./cache/')
    text_column = "text"
else:
    raise NameError("Wrong dataset!")

if args.task == 'mlm':
    model_checkpoint = "roberta-base"
    model_class = StackRobertaForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, cache_dir='./cache/')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
elif args.task == 'clm':
    model_checkpoint = "gpt2"
    model_class = StackGPT2LMHeadModel
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, cache_dir='./cache/')
    data_collator = None
else:
    raise NameError("Wrong task!")

config = AutoConfig.from_pretrained(model_checkpoint, cache_dir='./cache/', local_files_only = True)
config.use_stack = args.use_stack

def tokenize_function(examples):
    return tokenizer(examples[text_column])
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=[text_column])

block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)



perplexities = []
for seed in args.seeds:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if args.pretrained:
        model = model_class.from_pretrained(model_checkpoint, config=config)
    else:
        model = model_class(config=config)
    model.resize_token_embeddings(len(tokenizer))
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        report_to="none",
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        save_strategy='epoch',
        save_total_limit=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(5)],
    )

    trainer.train()

    test_results = trainer.evaluate(eval_dataset=lm_datasets['test'])
    perplexity = math.exp(test_results['eval_loss'])
    perplexities.append(perplexity)
    print(f"Perplexity: {perplexity:.2f}")
print(f"Average Perplexity: {mean(perplexities):.2f}")
print(f"Std Perplexity: {np.std(perplexities):.2f}")