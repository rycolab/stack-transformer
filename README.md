# stack-transformer
This repository is in accompany with the paper: [A Transformer with Stack Attention](https://arxiv.org/abs/2405.04515).

## Dependencies
- python 3.11.2
- pytorch 2.0.1+cu118
- jaxlib 0.4.16+cuda11.cudnn86

## Setup
Install required packages:
```
pip install -r requirements.txt
```

## Deterministic Context-Free Tasks
This part of the code is adapted from [neural_networks_chomsky_hierarchy](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/tree/main).
```
cd neural_networks_chomsky_hierarchy/
python training/example.py \
    --batch_size 32 \
    --training_steps 100000 \
    --task $stack \
    --architecture transformer_encoder \
    --stack \
    --pos $pos \
    --seed 0
```
Replace `$task` with one of `["reverse_string", "stack_manipulation", "modular_arithmetic_brackets", "solve_equation"]`,
and `$pos` with one of `["NONE", "SIN_COS", "ALIBI", "RELATIVE", "ROTARY"]`.

## Masked and Causal Language Modeling
```
cd language_modeling/
python run.py \
    --output_dir outputs/ \
    --epochs 1 \
    --seeds 0 \
    --dataset ptb \
    --use_stack \
    --task mlm
```
