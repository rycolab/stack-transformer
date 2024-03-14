# stack-transformer
This repository is in accompany with the paper: [A Transformer Encoder with Stack Attention]().

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