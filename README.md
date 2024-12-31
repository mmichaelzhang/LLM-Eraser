# LLM-Eraser
Pytorch implementation of paper "LLM-Eraser: Optimizing Large Language Model Unlearning through Selective Pruning"

The code is partially built upon the implementation of LLM-Pruner (https://github.com/horseee/LLM-Pruner).


### Python environment setup with conda

```bash
conda create -n LLM-Eraser python=3.10
pip install -r requirement.txt

```


### Running the Code for language tasks
Before running the code, please make sure to download the base model of llama-2-7b-hf in a proper folder
```bash
# Code for pruning
CUDA_VISIBLE_DEVICES=0 python hf_prune.py --base_model ../base_model_folder/llama-2-7b-hf/ --pruning_ratio 0.1 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name llama-2-7b-hf-0.1-no_neg-minus-code --pruner_type taylor  --taylor param_first --save_model --neg_task code --use_neg
# Code for post-training
CUDA_VISIBLE_DEVICES=0  python post_training_lang.py --prune_model prune_log/llama-2-7b-hf-0.1-no_neg-minus-code/pytorch_model.bin --output_dir tune_log/tune-llama-2-7b-hf-0.1-no_neg-minus-code --wandb_project tune-llama-2-7b-hf-0.1-no_neg-minus-code --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --neg_tasks code --use_neg

```
