import yaml
import torch
import random
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, GPT2Tokenizer
from datasets import load_dataset
from src.dataset import process_dataset_with_chunking
from src.utils import SimpleWordCountCallback, load_latest_checkpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_all_seed(seed)

def train():
    with open("configs/baby_lm_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg['seed'])

    tokenizer = GPT2Tokenizer.from_pretrained(cfg['tokenizer_path'])
    tokenizer.pad_token = "<pad>"

    model_config = GPT2Config(vocab_size=len(tokenizer), n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12, pad_token_id=tokenizer.pad_token_id)
    
    last_ckpt, total_words = load_latest_checkpoint(cfg['checkpoint_base_dir'])
    model = GPT2LMHeadModel.from_pretrained(last_ckpt) if last_ckpt else GPT2LMHeadModel(model_config)

    for phase in ["easy", "medium", "hard"]:
        raw_ds = load_dataset("json", data_files=cfg['phases'][phase])["train"]
        tokenized_ds = process_dataset_with_chunking(raw_ds, tokenizer, cfg['max_tokens'])
        
        word_callback = SimpleWordCountCallback(tokenizer, tokenized_ds, cfg['word_checkpoint_dir'], total_words)
        
        train_args = TrainingArguments(
            output_dir=f"./output_{phase}",
            num_train_epochs=cfg['num_epochs'],
            per_device_train_batch_size=cfg['batch_size'],
            gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
            learning_rate=float(cfg['learning_rate']),
            fp16=True,
            save_strategy="no",
            logging_steps=100
        )

        trainer = Trainer(model=model, args=train_args, train_dataset=tokenized_ds, callbacks=[word_callback])
        trainer.train()
        
        total_words = word_callback.total_words_seen
        model.save_pretrained(f"./final_model_{phase}")

if __name__ == "__main__":
    train()