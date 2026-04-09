import os
import argparse
import yaml
import random
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

# Import your src modules
from src.model import get_model_and_config
from src.dataset import process_dataset_with_chunking
from src.utils import SimpleWordCountCallback, load_latest_checkpoint

# Prevent distributed training communication errors (keep existing settings)
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_all_seed_all(seed)

def main():
    # 1. Set execution arguments
    parser = argparse.ArgumentParser(description="2025 BabyLM Training Pipeline")
    parser.add_argument("--model_type", type=str, required=True, choices=["gpt2", "bert"], 
                        help="Model type to train (gpt2 or bert)")
    parser.add_argument("--config", type=str, default="configs/baby_lm_config.yaml", 
                        help="Path to the config file")
    args = parser.parse_args()

    # 2. Load config file
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg["seed"])
    print(f"[{args.model_type.upper()}] Pipeline started (Seed: {cfg['seed']})")

    # 3. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_path"])
    # For GPT2, pad_token may be missing, so add extra handling
    if args.model_type == "gpt2" and tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>" 

    # 4. Load model configuration and checkpoint
    model_class, model_config = get_model_and_config(args.model_type, tokenizer)
    
    # Separate checkpoint folders by model type
    word_checkpoint_dir = os.path.join(cfg["word_checkpoint_dir"], args.model_type)
    
    latest_ckpt, total_words = load_latest_checkpoint(word_checkpoint_dir)
    if latest_ckpt:
        print(f"-> Loading previous checkpoint: {latest_ckpt} (cumulative word count: {total_words:,})")
        model = model_class.from_pretrained(latest_ckpt)
    else:
        print("-> Initializing new model")
        model = model_class(model_config)

    # 5. Set Data Collator (BERT requires MLM masking)
    data_collator = None
    if args.model_type == "bert":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

    # Set max tokens according to the model
    max_tokens = 1024 if args.model_type == "gpt2" else 512

    # 6. Training loop by curriculum phase
    for phase in ["easy", "medium", "hard"]:
        print(f"\n{'='*20} Phase: {phase.upper()} {'='*20}")
        
        # Load data
        raw_ds = load_dataset("json", data_files=cfg["phases"][phase])["train"]
        
        # Data chunking and tokenization
        tokenized_ds = process_dataset_with_chunking(
            raw_dataset=raw_ds, 
            tokenizer=tokenizer, 
            model_type=args.model_type, 
            max_tokens=max_tokens
        )
        
        # Initialize callback
        word_callback = SimpleWordCountCallback(
            tokenizer=tokenizer, 
            dataset=tokenized_ds, 
            save_base_path=word_checkpoint_dir, 
            model_type=args.model_type, 
            initial_word_count=total_words
        )
        
        # Keep existing training arguments
        training_args = TrainingArguments(
            output_dir=f"./output_{args.model_type}_{phase}",
            num_train_epochs=cfg["num_epochs"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=float(cfg["learning_rate"]),
            save_strategy="no",  # Keep "no" since saving is handled manually in the callback
            logging_steps=100,
            fp16=True,
            dataloader_num_workers=4,
            lr_scheduler_type="linear",
            warmup_steps=500,
            save_safetensors=False,
            seed=cfg["seed"],
            dataloader_drop_last=False,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=tokenized_ds,
            data_collator=data_collator, # None for GPT2, MLM-enabled for BERT
            callbacks=[word_callback]
        )

        # Start training
        trainer.train()

        # Update global counter
        total_words = word_callback.total_words_seen
        
        # Save final model after phase completion (use existing path template)
        final_save_path = os.path.join(
            cfg["checkpoint_base_dir"], 
            f"final_{args.model_type}_{phase}_sd{cfg['seed']}"
        )
        trainer.save_model(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        
        print(f"Phase {phase} completed! Cumulative word count: {total_words:,} words")

if __name__ == "__main__":
    main()