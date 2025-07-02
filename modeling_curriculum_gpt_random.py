from transformers import (
    GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments,
    GPT2Tokenizer, TrainerCallback
)
from datasets import Dataset
import os
import random
import numpy as np
import torch
import json
import re
from typing import Dict, Optional, Tuple

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


class SimpleWordCountCallback(TrainerCallback):
    # Above class is implemented to save checkpoints based on word count following BabyLM2025 guidelines.
    # If you prefer a different checkpoint saving strategy (e.g., step-based), feel free to modify
    def __init__(self, tokenizer, dataset, initial_word_count: int = 0):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.total_words_seen = initial_word_count
        
        self.dataset_total_words = self._calculate_dataset_words()
        
        print(f"Initial word count: {initial_word_count:,} ({initial_word_count/1_000_000:.1f}M)")
        print(f"Dataset total words: {self.dataset_total_words:,} ({self.dataset_total_words/1_000_000:.1f}M)")
    
    def _calculate_dataset_words(self) -> int:
        print("Pre-calculating dataset word count...")
        total_words = 0
        
        for i, sample in enumerate(self.dataset):
            words = self._count_words_in_sample(sample['input_ids'])
            total_words += words
            
            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i + 1:,}/{len(self.dataset):,}")
        
        return total_words
    
    def _count_words_in_sample(self, input_ids) -> int:
        if hasattr(input_ids, 'cpu'):
            input_ids = input_ids.cpu()
        
        non_pad_mask = input_ids != self.tokenizer.pad_token_id
        actual_tokens = input_ids[non_pad_mask]
        
        if len(actual_tokens) > 0:
            if actual_tokens[0] == self.tokenizer.bos_token_id:
                actual_tokens = actual_tokens[1:]
            if len(actual_tokens) > 0 and actual_tokens[-1] == self.tokenizer.eos_token_id:
                actual_tokens = actual_tokens[:-1]
        
        if len(actual_tokens) > 0:
            text = self.tokenizer.decode(actual_tokens, skip_special_tokens=True)
            return len([w for w in text.split() if w.strip()])
        
        return 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        print(f"Training started - current word count: {self.total_words_seen:,}")
        self.words_per_epoch = self.dataset_total_words
        print(f"Expected words per epoch: {self.words_per_epoch:,}")
        
        total_epochs = args.num_train_epochs
        expected_final_words = self.total_words_seen + (self.words_per_epoch * total_epochs)
        print(f"Expected total words after training: {expected_final_words:,} ({expected_final_words/1_000_000:.1f}M)")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_words = self.total_words_seen
        print(f"\nEpoch {int(state.epoch) + 1} started - current word count: {self.total_words_seen:,}")
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        self.total_words_seen += self.dataset_total_words
        
        print(f"\nEpoch {int(state.epoch)} completed!")
        print(f"  Words this epoch: {self.dataset_total_words:,}")
        print(f"  Cumulative words: {self.total_words_seen:,} ({self.total_words_seen/1_000_000:.1f}M)")
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % 100 == 0:
            samples_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
            dataset_size = len(self.dataset)
            
            current_epoch = int(state.global_step * samples_per_step / dataset_size)
            steps_in_current_epoch = state.global_step % (dataset_size // samples_per_step)
            total_steps_per_epoch = dataset_size // samples_per_step
            epoch_progress = min(steps_in_current_epoch / total_steps_per_epoch, 1.0)
            
            current_epoch_words = int(self.dataset_total_words * epoch_progress)
            estimated_total_words = self.total_words_seen + current_epoch_words
            
            print(f"Step {state.global_step} (Epoch {current_epoch + 1}, {epoch_progress*100:.1f}%): "
                  f"estimated words {estimated_total_words:,} ({estimated_total_words/1_000_000:.1f}M)")
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        print(f"Training completed! Final word count: {self.total_words_seen:,} words ({self.total_words_seen/1_000_000:.1f}M)")


class DatasetWordCounter:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def count_words_in_dataset(self, dataset) -> int:
        print("Calculating dataset word count...")
        total_words = 0
        
        for i, sample in enumerate(dataset):
            words = self._count_words_in_sample(sample['input_ids'])
            total_words += words
            
            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i + 1:,}/{len(dataset):,} ({total_words:,} words)")
        
        print(f"Dataset total words: {total_words:,} words ({total_words/1_000_000:.1f}M)")
        return total_words
    
    def _count_words_in_sample(self, input_ids) -> int:
        if hasattr(input_ids, 'cpu'):
            input_ids = input_ids.cpu()
        
        non_pad_mask = input_ids != self.tokenizer.pad_token_id
        actual_tokens = input_ids[non_pad_mask]
        
        if len(actual_tokens) > 0:
            if actual_tokens[0] == self.tokenizer.bos_token_id:
                actual_tokens = actual_tokens[1:]
            if len(actual_tokens) > 0 and actual_tokens[-1] == self.tokenizer.eos_token_id:
                actual_tokens = actual_tokens[:-1]
        
        if len(actual_tokens) > 0:
            text = self.tokenizer.decode(actual_tokens, skip_special_tokens=True)
            return len([w for w in text.split() if w.strip()])
        
        return 0


def load_latest_checkpoint(checkpoint_base_dir: str) -> Tuple[Optional[str], int]:
    return None, 0


def load_txt_dataset(file_path: str) -> Dataset:
    print(f"Loading TXT file: {file_path}")
    
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line: 
                texts.append(line)
            
            if line_num % 10000 == 0:
                print(f"  Lines read: {line_num:,}")
    
    print(f"Total {len(texts):,} sentences loaded")
    
    dataset = Dataset.from_dict({"text": texts})
    return dataset


def smart_group_texts(examples, block_size=1024):
    valid_texts = [text.strip() for text in examples["text"] if text and text.strip()]
    
    if not valid_texts:
        return {"text": []}
    
    global tokenizer
    
    concatenated_text = " ".join(valid_texts)
    
    all_tokens = tokenizer.encode(concatenated_text, add_special_tokens=False)
    
    result_texts = []
    for i in range(0, len(all_tokens), block_size):
        chunk_tokens = all_tokens[i:i + block_size]
        
        if len(chunk_tokens) >= 2:
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                result_texts.append(chunk_text.strip())
    
    if not result_texts:
        result_texts = valid_texts
    
    return {"text": result_texts}


def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_overflowing_tokens=False
    )
    tokens["labels"] = tokens["input_ids"]
    return tokens


def main():
    seed = 365
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    global tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("./your_custom_tokenizer")
    tokenizer.pad_token = "<pad>"
    
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    checkpoint_base_dir = "./word_checkpoints"
    os.makedirs(checkpoint_base_dir, exist_ok=True)
    
    latest_checkpoint, cumulative_word_count = load_latest_checkpoint(checkpoint_base_dir)
    
    print(f"=== Training Started ===")
    print(f"Cumulative word count: {cumulative_word_count:,} words ({cumulative_word_count/1_000_000:.1f}M)")
    if latest_checkpoint:
        print(f"Checkpoint: {latest_checkpoint}")
    
    phase_paths = {
        "random": "./set1.txt"
    }
    
    global_word_counter = cumulative_word_count
    
    for phase in ["random"]:
        print(f"\n{'='*50}")
        print(f"Phase: {phase.upper()} - Training Started")
        print(f"Current cumulative word count: {global_word_counter:,} words ({global_word_counter/1_000_000:.1f}M)")
        print(f"{'='*50}")
        
        print(f"\n=== Loading {phase.upper()} data ===")
        raw_dataset = load_txt_dataset(phase_paths[phase])
        print(f"Original data: {len(raw_dataset):,} samples")
        
        grouped_dataset = raw_dataset.map(
            smart_group_texts,
            batched=True,
            remove_columns=raw_dataset.column_names,
            num_proc=16
        )
        print(f"After grouping: {len(grouped_dataset):,} samples")
        
        tokenized_dataset = grouped_dataset.map(
            tokenize_function,
            batched=True,
            load_from_cache_file=False,
            num_proc=16
        )
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        print(f"Tokenization completed: {len(tokenized_dataset):,} samples")
        
        num_epochs = 10
        print(f"\n=== {phase.upper()} Training Plan ===")
        print(f"Total epochs: {num_epochs}")
        
        if phase == "random":
            if latest_checkpoint:
                print(f"\nLoading model from checkpoint: {latest_checkpoint}")
                try:
                    model = GPT2LMHeadModel.from_pretrained(
                        latest_checkpoint,
                        local_files_only=True,
                        config=config
                    )
                except Exception as e:
                    try:
                        model = GPT2LMHeadModel(config)
                        weight_file = os.path.join(latest_checkpoint, "pytorch_model.bin")
                        if os.path.exists(weight_file):
                            state_dict = torch.load(weight_file, map_location='cpu')
                            model.load_state_dict(state_dict)
                        else:
                    except Exception as e2:
                        model = GPT2LMHeadModel(config)
            else:
                print("\nInitializing new model")
                model = GPT2LMHeadModel(config)
        
        word_callback = SimpleWordCountCallback(tokenizer, tokenized_dataset, global_word_counter)
        
        output_dir = f"./training_gpt2_synthetic_random"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            save_strategy="no",  
            logging_steps=100,
            fp16=True,
            dataloader_num_workers=4,
            lr_scheduler_type="linear",
            warmup_steps=500,
            save_safetensors=False,
            seed=seed,
            dataloader_drop_last=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=tokenized_dataset,
            callbacks=[word_callback]
        )
        
        print(f"\n=== {phase.upper()} Training Started ===")
        trainer.train()
        
        global_word_counter = word_callback.total_words_seen
        
        final_save_path = f"./gpt2_synthetic_random"
        trainer.save_model(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        
        metadata = {
            "phase": phase,
            "total_words_seen": int(global_word_counter),
            "epochs": num_epochs,
            "completion_time": "completed"
        }
        
        with open(f"{final_save_path}/training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n=== {phase.upper()} Completed ===")
        print(f"Phase {phase} completed! Cumulative word count: {global_word_counter:,} words ({global_word_counter/1_000_000:.1f}M)")
        print(f"Final model saved: {final_save_path}")
    
    print(f"\n{'='*50}")
    print(f"Complete Training Finished")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()