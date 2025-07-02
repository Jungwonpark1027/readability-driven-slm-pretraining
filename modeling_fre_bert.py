from transformers import (
    BertConfig, BertTokenizerFast, BertForMaskedLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    TrainerCallback
)
from datasets import load_dataset
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
            if actual_tokens[0] == self.tokenizer.cls_token_id:
                actual_tokens = actual_tokens[1:]
            if len(actual_tokens) > 0 and actual_tokens[-1] == self.tokenizer.sep_token_id:
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
        print(f"Training completed")


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
            if actual_tokens[0] == self.tokenizer.cls_token_id:
                actual_tokens = actual_tokens[1:]
            if len(actual_tokens) > 0 and actual_tokens[-1] == self.tokenizer.sep_token_id:
                actual_tokens = actual_tokens[:-1]
        
        if len(actual_tokens) > 0:
            text = self.tokenizer.decode(actual_tokens, skip_special_tokens=True)
            return len([w for w in text.split() if w.strip()])
        
        return 0


def smart_group_texts(examples, block_size=512):
    valid_texts = [text.strip() for text in examples["text"] if text and text.strip()]
    
    if not valid_texts:
        return {"text": []}
    
    global tokenizer
    
    concatenated_text = " ".join(valid_texts)
    
    all_tokens = tokenizer.encode(concatenated_text, add_special_tokens=False)
    
    result_texts = []
    effective_block_size = block_size - 2
    
    for i in range(0, len(all_tokens), effective_block_size):
        chunk_tokens = all_tokens[i:i + effective_block_size]
        
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
        max_length=512,
        return_overflowing_tokens=False,
        add_special_tokens=True
    )
    return tokens


def main():
    seed = 27
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    global tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("./custom_tokenizer")
    
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        type_vocab_size=2,  
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    
    print(f"=== BERT Training Started ===")
    
    phase_paths = {
        "easy": "./easy.json",
        "medium": "./medium.json",
        "hard": "./hard.json"
    }
    
    global_word_counter = 0
    
    for phase in ["easy", "medium", "hard"]:
        print(f"\n{'='*50}")
        print(f"Phase: {phase.upper()} - BERT Training")
        print(f"Current cumulative word count: {global_word_counter:,} words ({global_word_counter/1_000_000:.1f}M)")
        print(f"{'='*50}")
        
        print(f"\n=== Loading {phase.upper()} data ===")
        raw_dataset = load_dataset("json", data_files=phase_paths[phase])["train"]
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
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        print(f"Tokenization completed: {len(tokenized_dataset):,} samples")
        
        num_epochs = 10
        print(f"\n=== {phase.upper()} Training Plan ===")
        print(f"Total epochs: {num_epochs}")
        
        if phase == "easy":
            print("\nInitializing new BERT model")
            model = BertForMaskedLM(config)
        else:
            prev_phase = "easy" if phase == "medium" else "medium"
            prev_model_path = f"./bert_synthetic_sentence_{prev_phase}_sd27_new"
            
            if os.path.exists(prev_model_path):
                print(f"\nLoading previous phase({prev_phase}) model: {prev_model_path}")
                try:
                    model = BertForMaskedLM.from_pretrained(
                        prev_model_path,
                        local_files_only=True,
                        config=config
                    )
                    print("✓ Previous model loaded successfully")
                except Exception as e:
                    print(f"Failed to load previous model: {e}")
                    print("Initializing new model.")
                    model = BertForMaskedLM(config)
            else:
                print(f"Previous model not found: {prev_model_path}")
                print("Initializing new model.")
                model = BertForMaskedLM(config)
        
        word_callback = SimpleWordCountCallback(tokenizer, tokenized_dataset, global_word_counter)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15  
        )
        
        output_dir = f"./training_bert_synthetic_sentence_{phase}_sd27_new"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=8,
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
            data_collator=data_collator,
            callbacks=[word_callback]
        )
        
        print(f"\n=== {phase.upper()} BERT MLM Training Started ===")
        trainer.train()
        
        global_word_counter = word_callback.total_words_seen
        
        final_save_path = f"./bert_synthetic_sentence_{phase}_sd27_new"
        trainer.save_model(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        
        print(f"\n=== {phase.upper()} Completed ===")
        print(f"Final model saved: {final_save_path}")
    
    print(f"Complete BERT Training Finished")
    


if __name__ == "__main__":
    main()