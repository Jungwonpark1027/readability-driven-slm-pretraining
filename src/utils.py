import os
import json
import re
import torch
from transformers import TrainerCallback

class SimpleWordCountCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, save_base_path: str, model_type: str, initial_word_count: int = 0):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.save_base_path = save_base_path
        self.model_type = model_type  # "gpt2" or "bert"
        self.total_words_seen = initial_word_count
        
        # Generate milestones in units of 1M, 2M... 10M... 100M
        self.milestones = self._generate_milestones()
        self.last_milestone = self._get_last_milestone(initial_word_count)
        
        print(f"[{model_type.upper()}] Calculating total number of words in the dataset...")
        self.dataset_total_words = self._calculate_dataset_words()
        print(f"Number of words per dataset epoch: {self.dataset_total_words:,} words")

    def _calculate_dataset_words(self) -> int:
        total_words = 0
        for sample in self.dataset:
            input_ids = sample['input_ids']
            
            # PyTorch tensor
            if isinstance(input_ids, torch.Tensor):
                non_pad_mask = input_ids != self.tokenizer.pad_token_id
                non_pad_ids = input_ids[non_pad_mask]
            else:
                non_pad_ids = [idx for idx in input_ids if idx != self.tokenizer.pad_token_id]
                non_pad_ids = torch.tensor(non_pad_ids)

            # Handle special tokens based on model type
            if self.model_type == "bert":
                # For BERT, remove the leading [CLS] and trailing [SEP] tokens before counting words
                if len(non_pad_ids) > 2:
                    non_pad_ids = non_pad_ids[1:-1]
            # For GPT-2, skip_special_tokens=True handles this sufficiently, so no additional slicing is needed
            
            text = self.tokenizer.decode(non_pad_ids, skip_special_tokens=True)
            words = [w for w in text.split() if w.strip()]
            total_words += len(words)
            
        return total_words

    def _generate_milestones(self):
        milestones = []
        milestones.extend([i * 1_000_000 for i in range(1, 10)])      # 1M ~ 9M
        milestones.extend([i * 10_000_000 for i in range(1, 10)])     # 10M ~ 90M
        milestones.extend([i * 100_000_000 for i in range(1, 11)])    # 100M ~ 1B
        return sorted(list(set(milestones)))

    def _get_last_milestone(self, current_count):
        for m in reversed(self.milestones):
            if current_count >= m:
                return m
        return 0

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        self.total_words_seen += self.dataset_total_words
        print(f"\n[Epoch End] Current Word Count: {self.total_words_seen:,} words")
        
        next_milestone = next((m for m in self.milestones if m > self.last_milestone), None)
        
        if next_milestone and self.total_words_seen >= next_milestone:
            print(f"\n*** Reached milestone! ({next_milestone:,} words) - Saving checkpoint***")
            self._save_checkpoint(model, next_milestone, state)

    def _save_checkpoint(self, model, milestone, state):
        milestone_str = f"{milestone // 1_000_000}M"
        save_path = os.path.join(self.save_base_path, f"checkpoint-{milestone_str}-words")
        
        os.makedirs(save_path, exist_ok=True)
        
        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        metadata = {
            "total_words_seen": self.total_words_seen,
            "milestone": milestone,
            "global_step": state.global_step,
            "model_type": self.model_type
        }
        with open(os.path.join(save_path, "word_count_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        self.last_milestone = milestone
        print(f"Saved to: {save_path}\n")


def load_latest_checkpoint(checkpoint_base_dir: str):
    if not os.path.exists(checkpoint_base_dir):
        return None, 0
        
    checkpoints = []
    for d in os.listdir(checkpoint_base_dir):
        if "words" in d:
            try:
                match = re.search(r'checkpoint-(\d+)M-words', d)
                if match:
                    milestone_m = int(match.group(1))
                    word_count = milestone_m * 1_000_000
                    checkpoints.append((word_count, os.path.join(checkpoint_base_dir, d)))
            except ValueError:
                continue
                
    if not checkpoints:
        return None, 0
        
    checkpoints.sort(key=lambda x: x[0])
    latest_word_count, latest_path = checkpoints[-1]
    
    return latest_path, latest_word_count