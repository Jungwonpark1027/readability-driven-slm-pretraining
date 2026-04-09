import re
from typing import List
from datasets import Dataset

def split_into_sentences(text: str) -> List[str]:
    sentence_endings = r'(?<=[.!?])(?:\s*\n\s*|\s+)'
    sentences = re.split(sentence_endings, text.strip())
    return [s.strip() for s in sentences if s.strip()]

def chunk_single_row(text: str, tokenizer, max_tokens: int) -> List[str]:
    if not text or not text.strip(): return []
    sentences = split_into_sentences(text)
    if not sentences: return []
    
    chunks, current_chunk = [], ""
    for sentence in sentences:
        test_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
        test_tokens = tokenizer.encode(test_chunk, add_special_tokens=True)
        
        if len(test_tokens) <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk = sentence
    if current_chunk: chunks.append(current_chunk)
    return chunks

def process_dataset_with_chunking(raw_dataset, tokenizer, model_type: str, max_tokens: int):
    def tokenize_chunked_data(example):
        tokens = tokenizer(example["text"], truncation=True, padding="max_length", 
                           max_length=max_tokens, return_tensors=None)
        
        if model_type == "gpt2":
            tokens["labels"] = tokens["input_ids"]
            
        return tokens

    print("Chunking data...")
    chunked_data = []
    for i, example in enumerate(raw_dataset):
        chunks = chunk_single_row(example["text"], tokenizer, max_tokens)
        for chunk in chunks:
            chunked_data.append({"text": chunk, "original_row_index": i})
    
    ds = Dataset.from_list(chunked_data)
    
    print(f"Tokeninzing... (model: {model_type}, max_tokens: {max_tokens})")
    tokenized_ds = ds.map(tokenize_chunked_data, batched=True, num_proc=16, remove_columns=ds.column_names)
    
    columns = ["input_ids", "attention_mask"]
    if model_type == "gpt2":
        columns.append("labels")
        
    tokenized_ds.set_format(type="torch", columns=columns)
    return tokenized_ds