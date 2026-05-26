import datasets
from transformers import AutoTokenizer
import random



class TokenizadorLlamadas():
    def __init__(self, max_length,model_name):
        self.max_length=max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize_function(self,example):
        
        text = example["question"][0] + example["answer"][0]
        # Asegurarse de que el texto termine con el token EOS
        text = text +  self.tokenizer.eos_token  # Añadir el token EOS manualmente
        
        tokenized_inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding=True 
        )
        max_length = min(
            tokenized_inputs["input_ids"].shape[1],
            self.max_length
            )
        
        self.tokenizer.truncation_side = "left"
        tokenized_inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=max_length 
            )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"]
        return tokenized_inputs
    
    def tokenize_and_split_data(self,data_path):
        finetuning_dataset_loaded = datasets.load_dataset("json", data_files=data_path,split="train")
        random.seed(42)
        tokenized_dataset = finetuning_dataset_loaded.map(
            self.tokenize_function, # returns tokenize_function
            batched=True,
            batch_size=1,
            drop_last_batch=True
        )
        tokenized_dataset = tokenized_dataset.with_format("torch")
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)
        return split_dataset