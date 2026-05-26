import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import os
import sys
current_dir = os.getcwd()
src_dir = os.path.join(current_dir, 'masters_degree/cali_go_chat_bot_judith/src')
sys.path.append(src_dir)
from data.tokenizer import TokenizadorLlamadas
from transformers import TrainingArguments, EarlyStoppingCallback, AutoModelForCausalLM, AutoConfig
import torch
from models.trainer import Trainer
import torch.nn as nn

max_length=2048
model_name = "meta-llama/Llama-3.2-1B-Instruct"

judith_tokenizer = TokenizadorLlamadas(max_length,model_name)
tokenized_dataset=judith_tokenizer.tokenize_and_split_data('masters_degree/cali_go_chat_bot_judith/data.json')
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

base_model = AutoModelForCausalLM.from_pretrained(model_name)
base_model.config.eos_token_id = judith_tokenizer.tokenizer.eos_token_id
print(train_dataset[0])
print(train_dataset[1000])
print(judith_tokenizer.tokenizer.batch_decode(train_dataset[1000]["input_ids"], skip_special_tokens=False))


device_count = torch.cuda.device_count()
if device_count > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
base_model.to(device)

trained_model_name = f"judith_llama3.2_1b_15epoch1_bSizeAdamW"
output_dir = trained_model_name

training_args = TrainingArguments(
    # Learning rate
    learning_rate=1.0e-5,
    weight_decay=0.02, # Decaimiento de pesos para mejorar la generalización del modelo
    warmup_steps=100, # Number of warmup steps for learning rate scheduler
    lr_scheduler_type="cosine",
    # Number of training epochs
    num_train_epochs=5,
    # Max steps to train for (each step is a batch of data)
    # Overrides num_train_epochs, if not -1
    # Batch size for training
    per_device_train_batch_size=1,
    # Directory to save model checkpoints
    output_dir=output_dir,
    # Other arguments
    overwrite_output_dir=False, # Overwrite the content of the output directory
    disable_tqdm=False, # Disable progress bars
    eval_steps=100, # Number of update steps between two evaluations
    save_steps=100, # After # steps model is saved
    per_device_eval_batch_size=1, # Batch size for evaluation
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_dir=f'{trained_model_name}/logs',
    logging_steps=1,
    optim="adamw_torch",
    gradient_accumulation_steps = 4,
    gradient_checkpointing=True,
    # Parameters for early stopping
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True
    )

model_flops = (
    base_model.floating_point_ops(
        {
            "input_ids": torch.zeros((1, max_length))
            }
        )
    * training_args.gradient_accumulation_steps
    )

print(base_model)
print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")
print("Flops", model_flops / 1e9, "GFLOPs")

trainer = Trainer(
    model=base_model,
    model_flops=model_flops,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=150)] 
)

training_output = trainer.train()

save_dir = f'{output_dir}/final'

trainer.save_model(save_dir)
print("Saved model to:", save_dir)


final_loss = training_output.training_loss  # Pérdida promedio en el conjunto de entrenamiento
metrics = training_output.metrics  # Métricas del entrenamiento
print("Final Loss:", final_loss)
print("Training Metrics:", metrics)

eval_metrics = trainer.evaluate()
print("Evaluation Metrics:", eval_metrics)

final_results = f"Final Loss: {final_loss} Training metrics: {metrics} Evaluation metrics {eval_metrics}"
with open("train_results.txt", 'w',encoding='latin-1') as file:
    file.write(final_results)