from data.tokenizer import TokenizadorLlamadas
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
import torch
from models.trainer import Trainer

max_length=2048
model_name = "meta-llama/Llama-3.2-3B-Instruct"

my_tokenizer = TokenizadorLlamadas(max_length,model_name)
tokenized_dataset=my_tokenizer.tokenize_and_split_data('masters_degree/cali_go_chat_bot_judith/data.json')
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

base_model = AutoModelForCausalLM.from_pretrained(model_name)

device_count = torch.cuda.device_count()
if device_count > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
base_model.to(device)

max_steps=2596
trained_model_name = f"judith_llama3.2_3b{max_steps}_steps"
output_dir = trained_model_name

training_args = TrainingArguments(
    # Learning rate
    learning_rate=1.0e-5,
    # Number of training epochs
    num_train_epochs=5,
    # Max steps to train for (each step is a batch of data)
    # Overrides num_train_epochs, if not -1
    max_steps=max_steps,
    # Batch size for training
    per_device_train_batch_size=1,
    # Directory to save model checkpoints
    output_dir=output_dir,
    # Other arguments
    overwrite_output_dir=False, # Overwrite the content of the output directory
    disable_tqdm=False, # Disable progress bars
    eval_steps=120, # Number of update steps between two evaluations
    save_steps=120, # After # steps model is saved
    warmup_steps=1, # Number of warmup steps for learning rate scheduler
    per_device_eval_batch_size=1, # Batch size for evaluation
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=1,
    optim="adafactor",
    gradient_accumulation_steps = 4,
    gradient_checkpointing=False,
    # Parameters for early stopping
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    greater_is_better=False
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
    total_steps=max_steps,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

eval_metrics = trainer.evaluate()
print("Evaluation Metrics before finetuning:", eval_metrics)

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