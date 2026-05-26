from data.tokenizer import TokenizadorLlamadas
from transformers import AutoModelForCausalLM
from models.inference import inference
import torch 

max_length=2048
model_name = "meta-llama/Llama-3.2-1B-Instruct"
max_output=100

my_tokenizer = TokenizadorLlamadas(max_length,model_name)
tokenized_dataset=my_tokenizer.tokenize_and_split_data('masters_degree/cali_go_chat_bot_judith/data.json')
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]
trained_model_name = 'judith_llama3.2_1b_15epoch1_bSizeAdamW/final'


device_count = torch.cuda.device_count()
if device_count > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

trained_model = AutoModelForCausalLM.from_pretrained(trained_model_name, local_files_only=True)
trained_model.to(device) 

test_question = "donde se encuentran ubicados?"
finetuned_model_answer=inference(test_question, trained_model, my_tokenizer.tokenizer,max_input_tokens=max_length, max_output_tokens=max_output)

print("Question input (test): ", test_question)
print("Finetuned slightly model's answer: ",finetuned_model_answer)

test_answer = test_dataset[70]['answer']
print("Real Answer (test):", test_answer)



##cambiar modelo a modelo llama multilenguaje probar el llama 3.2b
