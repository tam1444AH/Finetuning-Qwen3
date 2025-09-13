from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from transformers import EvalPrediction
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
import torch
import pandas as pd
import wandb
from transformers import set_seed
from sklearn.model_selection import train_test_split
import os

set_seed(42)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
    perplexity = torch.exp(loss).item()
    return {"eval_perplexity": perplexity}

class PerplexityLogger(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if state.is_local_process_zero:
            if 'loss' in logs:
                perplexity = torch.exp(torch.tensor(logs['loss'])).item()
                wandb.log({"train_perplexity": perplexity}, step=state.global_step)
    
    # def on_evaluate(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
    #     if state.is_local_process_zero:
    #         if 'eval_loss' in logs:
    #             perplexity = torch.exp(torch.tensor(logs['eval_loss'])).item()
    #             wandb.log({"eval_perplexity": perplexity}, step=state.global_step)
WANDB_KEY = os.environ["WANDB_KEY"]
wandb.login(key=WANDB_KEY)
run = wandb.init(project='InsureHub', job_type="training")

base_model = "cognitivecomputations/dolphin-2.8-mistral-7b-v02" 

bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=False,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    #trust_remote_code=True
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    padding_side="right",
    add_eos_token=True,
    add_bos_token=True
)
tokenizer.pad_token = tokenizer.eos_token

class EmotionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        tweet = item['text']
        encoding = tokenizer(tweet, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Load the original dataset
data_types = ['emotion', 'sentiment', 'offensive', 'hate', 'irony']
base_path = 'data/original_data/'

train_data = {data_type: pd.read_json(f'{base_path}{data_type}/train_{data_type}.json', lines=True) for data_type in data_types}
eval_data = {data_type: pd.read_json(f'{base_path}{data_type}/val_{data_type}.json', lines=True) for data_type in data_types}
test_data = {data_type: pd.read_json(f'{base_path}{data_type}/test_{data_type}.json', lines=True) for data_type in data_types}

# Concatenate all the original data
original_data = pd.concat([train_data[dt] for dt in data_types])
eval_data = pd.concat([eval_data[dt] for dt in data_types])
test_data = pd.concat([test_data[dt] for dt in data_types])

all_data = pd.concat([original_data, eval_data, test_data])
all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Length of all data: {len(all_data)}")

# Remove all the rows with missing values
all_data = all_data.dropna()
print(f"Length of all data after removing NaN: {len(all_data)}")

# Drop duplicates
all_data = all_data.drop_duplicates(subset=['text'], keep='first')
print(f"Length of all data after removing duplicates: {len(all_data)}")

# Remove rows with length less than 10
all_data = all_data[all_data['text'].str.len() >= 10]
print(f"Length of all data after removing rows with length less than 10: {len(all_data)}")

print(f"Length of all data after processing: {len(all_data)}")

train,test = train_test_split(all_data,test_size=0.05,random_state=42)
print(f"Length of train data: {len(train)}")
print(f"Length of test data: {len(test)}")

train_dataset = EmotionDataset(train)
eval_dataset = EmotionDataset(test)

# Calculate total training steps for one epoch
total_train_steps = (len(train_dataset) // 8) // 4  # per_device_train_batch_size=8, gradient_accumulation_steps=4

training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    optim="adamw_bnb_8bit",
    save_steps=500,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    max_grad_norm=1.0,
    max_steps=total_train_steps,
    warmup_ratio=0.05,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="wandb",
    evaluation_strategy="steps", 
    eval_steps=500,
    save_strategy="steps", 
    load_best_model_at_end=True,    
    metric_for_best_model="eval_loss", 
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[PerplexityLogger()]
)

trainer.train()

model.save_pretrained("models/dolphin-2.8-mistral-7b-v02")
tokenizer.save_pretrained("models/dolphin-2.8-mistral-7b-v02")

wandb.finish()
model.config.use_cache = True
model.eval()