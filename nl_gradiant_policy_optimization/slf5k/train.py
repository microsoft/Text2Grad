from dataclasses import dataclass, field
from typing import Optional
import yaml
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    Adafactor, 
    AutoTokenizer, 
    HfArgumentParser, 
    pipeline, 
    AutoModelForCausalLM, 
    GenerationConfig,
    get_scheduler
)
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import set_seed, LengthSampler
from text2grad_trainer import TEXT2GRADTrainer

import wandb
import re
import os
import tyro
import json
from typing_extensions import Annotated
import datetime
import traceback
import safetensors


JSONDict = Annotated[Optional[dict], tyro.conf.arg(metavar="JSON", constructor=json.loads)]

tqdm.pandas()
os.environ["WANDB_API_KEY"] = "ef77ea5b53addca05920b3d22648e70abfd40029"
os.environ["WANDB_MODE"] = "online"


class QACDataset(Dataset):
    def __init__(self, json_path, tokenizer, prompt_max_length, max_length):
        with open(json_path, "r") as json_f:
            self.json_data = json.load(json_f)
        self.keys = list(self.json_data.keys())
        self.prompt_max_length = prompt_max_length
        self.max_length = max_length
        self.answer_max_length = max_length - prompt_max_length - 1
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]
        value = self.json_data[key]
        post = value["post"]
        ideal_human_summary = value["ideal_human_summary"]
        question = post
        answer = ideal_human_summary
        cur_example = self.build_dataset(question, answer)
        return cur_example

    def build_dataset(self, question, answer):
        def preprocess_function(question, answer):
            query = question
            new_question = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n" + \
                           "Please provide a comprehensive and concise summary of the following text. " + \
                           "Focus on the main ideas and key points while maintaining clarity and coherence.\n\n" + \
                           "Text: " + query + "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            query_num = len(answer.split(","))
            input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(new_question)),
                                     dtype=torch.long)
            attention_mask = torch.ones(input_ids.shape)
            new_examples = {
                "query": question,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "query_nums": query_num,
                "answer": answer,
            }
            return new_examples

        new_examples = preprocess_function(question, answer)
        return new_examples


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    base_model_name: Optional[str] = field(default="", metadata={"help": "The name of the base model to use."})
    base_model_adapter_model: Optional[str] = field(default="",
                                                    metadata={"help": "The name of the adapter model to use."})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="linear", metadata={"help": "the learning rate scheduler type"})
    output_max_length: Optional[int] = field(default=400, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.5,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=True, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=5, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="ckpt/superw_token_ppo", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})

    train_epochs: Optional[int] = field(default=2, metadata={"help": "number of epochs"})
    steps: Optional[int] = field(default=1200, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    local_rank: Optional[int] = field(default=0, metadata={"help": "local rank"})
    project_name: Optional[str] = field(default="superw_token_ppo", metadata={"help": "wandb project name"})
    data_file_path: Optional[str] = field(default="", metadata={"help": "data file path"})
    valid_data_file_path: Optional[str] = field(default="", metadata={"help": "data file path for validation"})
    tracker_kwargs: Optional[str] = field(default=None, metadata={"help": "tracker kwargs of wandb"})
    prompt_max_length: Optional[int] = field(default=1000, metadata={"help": "the length of prompt"})
    answer_max_length: Optional[int] = field(default=500, metadata={"help": "the length of answer"})
    kl_penalty: Optional[str] = field(default="full", metadata={"help": "way of kl penalty"})
    mask_loss: Optional[str] = field(default="", metadata={"help": "mask_loss"})
    reward_mode: Optional[int] = field(default=0, metadata={"help": "reward mode for reward model"})
    strategy: Optional[str] = field(default="", metadata={"help": "strategy of prompt"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

# 将 JSON 字符串转换成字典
if script_args.tracker_kwargs:
    tracker_kwargs_dict = json.loads(script_args.tracker_kwargs)
else:
    tracker_kwargs_dict = {}

if not os.path.exists(script_args.output_dir):
    os.makedirs(script_args.output_dir)

reward_model_name = script_args.reward_model_name

config = PPOConfig(
    vf_coef=0.1,
    steps=script_args.steps,
    model_name=script_args.base_model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
    tracker_kwargs=tracker_kwargs_dict,
    tracker_project_name=script_args.project_name,
    kl_penalty=script_args.kl_penalty
)

# prompt format
with open("./rm_instruction.yaml", 'r') as file:
    instrcution_config = yaml.safe_load(file)

project_name = script_args.output_dir.split("/")[-1]
wandb.init(project=project_name, name="text2grad-slf5k")

tokenizer = AutoTokenizer.from_pretrained(
    script_args.base_model_name,
    model_max_length=1545,
    use_fast=True,
    padding_side='right'
)

reward_tokenizer = AutoTokenizer.from_pretrained(
    script_args.reward_model_name,
    model_max_length=1700,
    use_fast=True,
    padding_side='right'
)  # 1024+521 =

if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

if not reward_tokenizer.pad_token:
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# We retrieve the dataloader by calling the `build_dataset` function.
ds = QACDataset(script_args.data_file_path, tokenizer, script_args.prompt_max_length, script_args.answer_max_length)

# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    script_args.base_model_name,
    device_map="balanced",
    peft_config=lora_config,
)

if script_args.base_model_adapter_model:
    adapter_path = script_args.base_model_adapter_model
    print(f"Loading adapter model from {adapter_path}")
    try:
        # Load adapter weights
        adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            adapter_model_state = safetensors.torch.load_file(adapter_file)
            
            # Load value head weights
            v_head_file = os.path.join(adapter_path, "pytorch_model.bin")
            if os.path.exists(v_head_file):
                v_head = torch.load(v_head_file, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
                
                # Process adapter weight keys to match expected format
                new_adapter_model_state = {
                    "pretrained_model." + k[:-7] + ".default" + k[-7:]: v
                    for k, v in adapter_model_state.items()
                }
                
                # Merge model state dictionaries
                model_dict = model.state_dict()
                new_adapter_model_state.update(v_head)
                model_dict.update(new_adapter_model_state)
                
                # Load the merged state dictionary
                model.load_state_dict(model_dict, strict=False)
                print("Successfully loaded adapter model and value head")
            else:
                raise FileNotFoundError(f"Value head file not found at {v_head_file}")
        else:
            raise FileNotFoundError(f"Adapter model file not found at {adapter_file}")
        
        # Load optimizer state if available
        optimizer_path = os.path.join(adapter_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            print(f"Loading optimizer state from {optimizer_path}")
            try:
                # Use the same device as the model for loading optimizer state
                device = next(model.parameters()).device
                optimizer_state = torch.load(optimizer_path, map_location=device)
                
                # Load optimizer state
                if 'optimizer' in optimizer_state:
                    optimizer.load_state_dict(optimizer_state['optimizer'])
                    print("Successfully loaded optimizer state")
                else:
                    print("Warning: Optimizer state dictionary has unexpected format")
                
                # Load learning rate scheduler state if available
                if 'lr_scheduler' in optimizer_state and lr_scheduler is not None:
                    lr_scheduler.load_state_dict(optimizer_state['lr_scheduler'])
                    print("Successfully loaded learning rate scheduler state")
            except Exception as e:
                print(f"Error loading optimizer state: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                print("Continuing with fresh optimizer")
        else:
            print("No optimizer state found, starting with fresh optimizer")
            
    except Exception as e:
        print(f"Error loading adapter model: {str(e)}")
        traceback.print_exc()
        raise RuntimeError(f"Failed to load adapter model: {str(e)}")

# Ensure value head parameters are trainable
for param in model.v_head.parameters():
    param.requires_grad = True  # Optimize value head

print("Start to print all the parameters needed to be optimized!")
for name, value in model.named_parameters():
    if value.requires_grad:
        print(name + "\n")

ref_model = None

if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
        weight_decay=1e-4
    )

lr_scheduler = get_scheduler(
    name=script_args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=script_args.steps,
)

ppo_trainer = TEXT2GRADTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=ds,
    data_collator=collator,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
)

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

from accelerate.utils import infer_auto_device_map

# Load reward model efficiently by reusing the same model instance
try:
    print(f"Loading reward model from {script_args.reward_model_name}")
    
    # Load the model once with proper error handling
    reward_model_instance = AutoModelForCausalLM.from_pretrained(
        script_args.reward_model_name,
        torch_dtype=torch.float16,
        device_map="balanced"
    )
    
    # Create the pipeline using the loaded model
    reward_model = pipeline(
        "text-generation",
        model=reward_model_instance,
        tokenizer=reward_tokenizer
    )
    
    print(f"Successfully loaded reward model with device map: {reward_model_instance.hf_device_map}")
except Exception as e:
    print(f"Error loading reward model: {str(e)}")
    traceback.print_exc()
    raise RuntimeError(f"Failed to load reward model: {str(e)}")

# for ppo trainer
generation_kwargs = {
    "temperature": 0.6,     
    "do_sample": True,        
    "top_p": 0.9,                 
    "num_beams": 2,               
    "repetition_penalty": 1.2,      
    "max_new_tokens": 100,      
    "min_new_tokens": 45,     
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

# for reward model
sent_kwargs = {
    "batch_size": script_args.batch_size,
    "do_sample": False,
    "temperature": 0.05,          
    "num_beams": 3,              
    "min_new_tokens": 20,          
    "max_new_tokens": 500, 
    "pad_token_id": reward_tokenizer.eos_token_id,
    "eos_token_id": reward_tokenizer.eos_token_id,
}

output_min_length = 45
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)


def assign_word_rewards(answer, score_list):
    """
    Process word rewards with error handling and type checking
    
    Args:
        answer: The generated response text
        score_list: List of (word, score) pairs from the reward model
        
    Returns:
        Tuple of (word_rewards, words) lists
    """
    try:
        # Type checking - accept both list and tuple
        if not isinstance(score_list, (list, tuple)):
            print(f"Warning: score_list is not a list or tuple, got {type(score_list)}")
            return [], []

        # Convert tuple to list if needed
        if isinstance(score_list, tuple):
            score_list = list(score_list)

        # Check list content format
        if not score_list or not all(isinstance(item, (list, tuple)) for item in score_list):
            print(f"Warning: Invalid score_list format: {score_list[:10]}...")
            return [], []

        words = []
        word_rewards = []

        for item in score_list:
            try:
                # Ensure item is a two-element list or tuple
                if len(item) != 2:
                    print(f"Warning: Invalid item length: {item}")
                    continue

                word, score = item

                # Ensure word is string and score is numeric
                if not isinstance(word, str):
                    print(f"Warning: word is not string: {word}")
                    continue

                if not isinstance(score, (int, float)):
                    print(f"Warning: score is not numeric: {score}")
                    continue

                # Clean word
                word = re.sub(r'^[^a-zA-Z0-9$￥&]+|[^a-zA-Z0-9$￥&]+$', '', word)
                word = re.sub(r"[''‛]", "'", word)

                if word:  # Only add non-empty words
                    words.append(word)
                    word_rewards.append(float(score))

            except Exception as e:
                print(f"Error processing item {item}: {str(e)}")
                continue

        if not words:  # If no valid words
            print("Warning: No valid words extracted")
            return [], []

        return word_rewards, words

    except Exception as e:
        print(f"Error in assign_word_rewards: {str(e)}")
        traceback.print_exc()
        return [], []


def check_and_fix_tensor(tensor, eos_id):
    """
    Ensure tensor has proper EOS token structure
    
    Args:
        tensor: Input tensor to check and fix
        eos_id: The end-of-sequence token ID
        
    Returns:
        Fixed tensor with proper EOS token
    """
    # Ensure input is a 1D Tensor
    if len(tensor.shape) != 1:
        raise ValueError("Input tensor must be 1D.")

    # Check if there are multiple EOS tokens at the end
    while len(tensor) > 1 and tensor[-1] == eos_id and tensor[-2] == eos_id:
        tensor = tensor[:-1]  # Remove extra EOS tokens

    # If there's no EOS token at the end, add one
    if tensor[-1] != eos_id or len(tensor) == 0:
        tensor = torch.cat([tensor, torch.tensor([eos_id], dtype=tensor.dtype, device=tensor.device)])

    return tensor


def generate_prompt(system, prompt):
    """
    Generate a formatted prompt with system and user content
    
    Args:
        system: System content
        prompt: User prompt
        
    Returns:
        Formatted prompt string
    """
    p = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{system}\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    return p


def prepare_input_data(queries, responses, question_template):
    """
    Prepares input data with system and user prompts
    
    Args:
        queries: List of original posts
        responses: List of generated summaries
        question_template: Template for the question
        
    Returns:
        List of formatted prompts
    """
    input_datas = []
    for query, response in zip(queries, responses):
        # Escape quotes in query and response to prevent JSON formatting issues
        query_escaped = query.replace('"', '\\"')
        response_escaped = response.replace('"', '\\"')

        system = f"""# User Input
{{
  "original_post": "{query_escaped}",
  "generated_summary": "{response_escaped}"
}}
"""
        input_datas.append(generate_prompt(question_template, system))
    return input_datas


def inference_reward(reward_model, input_datas):
    """
    Run inference with the reward model with enhanced error handling
    
    Args:
        reward_model: The reward model pipeline
        input_datas: List of input prompts
        
    Returns:
        List of processed results from the reward model
    """
    try:
        reward_model.model.eval()
        with torch.no_grad():
            print("Start inference.")
            results = reward_model(input_datas, **sent_kwargs)
            print("Finish inference.")

            # Validate result format
            processed_results = []
            for result in results:
                try:
                    if isinstance(result, (list, tuple)) and len(result) > 0:
                        if isinstance(result[0], dict) and 'generated_text' in result[0]:
                            processed_results.append(result)
                        else:
                            print(f"Warning: Invalid result format: {result}")
                            processed_results.append([{'generated_text': ''}])
                    else:
                        print(f"Warning: Invalid result structure: {result}")
                        processed_results.append([{'generated_text': ''}])
                except Exception as e:
                    print(f"Error processing result: {str(e)}")
                    processed_results.append([{'generated_text': ''}])

            return processed_results

    except Exception as e:
        print(f"Error in inference_reward: {str(e)}")
        traceback.print_exc()
        return [[{'generated_text': ''}] for _ in input_datas]]  # Return empty results list


wandb_step = 0
cur_epoch = 0
cur_step = 0
if script_args.base_model_adapter_model:
    cur_epoch = int(script_args.base_model_adapter_model.split("/")[-1].split("_")[1])
    cur_step = int(script_args.base_model_adapter_model.split("/")[-1].split("_")[-1])
    num_batches_per_epoch = len(ds) // script_args.batch_size

    num_batches_per_epoch += (1 if len(ds) % script_args.batch_size != 0 else 0)

    # Update wandb_step for tracking
    wandb_step = cur_epoch * num_batches_per_epoch + cur_step

    # Calculate number of samples to skip for resuming training
    total_samples_to_skip = cur_epoch * (
            num_batches_per_epoch * script_args.batch_size) + cur_step * script_args.batch_size

    # Set random seed to ensure consistent data ordering
    set_seed(script_args.seed)

    # Create DataLoader with proper generator for reproducibility
    g = torch.Generator()
    g.manual_seed(script_args.seed)

    ppo_trainer.dataloader = DataLoader(
        ds,
        batch_size=script_args.batch_size,
        shuffle=True,
        generator=g,
        collate_fn=collator,
        drop_last=True,
    )

    # Skip previously processed samples
    for _ in range(total_samples_to_skip // script_args.batch_size):
        next(iter(ppo_trainer.dataloader))


def log_error(error_dir, step, error, context=None):
    """
    Helper function to log errors with detailed information
    
    Args:
        error_dir: Directory to save error logs
        step: Current training step
        error: The exception object
        context: Optional dictionary with additional context information
    """
    os.makedirs(error_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    error_file = os.path.join(error_dir, f"error_step_{step}_{timestamp}.txt")

    with open(error_file, "w") as f:
        f.write(f"Error occurred at step {step}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Exception type: {type(error).__name__}\n")
        f.write(f"Exception message: {str(error)}\n")
        if context:
            f.write("\nContext:\n")
            for key, value in context.items():
                f.write(f"{key}: {value}\n")
        f.write("\nFull traceback:\n")
        f.write(traceback.format_exc())


def extract_word_scores_directly(text, log_details=False):
    """
    Directly extract word-score pairs from text without relying on JSON parsing
    """
    try:
        # Pattern to match word-score pairs in various formats
        patterns = [
            r'"([^"]+)"\s*:\s*([-+]?\d*\.?\d+)',  # "word": 0.5
            r'([a-zA-Z0-9$￥&\']+)\s*:\s*([-+]?\d*\.?\d+)',  # word: 0.5
            r'\(([^,]+),\s*([-+]?\d*\.?\d+)\)'  # (word, 0.5)
        ]

        all_matches = []
        pattern_counts = []
        pattern_samples = []

        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, text)
            pattern_counts.append(len(matches))

            # Store samples of each pattern match
            if matches and log_details:
                sample = matches[:3] if len(matches) > 3 else matches
                pattern_samples.append(sample)

            all_matches.extend(matches)

        if log_details:
            print(f"Direct extraction found: {pattern_counts} matches for each pattern, total: {len(all_matches)}")
            for i, samples in enumerate(pattern_samples):
                if samples:
                    print(f"Pattern {i + 1} samples: {samples}")

        if all_matches:
            result = [(word.strip('"\''), float(score)) for word, score in all_matches]
            if log_details:
                print(f"Extracted {len(result)} valid word-score pairs")
                # Log a sample of the extracted pairs
                if result:
                    sample = result[:3] if len(result) > 3 else result
                    print(f"Sample pairs: {sample}")
            return result
        else:
            if log_details:
                print("No word-score pairs extracted")
                # Log text fragments to help debugging
                print(f"Text sample (first 200 chars): {text[:200]}...")
                print(f"Text sample (last 200 chars): {text[-200:] if len(text) > 200 else text}")
            return []
    except Exception as e:
        print(f"Error in direct word-score extraction: {str(e)}")
        print(f"Text sample (first 200 chars): {text[:200]}...")
        traceback.print_exc()
        return []


def load_json_from_string(text, log_details=False):
    """
    Try to extract JSON from text using regex patterns first, then fallback to json.loads
    """
    try:
        # First try to find JSON pattern with regex
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        matches = re.findall(json_pattern, text)

        if matches:
            if log_details:
                print(f"Found {len(matches)} potential JSON matches")
                for i, match in enumerate(matches[:2]):  # Only show first two matches
                    print(f"Match {i + 1} sample (first 100 chars): {match[:100]}...")

            for i, match in enumerate(matches):
                try:
                    result = json.loads(match)
                    if log_details:
                        print(f"Successfully parsed JSON from match {i + 1}")
                        if isinstance(result, dict):
                            print(f"JSON keys: {list(result.keys())}")
                            if "word_score_list" in result:
                                word_score_sample = result["word_score_list"][:3] if len(
                                    result["word_score_list"]) > 3 else result["word_score_list"]
                                print(f"word_score_list sample: {word_score_sample}")
                    return result
                except json.JSONDecodeError as e:
                    if log_details:
                        print(f"Failed to parse match {i + 1}: {str(e)}")
                    continue
        else:
            if log_details:
                print("No JSON pattern matches found with regex")

        # If regex approach fails, try direct json loading
        if log_details:
            print("Attempting direct JSON loading")
        return json.loads(text)
    except Exception as e:
        if log_details:
            print(f"Error parsing JSON: {str(e)}")
            print(f"Problematic text (first 200 chars): {text[:200]}...")
            print(f"Problematic text (last 200 chars): {text[-200:] if len(text) > 200 else text}")

        # Try to extract word-score pairs directly with regex as fallback
        try:
            if log_details:
                print("Attempting word-score extraction fallback")
            word_score_pattern = r'"([^"]+)"\s*:\s*([-+]?\d*\.?\d+)'
            word_score_matches = re.findall(word_score_pattern, text)
            if word_score_matches:
                if log_details:
                    print(f"Found {len(word_score_matches)} word-score pairs with fallback regex")
                    sample = word_score_matches[:3] if len(word_score_matches) > 3 else word_score_matches
                    print(f"Sample pairs: {sample}")
                return {"word_score_list": [(word, float(score)) for word, score in word_score_matches]}
            else:
                if log_details:
                    print("No word-score pairs found with fallback regex")
        except Exception as nested_e:
            if log_details:
                print(f"Regex extraction fallback failed: {str(nested_e)}")

        return None


for epoch in range(script_args.train_epochs):
    if epoch < cur_epoch:
        continue

    # Create log directory for this epoch
    epoch_log_dir = os.path.join(script_args.output_dir, f"logs/epoch_{epoch}")
    os.makedirs(epoch_log_dir, exist_ok=True)

    for step, batch in tqdm(enumerate(ppo_trainer.dataloader), desc=f"Epoch {epoch + 1} "):
        try:
            if epoch == cur_epoch and step <= cur_step:
                continue

            # Create log directory for this step
            step_log_dir = os.path.join(epoch_log_dir, f"step_{step}")
            os.makedirs(step_log_dir, exist_ok=True)

            question_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(
                batch["input_ids"],
                return_prompt=False,
                length_sampler=output_length_sampler,
                **generation_kwargs,
            )

            for ind in range(len(response_tensors)):
                response_tensors[ind] = check_and_fix_tensor(response_tensors[ind], tokenizer.eos_token_id)
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            batch["response"] = [text.strip("assistant") for text in batch["response"]]
            print(batch["response"][0])
            input_datas = prepare_input_data(batch["query"], batch["response"],
                                             instrcution_config[script_args.strategy])
            result = inference_reward(reward_model, input_datas)
            rewards = []
            words = []
            final_question_tensors = []
            final_response_tensors = []
            new_responses = []
            new_questions = []
            fail = 0

            # Record batch-level information
            batch_log = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch,
                "step": step,
                "batch_size": len(batch["query"]),
                "successful_samples": 0,
                "failed_samples": 0,
                "sample_logs": []
            }

            for ind, llm_output in enumerate(result):
                try:
                    sample_log = {
                        "index": ind,
                        "query": batch["query"][ind][:100] + "..." if len(batch["query"][ind]) > 100 else
                        batch["query"][ind],
                        "response": batch["response"][ind][:100] + "..." if len(batch["response"][ind]) > 100 else
                        batch["response"][ind],
                        "status": "processing"
                    }

                    if not isinstance(llm_output, (list, tuple)) or not llm_output:
                        print(f"Warning: Invalid llm_output format at index {ind}: {type(llm_output)}")
                        sample_log["status"] = "failed"
                        sample_log["error"] = "Invalid llm_output format"
                        batch_log["sample_logs"].append(sample_log)
                        fail += 1
                        continue

                    generated_text = llm_output[0].get('generated_text', '')
                    if not generated_text:
                        print(f"Warning: Empty generated_text at index {ind}")
                        sample_log["status"] = "failed"
                        sample_log["error"] = "Empty generated_text"
                        batch_log["sample_logs"].append(sample_log)
                        fail += 1
                        continue

                    text_to_parse = generated_text[len(input_datas[ind]):]
                    print(f"Processing output {ind}, text length: {len(text_to_parse)}")

                    # Log raw input and output to file
                    log_file = os.path.join(step_log_dir, f"sample_{ind}.txt")
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write("=" * 80 + "\n")
                        f.write(f"SAMPLE {ind} - EPOCH {epoch} - STEP {step}\n")
                        f.write("=" * 80 + "\n\n")

                        f.write("QUERY:\n")
                        f.write("-" * 80 + "\n")
                        f.write(batch["query"][ind] + "\n\n")

                        f.write("RESPONSE:\n")
                        f.write("-" * 80 + "\n")
                        f.write(batch["response"][ind] + "\n\n")

                        f.write("REWARD MODEL INPUT:\n")
                        f.write("-" * 80 + "\n")
                        f.write(input_datas[ind] + "\n\n")

                        f.write("REWARD MODEL RAW OUTPUT:\n")
                        f.write("-" * 80 + "\n")
                        f.write(generated_text + "\n\n")

                        f.write("TEXT TO PARSE:\n")
                        f.write("-" * 80 + "\n")
                        f.write(text_to_parse + "\n\n")

                    sample_log["log_file"] = log_file

                    if len(text_to_parse) < 20:  # Likely too short to contain valid data
                        print(f"Text too short, content: {text_to_parse}")
                        sample_log["status"] = "failed"
                        sample_log["error"] = "Text too short to parse"
                        batch_log["sample_logs"].append(sample_log)
                        fail += 1
                        continue

                    # Try JSON parsing first
                    print(f"Attempting JSON parsing for output {ind}")
                    json_result = load_json_from_string(text_to_parse, log_details=True)
                    score_list = []

                    if json_result and isinstance(json_result, dict):
                        print(f"JSON parsing successful, keys: {json_result.keys()}")
                        score_list = json_result.get("word_score_list", [])
                        print(f"Found {len(score_list)} items in word_score_list")

                        # Log parsing results
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write("JSON PARSING RESULT:\n")
                            f.write("-" * 80 + "\n")
                            f.write(f"Keys: {json_result.keys()}\n")
                            f.write(f"Word-score list length: {len(score_list)}\n")
                            if score_list:
                                f.write("Sample word-score pairs:\n")
                                for i, (word, score) in enumerate(score_list[:10]):
                                    f.write(f"  {i + 1}. '{word}': {score}\n")
                                if len(score_list) > 10:
                                    f.write(f"  ... and {len(score_list) - 10} more pairs\n")
                    else:
                        print(f"JSON parsing failed or returned non-dict: {type(json_result)}")

                        # Log parsing failure
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write("JSON PARSING FAILED\n")
                            f.write("-" * 80 + "\n")
                            f.write(f"Result type: {type(json_result)}\n\n")

                    # If JSON parsing didn't yield results, try direct extraction
                    if not score_list:
                        print(f"Attempting direct word-score extraction for output {ind}")
                        score_list = extract_word_scores_directly(text_to_parse, log_details=True)
                        print(f"Direct extraction found {len(score_list)} word-score pairs")

                        # Log direct extraction results
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write("DIRECT EXTRACTION RESULT:\n")
                            f.write("-" * 80 + "\n")
                            f.write(f"Word-score list length: {len(score_list)}\n")
                            if score_list:
                                f.write("Sample word-score pairs:\n")
                                for i, (word, score) in enumerate(score_list[:10]):
                                    f.write(f"  {i + 1}. '{word}': {score}\n")
                                if len(score_list) > 10:
                                    f.write(f"  ... and {len(score_list) - 10} more pairs\n")

                    print(f"Assigning word rewards for output {ind}")
                    reward, word = assign_word_rewards(batch['response'][ind], score_list)
                    print(f"Reward assignment result: {len(reward)} rewards, {len(word)} words")

                    # Log assignment results
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("REWARD ASSIGNMENT RESULT:\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"Rewards length: {len(reward)}\n")
                        f.write(f"Words length: {len(word)}\n")
                        if word and reward:
                            f.write("Sample word-reward pairs:\n")
                            for i in range(min(10, len(word))):
                                f.write(f"  {i + 1}. '{word[i]}': {float(reward[i])}\n")
                            if len(word) > 10:
                                f.write(f"  ... and {len(word) - 10} more pairs\n")

                    if reward and word:  # Only add when there are valid results
                        rewards.append(torch.tensor(reward))
                        words.append(word)
                        final_question_tensors.append(question_tensors[ind])
                        final_response_tensors.append(response_tensors[ind])
                        new_responses.append(batch['response'][ind])
                        new_questions.append(batch['query'][ind])
                        print(f"Successfully processed output {ind}")

                        sample_log["status"] = "success"
                        sample_log["rewards_count"] = len(reward)
                        sample_log["words_count"] = len(word)
                        batch_log["successful_samples"] += 1
                    else:
                        print(f"Failed to extract valid rewards/words for output {ind}")
                        fail += 1

                        sample_log["status"] = "failed"
                        sample_log["error"] = "No valid rewards/words extracted"
                        batch_log["failed_samples"] += 1

                    batch_log["sample_logs"].append(sample_log)

                except Exception as e:
                    print(f"Error processing output {ind}: {str(e)}")
                    print(f"Generated text sample: {generated_text[:100]}...")
                    traceback.print_exc()
                    fail += 1

                    sample_log = {
                        "index": ind,
                        "status": "error",
                        "error_message": str(e),
                        "traceback": traceback.format_exc()
                    }
                    batch_log["sample_logs"].append(sample_log)
                    batch_log["failed_samples"] += 1
                    continue

            print(f"Failed {fail} samples in current step! Successfully processed {len(rewards)} samples")

            # Save batch log
            batch_log_file = os.path.join(step_log_dir, "batch_summary.json")
            with open(batch_log_file, "w", encoding="utf-8") as f:
                json.dump(batch_log, f, indent=2, ensure_ascii=False)

            # Ensure enough valid samples to continue training
            if len(final_question_tensors) < script_args.batch_size // 2:
                print(f"Warning: Too few valid samples ({len(final_question_tensors)}), skipping step")
                continue

            batch["query"] = new_questions
            batch["response"] = new_responses

            try:
                stats, loss_ps, loss_vs, average_rewards = ppo_trainer.step(final_question_tensors,
                                                                            final_response_tensors, rewards, words,
                                                                            mask_loss=script_args.mask_loss)
                print(loss_ps)
                print(loss_vs)
                print(average_rewards.item())
                wandb.log({"train/loss_advantage": loss_ps}, step=wandb_step)
                wandb.log({"train/loss_value_kl": loss_vs}, step=wandb_step)
                wandb.log({"train/average_advantages": average_rewards.item()}, step=wandb_step)

                ppo_trainer.log_stats(wandb_step, stats, batch, rewards)

                if step != 0 and step % script_args.save_freq == 0:
                    try:
                        save_path = os.path.join(script_args.output_dir, f"epoch_{epoch}_step_{step}")
                        os.makedirs(save_path, exist_ok=True)

                        # Save the model with proper error handling
                        print(f"Saving model to {save_path}...")
                        ppo_trainer.save_pretrained(save_path)

                        # Save optimizer state with proper error handling
                        optimizer_path = os.path.join(save_path, "optimizer.pt")
                        print(f"Saving optimizer state to {optimizer_path}...")
                        torch.save({
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
                        }, optimizer_path)

                        print(f"Model and optimizer successfully saved to {save_path}")

                        # Verify the saved files exist
                        expected_files = ["adapter_model.safetensors", "pytorch_model.bin", "optimizer.pt"]
                        missing_files = [f for f in expected_files if not os.path.exists(os.path.join(save_path, f))]

                        if missing_files:
                            print(f"Warning: Some expected files are missing after save: {missing_files}")
                        else:
                            print("All expected model files were successfully saved")

                    except Exception as save_error:
                        print(f"Error during model saving: {str(save_error)}")
                        error_dir = os.path.join(script_args.output_dir, "save_error_logs")
                        os.makedirs(error_dir, exist_ok=True)
                        save_error_file = os.path.join(error_dir, f"save_error_epoch_{epoch}_step_{step}.txt")
                        with open(save_error_file, "w") as f:
                            f.write(f"Error saving model at epoch {epoch}, step {step}:\n")
                            f.write(str(save_error) + "\n\n")
                            f.write(traceback.format_exc())
                        print(f"Save error details written to {save_error_file}")

                wandb_step += 1
            except Exception as e:
                error_dir = os.path.join(script_args.output_dir, "error_logs")
                context = {
                    "epoch": epoch,
                    "step": step,
                    "batch_size": len(final_question_tensors) if 'final_question_tensors' in locals() else "N/A",
                    "wandb_step": wandb_step,
                    "tensor_shapes": {
                        "question_tensors": [t.shape for t in
                                             final_question_tensors] if 'final_question_tensors' in locals() else "N/A",
                        "response_tensors": [t.shape for t in
                                             final_response_tensors] if 'final_response_tensors' in locals() else "N/A",
                        "rewards": [r.shape for r in rewards] if 'rewards' in locals() else "N/A"
                    }
                }
                log_error(error_dir, step, e, context)
                print(f"Error logged to {error_dir}. See logs for details.")
                continue

        except Exception as e:
            print(f"Error in training step {step}: {str(e)}")
            traceback.print_exc()
            continue

