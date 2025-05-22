from dataclasses import dataclass, field
from typing import Optional
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
    get_scheduler
)
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import set_seed, LengthSampler
from text2grad_trainer import TEXT2GRADTrainer
from utils import load_json_from_string

import wandb
import re
import os
import json
from typing_extensions import Annotated
import datetime
import traceback
import difflib
import safetensors

JSONDict = Annotated[Optional[dict], tyro.conf.arg(metavar="JSON", constructor=json.loads)]

tqdm.pandas()

cur_epoch = 0
cur_step = 0
wandb_step = 0

class QACDataset(Dataset):
    def __init__(self, json_path, tokenizer, prompt_max_length, max_length):
        with open(json_path, "r") as json_f:
            self.data = json.load(json_f)
        self.prompt_max_length = prompt_max_length
        self.max_length = max_length
        self.answer_max_length = max_length - prompt_max_length - 1
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        item = self.data[ind]
        question = item["question"]
        response = item.get("response", "")

        chosen = item.get("response_j", [])
        rejected = item.get("response_k", [])

        query = question

        if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict) and "content" in response[0]:
            response = response[0]["content"]

        critique = item.get("critique", "")

        cur_example = self.build_dataset(question, response, query, critique, chosen, rejected)
        return cur_example

    def build_dataset(self, question, response, query, critique="", response_j=[], response_k=[]):
        def preprocess_function(question_text, response_text):
            new_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n" + \
                         question_text + "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

            input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(new_prompt)), dtype=torch.long)
            attention_mask = torch.ones(input_ids.shape)

            new_examples = {
                "question": question_text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "response": response_text,
                "query": query,
                "critique": critique,
                "response_j": response_j,
                "response_k": response_k
            }
            return new_examples

        new_examples = preprocess_function(question, response)
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
    tracker_kwargs: Optional[str] = field(default=None, metadata={"help": "tracker kwargs of wandb"})
    prompt_max_length: Optional[int] = field(default=1000, metadata={"help": "the length of prompt"})
    answer_max_length: Optional[int] = field(default=500, metadata={"help": "the length of answer"})
    kl_penalty: Optional[str] = field(default="full", metadata={"help": "way of kl penalty"})
    mask_loss: Optional[str] = field(default="", metadata={"help": "mask_loss"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

# Convert JSON string to dictionary
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

project_name = script_args.output_dir.split("/")[-1]
wandb.init(project=project_name, name="text2grad-ultrafeedback")

tokenizer = AutoTokenizer.from_pretrained(
    script_args.base_model_name,
    model_max_length=1600,
    use_fast=True,
    padding_side='right'
)

reward_tokenizer = AutoTokenizer.from_pretrained(
    script_args.reward_model_name,
    model_max_length=2500,
    use_fast=True,
    padding_side='left'
)  

if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

if not reward_tokenizer.pad_token:
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


ds = QACDataset(script_args.data_file_path, tokenizer, script_args.prompt_max_length, script_args.answer_max_length)
valid_ds = QACDataset(script_args.valid_data_file_path, tokenizer, script_args.prompt_max_length,
                      script_args.answer_max_length)
valid_dataloader = torch.utils.data.DataLoader(
    valid_ds,
    batch_size=15,
    collate_fn=collator,
    shuffle=True,
    drop_last=True,
)

set_seed(config.seed)

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
    device_map={"": current_device},
    peft_config=lora_config,
)

if script_args.base_model_adapter_model:
    print(f"Loading adapter model from {script_args.base_model_adapter_model}")
    try:
        adapter_model_state = safetensors.torch.load_file(
            os.path.join(script_args.base_model_adapter_model, "adapter_model.safetensors"))
        v_head = torch.load(os.path.join(script_args.base_model_adapter_model, "pytorch_model.bin"),
                            map_location="cpu")

        new_adapter_model_state = {
            "pretrained_model." + k[:-7] + ".default" + k[-7:]: v
            for k, v in adapter_model_state.items()
        }

        model_dict = model.state_dict()
        new_adapter_model_state.update(v_head)
        model_dict.update(new_adapter_model_state)

        model.load_state_dict(model_dict, strict=False)
        print("Successfully loaded adapter model and value head")

        optimizer_path = os.path.join(script_args.base_model_adapter_model, "optimizer.pt")
        if os.path.exists(optimizer_path):
            try:
                optimizer_state = torch.load(optimizer_path, map_location="cuda:1", weights_only=True)
                if not optimizer_state:
                    raise ValueError("Empty optimizer state")
                optimizer.load_state_dict(optimizer_state['optimizer'])
                if optimizer_state.get('lr_scheduler') and lr_scheduler:
                    lr_scheduler.load_state_dict(optimizer_state['lr_scheduler'])
                print("Successfully loaded optimizer state")
            except Exception as e:
                print(f"Error loading optimizer state: {str(e)}")
                print("Continuing with fresh optimizer")
        else:
            print("No optimizer state found, starting with fresh optimizer")

        try:
            checkpoint_path = script_args.base_model_adapter_model
            path_parts = checkpoint_path.split('/')[-1].split('_')

            cur_epoch = int(path_parts[path_parts.index('epoch') + 1])
            cur_step = int(path_parts[path_parts.index('step') + 1])

            batch_size = script_args.batch_size
            num_processes = torch.cuda.device_count() 
            steps_per_epoch = len(ds) // (batch_size * num_processes)
            if len(ds) % (batch_size * num_processes) != 0:
                steps_per_epoch += 1

            wandb_step = cur_epoch * steps_per_epoch + cur_step
            print(f"Restored training state: epoch={cur_epoch}, step={cur_step}, wandb_step={wandb_step}")

            # 打印详细信息用于调试
            print(f"Debug info:")
            print(f"Dataset size: {len(ds)}")
            print(f"Batch size: {batch_size}")
            print(f"Number of processes: {num_processes}")
            print(f"Steps per epoch: {steps_per_epoch}")

        except Exception as e:
            print(f"Error extracting epoch/step from checkpoint path: {str(e)}")
            print("Starting from epoch 0, step 0")
            cur_epoch = 0
            cur_step = 0
            wandb_step = 0

    except Exception as e:
        print(f"Error loading adapter model: {str(e)}")
        traceback.print_exc()
        raise RuntimeError("Failed to load adapter model")

for param in model.v_head.parameters():
    param.requires_grad = True  # 优化vhead

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
ds_config_path = os.environ.get('DEEPSPEED_CONFIG_FILE', '/path/to/your/ds_config.json')
if os.path.exists(ds_config_path):
    with open(ds_config_path, 'r') as f:
        ds_config = json.load(f)
        print(f"DeepSpeed config: {json.dumps(ds_config, indent=2)}")
        if 'optimizer' in ds_config:
            print("WARNING: Optimizer is still defined in DeepSpeed config!")

# Then create your trainer
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
    device = 0 if torch.cuda.is_available() else "cpu"

from accelerate.utils import infer_auto_device_map

model = AutoModelForCausalLM.from_pretrained(
    script_args.reward_model_name,
    torch_dtype="float16",
)

reward_model = pipeline("text-generation",
                        model=AutoModelForCausalLM.from_pretrained(
                            script_args.reward_model_name,
                            torch_dtype="float16",
                            device_map={"": current_device}
                        ),
                        tokenizer=reward_tokenizer
                        )

# for text2grad trainer
generation_kwargs = {
    "temperature": 0.6,
    "do_sample": True,
    "top_p": 1,
    "top_k": 0,
    "max_new_tokens": 512,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

# for reward model
sent_kwargs = {
    "batch_size": script_args.batch_size,
    "do_sample": True,
    "temperature": 0.1,
    "min_new_tokens": 50,
    "max_new_tokens": 768,
    "pad_token_id": reward_tokenizer.eos_token_id,
    "eos_token_id": reward_tokenizer.eos_token_id,
}

def check_and_fix_tensor(tensor, eos_id):
    # Ensure input is a 1D Tensor
    if len(tensor.shape) != 1:
        raise ValueError("Input tensor must be 1D.")

    # Check if there are multiple eos_id tokens at the end
    while len(tensor) > 1 and tensor[-1] == eos_id and tensor[-2] == eos_id:
        tensor = tensor[:-1]  # Remove redundant eos_id tokens

    # If there's no eos_id at the end, add one
    if tensor[-1] != eos_id or len(tensor) == 0:
        tensor = torch.cat([tensor, torch.tensor([eos_id], dtype=tensor.dtype, device=tensor.device)])

    return tensor

def prepare_input_data(queries, responses):
    """
    Prepares input data with direct user prompt format
    Args:
        queries: List of original user inputs
        responses: List of generated responses
    Returns:
        List of formatted prompts
    """
    input_datas = []
    for user_prompt, assistant_response in zip(queries, responses):
        prompt = f'''Please critique the following response to a user input and provide feedback and word-level score list:

---
**User Input**:
{user_prompt}
---

---
**Generated Response**:
{assistant_response}
---

---
**Definitions:**
- **good_spans**: phrases from the response that greatly improve its quality by accurately addressing the user input, providing key information, or capturing the core intent effectively, as explained in 'textual_feedback'. Empty if none apply.
- **poor_spans**: phrases from the response that noticeably harm its quality due to inaccuracy, irrelevance, redundancy, poor wording, or missing critical aspects of the input, as explained in 'textual_feedback'. Empty if none apply.

---
**Instructions:**
1. Evaluate the response based on:
    - **Accuracy**: Does it correctly address the input?
    - **Relevance**: Does it stay on topic?
    - **Clarity**: Is it easy to understand?
    - **Completeness**: Does it cover the input's core needs?
2. Select the most significant phrases for 'good_spans' and 'poor_spans', keeping them impactful, and essence-focused, with brief justifications. Include none if no phrases stand out.
3. Ensure 'good_spans' and 'poor_spans' are directly supported by the analysis in 'textual_feedback'.

---
**Output Format:**
{{
"textual_feedback": "Your critique here summarizing key strengths and weaknesses in one paragraph.",
"good_spans": ["phrase1", "phrase2",...],
"poor_spans": ["phrase1", "phrase2",...]
}}'''

        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        input_datas.append(formatted_prompt)

    return input_datas


def normalize_text(text):
    if not text:
        return ""

    # 移除多余空格
    text = re.sub(r'[,.;:!?"\'\(\)\[\]\{\}]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fuzzy_find(text, pattern, threshold=0.8):
    """
    Use fuzzy matching to find a pattern in text, employing multiple strategies to improve match rate
    
    Args:
        text: The text to search in
        pattern: The pattern to find
        threshold: Matching threshold between 0-1, higher means stricter matching
        
    Returns:
        tuple: (start_index, end_index) if a match is found, otherwise (-1, -1)
    """
    if not text or not pattern:
        return -1, -1

    min_length = 5
    adjusted_threshold = threshold
    if len(pattern) < min_length:
        adjusted_threshold = max(0.5, threshold - 0.2)

    exact_match = text.find(pattern)
    if exact_match >= 0:
        return exact_match, exact_match + len(pattern)

    norm_text = normalize_text(text)
    norm_pattern = normalize_text(pattern)

    if norm_pattern and len(norm_pattern) > 3:
        norm_match = norm_text.find(norm_pattern)
        if norm_match >= 0:
            words_before = len(norm_text[:norm_match].split())
            original_words = text.split()
            if words_before < len(original_words):
                approx_start = sum(len(w) + 1 for w in original_words[:words_before])
                return approx_start, approx_start + len(pattern)

    chunks = []
    step_size = max(1, len(pattern) // 4) 

    for i in range(0, len(text) - len(pattern) + 1, step_size):
        chunk_size = min(len(pattern) * 2, len(text) - i)
        chunks.append((i, text[i:i+chunk_size]))

    if not chunks and text:
        chunks.append((0, text))

    best_ratio = 0
    best_start = -1
    best_end = -1

    for start_idx, chunk in chunks:
        matcher = difflib.SequenceMatcher(None, chunk, pattern)
        ratio = matcher.ratio()

        matching_blocks = matcher.get_matching_blocks()
        if matching_blocks:
            longest_block = max(matching_blocks, key=lambda x: x[2])
            if longest_block[2] > len(pattern) * 0.7:  # If more than 70% matches
                ratio = max(ratio, 0.7)

        if ratio > best_ratio:
            best_ratio = ratio
            best_start = start_idx
            best_end = start_idx + len(pattern)

    if best_ratio >= adjusted_threshold:
        return best_start, best_end

    if len(pattern) > 10:
        front_part = pattern[:len(pattern)//2]
        front_start, front_end = fuzzy_find(text, front_part, threshold - 0.1)
        if front_start >= 0:
            return front_start, front_start + len(pattern)

        back_part = pattern[len(pattern)//2:]
        back_start, back_end = fuzzy_find(text, back_part, threshold - 0.1)
        if back_start >= 0:
            approx_start = max(0, back_start - len(pattern)//2)
            return approx_start, approx_start + len(pattern)

    return -1, -1

def clean_spans(spans):
    if not spans:
        return []

    if isinstance(spans, list):
        cleaned_spans = []
        for span in spans:
            if not span:
                continue
            cleaned = span.replace('\\"', '"')
            cleaned = cleaned.strip()
            if cleaned:
                cleaned_spans.append(cleaned)
        return cleaned_spans
    return spans

def check_spans_in_response(response, good_spans, poor_spans, fuzzy_threshold=0.8):
    """
    Check if good/poor spans can be found in the response using improved matching algorithms
    
    Args:
        response: Assistant's text response
        good_spans: List of text spans marked as good
        poor_spans: List of text spans marked as poor
        fuzzy_threshold: Threshold for fuzzy matching
        
    Returns:
        tuple: (matching success flag, list of unmatched spans, dictionary of matched spans)
    """
    if not response:
        return False, [], {}

    good_spans = clean_spans(good_spans)
    poor_spans = clean_spans(poor_spans)

    unmatched_spans = []
    matched_spans = {}

    for span in good_spans:
        try:
            if not span or len(span.strip()) == 0:
                continue

            span_start = response.find(span)
            span_end = -1

            if span_start < 0:
                thresholds = [fuzzy_threshold, fuzzy_threshold - 0.1, fuzzy_threshold - 0.2]
                for threshold in thresholds:
                    span_start, span_end = fuzzy_find(response, span, threshold)
                    if span_start >= 0:
                        break

            if span_start < 0:
                words = span.split()
                if len(words) > 5:
                    half_point = len(words) // 2
                    front_part = ' '.join(words[:half_point])
                    front_start = response.find(front_part)
                    if front_start >= 0:
                        span_start = front_start
                        span_end = front_start + len(span)
                    else:
                        back_part = ' '.join(words[half_point:])
                        back_start = response.find(back_part)
                        if back_start >= 0:
                            span_start = max(0, back_start - len(' '.join(words[:half_point])))
                            span_end = back_start + len(back_part)

            if span_start >= 0:
                if span_end < 0:
                    span_end = span_start + len(span)
                matched_spans[span] = (span_start, span_end, 1)
            else:
                unmatched_spans.append(f"good_span: {span}")
        except Exception as e:
            unmatched_spans.append(f"good_span_error: {span}")

    for span in poor_spans:
        try:
            if not span or len(span.strip()) == 0:
                continue

            span_start = response.find(span)
            span_end = -1

            if span_start < 0:
                thresholds = [fuzzy_threshold, fuzzy_threshold - 0.1, fuzzy_threshold - 0.2]
                for threshold in thresholds:
                    span_start, span_end = fuzzy_find(response, span, threshold)
                    if span_start >= 0:
                        break

            if span_start < 0:
                words = span.split()
                if len(words) > 5:
                    half_point = len(words) // 2
                    front_part = ' '.join(words[:half_point])
                    front_start = response.find(front_part)
                    if front_start >= 0:
                        span_start = front_start
                        span_end = front_start + len(span)
                    else:
                        back_part = ' '.join(words[half_point:])
                        back_start = response.find(back_part)
                        if back_start >= 0:
                            span_start = max(0, back_start - len(' '.join(words[:half_point])))
                            span_end = back_start + len(back_part)

            if span_start >= 0:
                if span_end < 0:
                    span_end = span_start + len(span)
                matched_spans[span] = (span_start, span_end, -1)
            else:
                unmatched_spans.append(f"poor_span: {span}")
        except Exception as e:
            unmatched_spans.append(f"poor_span_error: {span}")

    all_matched = len(unmatched_spans) == 0 and (len(good_spans) > 0 or len(poor_spans) > 0)
    return all_matched, unmatched_spans, matched_spans

def process_response_with_spans(response, good_spans, poor_spans):
    """
    Process single response and its good/poor spans, generate word score list

    Args:
        response: Assistant's text response
        good_spans: List of text spans marked as good
        poor_spans: List of text spans marked as poor

    Returns:
        word_score_list: (word, score) tuple list, where good spans score 1, poor spans score -1, neutral 0
    """
    if not response:
        return []

    # Clean spans
    good_spans = clean_spans(good_spans)
    poor_spans = clean_spans(poor_spans)

    # Check spans in response
    all_matched, unmatched_spans, matched_spans = check_spans_in_response(
        response, good_spans, poor_spans, fuzzy_threshold=0.7
    )

    if not all_matched:
        print(f"Warning: Some spans are not matched: {unmatched_spans}")
        if not matched_spans:
            words = response.split()
            return [(word, 0.1) for word in words]
        print(f"Continuing with {len(matched_spans)} matched spans")

    # Split response into words
    words = response.split()
    word_scores = [0.0] * len(words)

    # Assign scores to each word
    for i in range(len(words)):
        word_pos = sum(len(w) + 1 for w in words[:i])

        # Check if the position is in any matched span
        for span, (start, end, score) in matched_spans.items():
            if start <= word_pos < end:
                word_scores[i] = score
                break

    # Create final word-score list
    word_score_list = [(word, score) for word, score in zip(words, word_scores)]

    return word_score_list

def extract_spans_from_reward_model_output(text_to_parse):
    """
    Extract textual feedback and span information from reward model output
    
    Args:
        text_to_parse: Text to parse
        
    Returns:
        dict: Dictionary containing textual_feedback, good_spans, and poor_spans; returns None if parsing fails
    """
    try:
        # Try direct JSON parsing
        json_result = load_json_from_string(text_to_parse, log_details=False)

        if json_result and isinstance(json_result, dict):
            # Check if it contains the required keys
            if "textual_feedback" in json_result:
                # Ensure good_spans and poor_spans are lists
                good_spans = json_result.get("good_spans", [])
                poor_spans = json_result.get("poor_spans", [])

                if not isinstance(good_spans, list):
                    good_spans = []
                if not isinstance(poor_spans, list):
                    poor_spans = []

                return {
                    "textual_feedback": json_result["textual_feedback"],
                    "good_spans": good_spans,
                    "poor_spans": poor_spans
                }

        # If direct parsing fails, try using regular expressions
        textual_feedback_pattern = r'"textual_feedback"\s*:\s*"([^"]*)"'
        good_spans_pattern = r'"good_spans"\s*:\s*\[(.*?)\]'
        poor_spans_pattern = r'"poor_spans"\s*:\s*\[(.*?)\]'

        textual_feedback_match = re.search(textual_feedback_pattern, text_to_parse)
        good_spans_match = re.search(good_spans_pattern, text_to_parse)
        poor_spans_match = re.search(poor_spans_pattern, text_to_parse)

        textual_feedback = textual_feedback_match.group(1) if textual_feedback_match else ""

        good_spans = []
        if good_spans_match:
            spans_text = good_spans_match.group(1)
            good_spans = re.findall(r'"([^"]*)"', spans_text)

        poor_spans = []
        if poor_spans_match:
            spans_text = poor_spans_match.group(1)
            poor_spans = re.findall(r'"([^"]*)"', spans_text)

        return {
            "textual_feedback": textual_feedback,
            "good_spans": good_spans,
            "poor_spans": poor_spans
        }

    except Exception as e:
        print(f"Error parsing reward model output: {str(e)}")
        traceback.print_exc()
        return None

def inference_reward(reward_model, input_datas):
    """
    Run inference with the reward model with enhanced error handling
    
    Args:
        reward_model: The reward model
        input_datas: List of input data
        
    Returns:
        list: List of processed results
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
        return [[{'generated_text': ''}] for _ in input_datas]]


def log_error(error_dir, step, error, context=None):
    """Helper function to log errors with detailed information"""
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


def load_json_from_string(text, log_details=True):
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
                for i, match in enumerate(matches[:2]): 
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

        if log_details:
            print("Attempting direct JSON loading")
        return json.loads(text)
    except Exception as e:
        if log_details:
            print(f"Error parsing JSON: {str(e)}")
            print(f"Problematic text (first 200 chars): {text[:200]}...")
            print(f"Problematic text (last 200 chars): {text[-200:] if len(text) > 200 else text}")

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

def calculate_similarity(text1, text2):
    """
    Calculate similarity between two texts using both ROUGE and sequence matcher

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        float: Similarity score between 0 and 1
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = scorer.score(text1, text2)
    rouge_score = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3

    seq_matcher = difflib.SequenceMatcher(None, text1, text2)
    seq_score = seq_matcher.ratio()

    final_score = (rouge_score + seq_score) / 2

    return final_score

def evaluate_response_accuracy(response, chosen_content, rejected_content):
    """
    Evaluate response accuracy by comparing similarity with chosen and rejected content

    Args:
        response: Model generated response
        chosen_content: List of chosen messages or single chosen content
        rejected_content: List of rejected messages or single rejected content

    Returns:
        float: Binary accuracy score (0 or 1)
    """
    try:
        if isinstance(chosen_content, list):
            chosen_text = ""
            for msg in chosen_content:
                if isinstance(msg, dict) and msg.get('role') == 'assistant':
                    chosen_text = msg.get('content', '')
                    break
                elif isinstance(msg, str):
                    chosen_text = msg
                    break
        else:
            chosen_text = chosen_content

        if isinstance(rejected_content, list):
            rejected_text = ""
            for msg in rejected_content:
                if isinstance(msg, dict) and msg.get('role') == 'assistant':
                    rejected_text = msg.get('content', '')
                    break
                elif isinstance(msg, str):
                    rejected_text = msg
                    break
        else:
            rejected_text = rejected_content

        if chosen_text and rejected_text:
            chosen_similarity = calculate_similarity(response, chosen_text)
            rejected_similarity = calculate_similarity(response, rejected_text)

            return 1.0 if chosen_similarity > rejected_similarity else 0.0
        else:
            print("Warning: Missing chosen or rejected content for comparison")
            return 0.0

    except Exception as e:
        print(f"Error in evaluate_response_accuracy: {str(e)}")
        traceback.print_exc()
        return 0.0  
for epoch in range(script_args.train_epochs):
    if epoch < cur_epoch:
        continue

    epoch_log_dir = os.path.join(script_args.output_dir, f"logs/epoch_{epoch}")
    os.makedirs(epoch_log_dir, exist_ok=True)

    for step, batch in tqdm(enumerate(ppo_trainer.dataloader), desc=f"Epoch {epoch + 1} "):
        try:
            os.environ["NCCL_P2P_DISABLE"] = "1"
            os.environ["NCCL_P2P_LEVEL"] = "NVL"

            if epoch == cur_epoch and step <= cur_step:
                continue

            step_log_dir = os.path.join(epoch_log_dir, f"step_{step}")
            os.makedirs(step_log_dir, exist_ok=True)

            question_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(
                batch["input_ids"],
                return_prompt=False,
                **generation_kwargs,
            )

            for ind in range(len(response_tensors)):
                response_tensors[ind] = check_and_fix_tensor(response_tensors[ind], tokenizer.eos_token_id)
            
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            batch["response"] = [text.strip("assistant") for text in batch["response"]]
            print(batch["response"][0])

            input_datas = prepare_input_data(batch["query"], batch["response"])
            result = inference_reward(reward_model, input_datas)
            
            rewards = []
            words = []
            final_question_tensors = []
            final_response_tensors = []
            new_responses = []
            new_questions = []
            fail = 0

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

                    if len(text_to_parse) < 20:
                        print(f"Text too short, content: {text_to_parse}")
                        sample_log["status"] = "failed"
                        sample_log["error"] = "Text too short to parse"
                        batch_log["sample_logs"].append(sample_log)
                        fail += 1
                        continue

                    parsed_result = extract_spans_from_reward_model_output(text_to_parse)
                    if not parsed_result:
                        print(f"Failed to parse reward model output for sample {ind}")
                        sample_log["status"] = "failed"
                        sample_log["error"] = "Failed to parse reward model output"
                        batch_log["sample_logs"].append(sample_log)
                        fail += 1
                        continue

                    score_list = process_response_with_spans(
                        batch['response'][ind],
                        parsed_result['good_spans'],
                        parsed_result['poor_spans']
                    )

                    if score_list:
                        word_rewards = [score for _, score in score_list]
                        word_list = [word for word, _ in score_list]

                        rewards.append(torch.tensor(word_rewards))
                        words.append(word_list)
                        final_question_tensors.append(question_tensors[ind])
                        final_response_tensors.append(response_tensors[ind])
                        new_responses.append(batch['response'][ind])
                        new_questions.append(batch['query'][ind])
                        print(f"Successfully processed output {ind}")

                        sample_log["status"] = "success"
                        sample_log["rewards_count"] = len(word_rewards)
                        sample_log["words_count"] = len(word_list)
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

            print(f"Failed {fail} samples in the current step!!! Successfully processed {len(rewards)} samples")

            valid_samples = len(final_question_tensors)

            if ppo_trainer.accelerator.num_processes > 1:
                valid_samples_tensor = torch.tensor(valid_samples, device=ppo_trainer.accelerator.device)

                all_valid_samples = [torch.zeros_like(valid_samples_tensor) for _ in range(ppo_trainer.accelerator.num_processes)]
                torch.distributed.all_gather(all_valid_samples, valid_samples_tensor)

                min_valid_samples = min([count.item() for count in all_valid_samples])

                if min_valid_samples < 8:
                    print(f"Warning: Some processes have fewer than 8 valid samples (min: {min_valid_samples}), skipping step")
                    continue

            elif valid_samples < 8:
                print(f"Warning: Too few valid samples ({valid_samples}), skipping step")
                continue

            batch["query"] = new_questions
            batch["response"] = new_responses

            try:
                stats, loss_ps, loss_vs, average_rewards = ppo_trainer.step(
                    final_question_tensors,
                    final_response_tensors, 
                    rewards, 
                    words,
                    mask_loss=script_args.mask_loss
                )

                wandb.log({"train/loss_advantage": loss_ps}, step=wandb_step)
                wandb.log({"train/loss_value_kl": loss_vs}, step=wandb_step)
                wandb.log({"train/average_advantages": average_rewards.item()}, step=wandb_step)

                if rewards:
                    all_rewards = [r.tolist() for r in rewards]
                    flat_rewards = [item for sublist in all_rewards for item in sublist]

                    wandb.log({
                        "rewards/mean": sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0,
                        "rewards/max": max(flat_rewards) if flat_rewards else 0,
                        "rewards/min": min(flat_rewards) if flat_rewards else 0,
                        "rewards/positive_ratio": sum(1 for r in flat_rewards if r > 0) / len(flat_rewards) if flat_rewards else 0,
                        "rewards/negative_ratio": sum(1 for r in flat_rewards if r < 0) / len(flat_rewards) if flat_rewards else 0,
                        "rewards/zero_ratio": sum(1 for r in flat_rewards if r == 0) / len(flat_rewards) if flat_rewards else 0,
                        "rewards/sample_count": len(flat_rewards)
                    }, step=wandb_step)

                    wandb.log({"rewards/distribution": wandb.Histogram(flat_rewards)}, step=wandb_step)

                    examples_to_log = min(3, len(rewards))
                    for i in range(examples_to_log):
                        token_table = wandb.Table(columns=["token", "reward"])
                        for token, reward in zip(words[i], rewards[i].tolist()):
                            token_table.add_data(token, reward)

                        wandb.log({
                            f"examples/example_{i+1}/token_rewards": token_table,
                            f"examples/example_{i+1}/query": batch["query"][i],
                            f"examples/example_{i+1}/response": batch["response"][i],
                            f"examples/example_{i+1}/avg_reward": sum(rewards[i].tolist()) / len(rewards[i]) if len(rewards[i]) > 0 else 0
                        }, step=wandb_step)

                ppo_trainer.log_stats(wandb_step, stats, batch, rewards)

                if step != 0 and step % script_args.save_freq == 0:
                    try:
                        if ppo_trainer.accelerator.is_main_process:
                            save_path = os.path.join(script_args.output_dir, f"epoch_{epoch}_step_{step}")
                            os.makedirs(save_path, exist_ok=True)

                            print(f"Saving model to {save_path}...")
                            ppo_trainer.save_pretrained(save_path)

                            optimizer_path = os.path.join(save_path, "optimizer.pt")
                            torch.save({
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
                            }, optimizer_path)

                            expected_files = ["adapter_model.safetensors", "pytorch_model.bin", "optimizer.pt"]
                            missing_files = [f for f in expected_files if not os.path.exists(os.path.join(save_path, f))]

                            if missing_files:
                                print(f"Warning: Some expected files are missing after save: {missing_files}")
                            else:
                                print("All expected model files were successfully saved")
                        else:
                            print("Skipping model save on non-main process")
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

                if step != 0 and step % 1 == 0:
                    print("Evaluating responses with similarity comparison...")
                    eval_size = min(20, len(batch["query"]))
                    accuracies = []

                    for i in range(eval_size):
                        try:
                            response = batch["response"][i]
                            chosen_content = batch["response_j"][i]
                            rejected_content = batch["response_k"][i]

                            accuracy = evaluate_response_accuracy(response, chosen_content, rejected_content)
                            accuracies.append(accuracy)

                        except Exception as e:
                            print(f"Error evaluating sample {i}: {str(e)}")
                            accuracies.append(0.0)

                    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

                    wandb.log({
                        "eval/mean_accuracy": mean_accuracy,
                        "eval/accuracies": accuracies
                    }, step=wandb_step)

                    print(f"Similarity evaluation complete. Mean accuracy: {mean_accuracy}")

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

