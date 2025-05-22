from dataclasses import dataclass, field
from typing import Optional
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline, AutoModelForCausalLM
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import set_seed
from transformers import get_scheduler
from text2grad_trainer import Text2GradTrainer
from utils import load_json_from_string

import wandb
import re
import os
import tyro
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
        response = item.get("solution", "")

        # Keep this check in case 'solution' could also be a list of dicts in some cases
        if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict) and "content" in response[0]:
            response = response[0]["content"]

        # Call build_dataset without the removed fields
        cur_example = self.build_dataset(question, response)
        return cur_example

    def build_dataset(self, question, response):
        def preprocess_function(question_text, response_text):
            # Using the same prompt format as before
            new_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n" + \
                         question_text + "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

            input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(new_prompt)), dtype=torch.long)
            attention_mask = torch.ones(input_ids.shape)

            new_examples = {
                "question": question_text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "response": response_text # This now holds the content from 'solution'
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
wandb.init(project=project_name, name="Text2Grad-KodCode")

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
    torch_dtype=torch.bfloat16
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
            print(f"Loading optimizer state from {optimizer_path}")
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
    param.requires_grad = True  # Optimize vhead

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
        # Check if optimizer is in the config
        if 'optimizer' in ds_config:
            print("WARNING: Optimizer is still defined in DeepSpeed config!")

# Then create your trainer
ppo_trainer = Text2GradTrainer(
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
    torch_dtype=torch.bfloat16,
)

reward_model = pipeline("text-generation",
                        model=AutoModelForCausalLM.from_pretrained(
                            script_args.reward_model_name,
                            torch_dtype="float16",
                            device_map={"": current_device}
                        ),
                        tokenizer=reward_tokenizer
                        )

# for ppo trainer
generation_kwargs = {
    "temperature": 0.6,
    "do_sample": True,
    "top_p": 1,
    "top_k": 0,
    "max_new_tokens": 256,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

# for reward model
sent_kwargs = {
    "batch_size": script_args.batch_size,
    "max_new_tokens": 768,
    "pad_token_id": reward_tokenizer.eos_token_id,
    "eos_token_id": reward_tokenizer.eos_token_id,
}

def check_and_fix_tensor(tensor, eos_id):
    if len(tensor.shape) != 1:
        raise ValueError("Input tensor must be 1D.")

    # Find if there are multiple eos_id at the end
    while len(tensor) > 1 and tensor[-1] == eos_id and tensor[-2] == eos_id:
        tensor = tensor[:-1]  # Remove extra eos_id

    # If there is no eos_id at the end, add one
    if tensor[-1] != eos_id or len(tensor) == 0:
        tensor = torch.cat([tensor, torch.tensor([eos_id], dtype=tensor.dtype, device=tensor.device)])

    return tensor



def prepare_input_data(questions, responses):
    """
    Prepares input data by extracting code blocks from responses
    Args:
        questions: List of original user inputs (questions)
        responses: List of generated responses
    Returns:
        List of formatted prompts with extracted code
    """
    input_datas = []
    for user_prompt, assistant_response in zip(questions, responses):
        code_blocks = re.findall(r'```(?:python|py)?\n(.*?)```', assistant_response, re.DOTALL)

        if not code_blocks:
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', assistant_response, re.DOTALL)

        if code_blocks:
            code_content = '\n\n'.join(block.strip() for block in code_blocks)
        else:
            code_content = assistant_response.strip()

        code_content = code_content.replace('\\n', '\n')
        prompt = f'''Please analyze the following programming problem and solution:

Problem:
{user_prompt}

Submitted Solution:
{code_content}

---
**Instructions:**
1. Evaluate the code solution based on:
   - **Correctness**: Does it solve the problem without bugs?
   - **Efficiency**: Is the time and space complexity optimal?
   - **Readability**: Is the code clean and well-structured?
   - **Completeness**: Does it handle all edge cases?
2. Identify specific code snippets that:
   - Have errors or bugs (for wrong_code)
   - Work correctly but could be improved (for improvement_code)
3. Provide a concise paragraph summarizing the overall quality of the solution.

---
**Output Format:**
```json
{{
    "code_feedback": "Evaluate the strengths and weaknesses (if any) of the code solution, concisely written in one paragraph.",
    "wrong_code": ["Include only code snippets with errors or bugs causing test failures. Leave as an empty array if no such errors are found."],
    "improvement_code": ["Include only original code snippets in the Submitted Code that work correctly but could be improved. Leave as an empty array if no improvements are needed."]
}}
```
'''
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        input_datas.append(formatted_prompt)

    return input_datas


def normalize_code_whitespace(text):
    """Normalizes whitespace in code snippets for better matching."""
    if not text:
        return ""
    # Replace multiple whitespace chars (space, tab, newline) with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    return text.strip()


def fuzzy_find(text, pattern, threshold=0.8):
    """
    Use fuzzy matching with whitespace normalization.
    """
    if not text or not pattern:
        return -1, -1

    # Normalize the text and pattern for comparison
    norm_text_full = normalize_code_whitespace(text)
    norm_pattern = normalize_code_whitespace(pattern)

    if not norm_pattern: # Don't match empty patterns
        return -1, -1

    exact_match_start = text.find(pattern)
    if exact_match_start != -1:
        return exact_match_start, exact_match_start + len(pattern)

    if len(norm_pattern) < 20:
        words_in_pattern = norm_pattern.split()
        if words_in_pattern:
            anchor = ' '.join(words_in_pattern[:min(3, len(words_in_pattern))])
            anchor_pos = norm_text_full.find(anchor)
            if anchor_pos >= 0:
                approx_start = text.find(anchor, max(0, anchor_pos - 20))
                if approx_start >= 0:
                    approx_end = min(len(text), approx_start + len(pattern) * 1.2)
                    return approx_start, int(approx_end)

    best_ratio = 0
    best_match_info = None

    pattern_len = len(pattern)
    step_size = max(1, pattern_len // 8)

    for i in range(0, len(text) - min(len(text), pattern_len) + 1, step_size):
        chunk_end = min(len(text), i + int(pattern_len * 2.0))
        original_chunk = text[i:chunk_end]

        norm_chunk = normalize_code_whitespace(original_chunk)

        if not norm_chunk: continue

        matcher = difflib.SequenceMatcher(None, norm_chunk, norm_pattern, autojunk=False)
        ratio = matcher.ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            match = matcher.find_longest_match(0, len(norm_chunk), 0, len(norm_pattern))
            if match.size > 0:
                matched_text_in_norm = norm_chunk[match.a:match.a + match.size]
                chunk_start_in_original = i
                match_start_in_original = chunk_start_in_original

                for j in range(len(original_chunk) - match.size + 1):
                    if normalize_code_whitespace(original_chunk[j:j+match.size]) == matched_text_in_norm:
                        match_start_in_original = chunk_start_in_original + j
                        break

                match_end_in_original = match_start_in_original + len(pattern)
                best_match_info = (match_start_in_original, match_end_in_original, ratio)

    if best_match_info and best_match_info[2] >= threshold:
        return best_match_info[0], best_match_info[1]

    if best_match_info and best_match_info[2] >= threshold * 0.8:
        print(f"Using lower threshold match: Ratio={best_match_info[2]:.2f}, Pattern='{pattern[:50]}...'")
        return best_match_info[0], best_match_info[1]

    return -1, -1


def process_response_with_spans(response, wrong_code, improvement_code):
    """
    Process response and code snippets to generate word score list, keeping non-code parts score as 1
    Using block processing for efficiency
    """
    if not response:
        return []

    original_wrong_code = [str(s).strip() for s in wrong_code if s and isinstance(s, str)]
    original_improvement_code = [str(s).strip() for s in improvement_code if s and isinstance(s, str)]

    # Find all matching spans
    matched_spans = []

    for span_text in original_wrong_code:
        if len(span_text) < 2:
            print(f"[SKIPPED - TOO SHORT] Wrong code span: '{span_text}'")
            continue

        start, end = fuzzy_find(response, span_text, threshold=0.75)
        if start >= 0:
            matched_spans.append((start, end, -1))
            print(f"[MATCHED - WRONG] Span: '{span_text[:50]}...' found at ({start}:{end})")
        else:
            words = span_text.split()
            if len(words) > 3:
                partial_span = ' '.join(words[:min(10, len(words))])
                start, end = fuzzy_find(response, partial_span, threshold=0.7)
                if start >= 0:
                    estimated_end = min(len(response), start + len(span_text))
                    matched_spans.append((start, estimated_end, -1))
                    print(f"[PARTIAL MATCH - WRONG] Span: '{partial_span[:50]}...' found at ({start}:{estimated_end})")
                else:
                    print(f"[NOT MATCHED - WRONG] Span: '{span_text[:50]}...'")
            else:
                print(f"[NOT MATCHED - WRONG] Span: '{span_text[:50]}...'")

    for span_text in original_improvement_code:
        if len(span_text) < 2:
            print(f"[SKIPPED - TOO SHORT] Improvement code span: '{span_text}'")
            continue

        start, end = fuzzy_find(response, span_text, threshold=0.75)
        if start >= 0:
            matched_spans.append((start, end, 0.0))
            print(f"[MATCHED - IMPROVE] Span: '{span_text[:50]}...' found at ({start}:{end})")
        else:
            words = span_text.split()
            if len(words) > 3:
                partial_span = ' '.join(words[:min(10, len(words))])
                start, end = fuzzy_find(response, partial_span, threshold=0.7)
                if start >= 0:
                    estimated_end = min(len(response), start + len(span_text))
                    matched_spans.append((start, estimated_end, 0.0))
                    print(f"[PARTIAL MATCH - IMPROVE] Span: '{partial_span[:50]}...' found at ({start}:{estimated_end})")
                else:
                    print(f"[NOT MATCHED - IMPROVE] Span: '{span_text[:50]}...'")
            else:
                print(f"[NOT MATCHED - IMPROVE] Span: '{span_text[:50]}...'")

    matched_spans.sort(key=lambda x: x[0])

    merged_spans = []
    if matched_spans:
        current_span = matched_spans[0]
        for next_span in matched_spans[1:]:
            if current_span[1] >= next_span[0]:
                new_end = max(current_span[1], next_span[1])
                new_score = min(current_span[2], next_span[2])
                current_span = (current_span[0], new_end, new_score)
            else:
                merged_spans.append(current_span)
                current_span = next_span
        merged_spans.append(current_span)

    blocks = []
    current_pos = 0

    for start, end, score in merged_spans:
        if start > current_pos:
            blocks.append((current_pos, start, 1.0))  
        blocks.append((start, end, score))
        current_pos = end

    if current_pos < len(response):
        blocks.append((current_pos, len(response), 1.0))

    words = response.split()
    word_scores = [1.0] * len(words)
    word_positions = []
    temp_pos = 0
    for word in words:
        start = response.find(word, temp_pos)
        if start != -1:
            end = start + len(word)
            word_positions.append((start, end))
            temp_pos = end
        else:
            word_positions.append((temp_pos, temp_pos + len(word)))
            temp_pos += len(word) + 1

    for i, (word_start, word_end) in enumerate(word_positions):
        word_center = (word_start + word_end) // 2

        for block_start, block_end, block_score in blocks:
            if block_start <= word_center < block_end:
                word_scores[i] = block_score
                break

    if merged_spans and not any(score != 1.0 for score in word_scores):
        print("WARNING: No words were assigned non-default scores despite having matched spans!")
        for span_start, span_end, span_score in merged_spans:
            for i, (word_start, word_end) in enumerate(word_positions):
                if (word_start <= span_end and word_end >= span_start):
                    word_scores[i] = span_score
                    print(f"Directly assigned score {span_score} to word '{words[i]}'")

    print(f"Word scores summary: {len([s for s in word_scores if s < 1.0])} words with negative/zero scores out of {len(word_scores)} total")
    return [(word, score) for word, score in zip(words, word_scores)]

def extract_spans_from_reward_model_output(text_to_parse):
    """
    Extract feedback and code snippet information from reward model output

    Args:
        text_to_parse: Text to parse

    Returns:
        dict: Dictionary containing code_feedback, wrong_code and improvement_code
    """
    try:
        json_result = load_json_from_string(text_to_parse, log_details=True)

        if json_result and isinstance(json_result, dict):
            return {
                "code_feedback": json_result.get("code_feedback", ""),
                "wrong_code": json_result.get("wrong_code", []),
                "improvement_code": json_result.get("improvement_code", [])
            }

        code_feedback_pattern = r'"code_feedback"\s*:\s*"([^"]*)"'
        wrong_code_pattern = r'"wrong_code"\s*:\s*\[(.*?)\]'
        improvement_code_pattern = r'"improvement_code"\s*:\s*\[(.*?)\]'

        code_feedback_match = re.search(code_feedback_pattern, text_to_parse)
        wrong_code_match = re.search(wrong_code_pattern, text_to_parse)
        improvement_code_match = re.search(improvement_code_pattern, text_to_parse)

        code_feedback = code_feedback_match.group(1) if code_feedback_match else ""

        wrong_code = []
        if wrong_code_match:
            spans_text = wrong_code_match.group(1)
            wrong_code = re.findall(r'"([^"]*)"', spans_text)

        improvement_code = []
        if improvement_code_match:
            spans_text = improvement_code_match.group(1)
            improvement_code = re.findall(r'"([^"]*)"', spans_text)

        return {
            "code_feedback": code_feedback,
            "wrong_code": wrong_code,
            "improvement_code": improvement_code
        }

    except Exception as e:
        print(f"Error parsing reward model output: {str(e)}")
        traceback.print_exc()
        return None

def inference_reward(reward_model, input_datas):
    """
    Enhanced error handling for reward model inference
    """
    try:
        reward_model.model.eval()
        with torch.no_grad():
            print("Start inference.")
            results = reward_model(input_datas, **sent_kwargs)
            print("Finish inference.")

            # Verify result format
            processed_results = []

            for result in results:
                print(result)
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
        return [[{'generated_text': ''}] for _ in input_datas]


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


def load_json_from_string(text, log_details=False):
    """
    Try to extract JSON from text using regex patterns first, then fallback to json.loads
    """
    try:
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

    # Create log directory for this round
    epoch_log_dir = os.path.join(script_args.output_dir, f"logs/epoch_{epoch}")
    os.makedirs(epoch_log_dir, exist_ok=True)

    for step, batch in tqdm(enumerate(ppo_trainer.dataloader), desc=f"Epoch {epoch + 1} "):
        try:
            os.environ["NCCL_P2P_DISABLE"] = "1"
            os.environ["NCCL_P2P_LEVEL"] = "NVL"

            if epoch == cur_epoch and step <= cur_step:
                continue

            # Create log directory for this step
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
            batch["response"] = [text.replace("assistant", "") for text in batch["response"]]

            print(batch["response"][0])  # Normal print should show the formatted text

            # Use new prepare_input_data function, no longer need instruction_config
            input_datas = prepare_input_data(batch["question"], batch["response"])
            result = inference_reward(reward_model, input_datas)
            rewards = []
            words = []
            final_question_tensors = []
            final_response_tensors = []
            new_responses = []
            # List to store the original question strings corresponding to successful samples
            final_questions = []
            fail = 0

            # Record batch level information
            batch_log = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch,
                "step": step,
                "batch_size": len(batch["question"]),
                "successful_samples": 0,
                "failed_samples": 0,
                "sample_logs": []
            }

            for ind, llm_output in enumerate(result):
                try:
                    sample_log = {
                        "index": ind,
                        # Use "question" instead of "query"
                        "question": batch["question"][ind][:100] + "..." if len(batch["question"][ind]) > 100 else batch["question"][ind],
                        "response": batch["response"][ind][:100] + "..." if len(batch["response"][ind]) > 100 else batch["response"][ind],
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

                    if len(text_to_parse) < 20:  # Likely too short to contain valid data
                        print(f"Text too short, content: {text_to_parse}")
                        sample_log["status"] = "failed"
                        sample_log["error"] = "Text too short to parse"
                        batch_log["sample_logs"].append(sample_log)
                        fail += 1
                        continue

                    # Parse reward model output, extract textual_feedback and spans
                    parsed_result = extract_spans_from_reward_model_output(text_to_parse)

                    if not parsed_result:
                        print(f"Failed to parse reward model output for sample {ind}")
                        sample_log["status"] = "failed"
                        sample_log["error"] = "Failed to parse reward model output"
                        batch_log["sample_logs"].append(sample_log)
                        fail += 1
                        continue

                    # --- Start: Added print statements for debugging span matching ---
                    print(f"\n--- Debug Span Matching (Sample {ind}) ---")
                    print(f"Response to check:\n{batch['response'][ind]}")
                    print(f"Wrong code spans:\n{parsed_result.get('wrong_code', 'N/A')}") # Use .get for safety
                    print(f"Improvement code spans:\n{parsed_result.get('improvement_code', 'N/A')}") # Use .get for safety
                    print("--- End Debug ---\n")
                    # --- End: Added print statements ---
                    # Use process_response_with_spans function to process response and code snippets
                    score_list = process_response_with_spans(
                        batch['response'][ind],
                        parsed_result['wrong_code'],
                        parsed_result['improvement_code']
                    )

                    # Extract rewards and words from score_list
                    if score_list:
                        word_rewards = [score for _, score in score_list]
                        word_list = [word for word, _ in score_list]

                        rewards.append(torch.tensor(word_rewards))
                        words.append(word_list)
                        final_question_tensors.append(question_tensors[ind])
                        final_response_tensors.append(response_tensors[ind])
                        new_responses.append(batch['response'][ind])
                        # Append the corresponding question string
                        final_questions.append(batch['question'][ind])
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

            print(f"A total of {fail} failures in the current step!!! Successfully processed {len(rewards)} samples")

            # Check the number of valid samples
            valid_samples = len(final_question_tensors)

            # In a multi-GPU environment, ensure all processes have enough valid samples
            if ppo_trainer.accelerator.num_processes > 1:
                # Get current process's valid samples
                valid_samples_tensor = torch.tensor(valid_samples, device=ppo_trainer.accelerator.device)

                # Collect all processes' valid samples
                all_valid_samples = [torch.zeros_like(valid_samples_tensor) for _ in range(ppo_trainer.accelerator.num_processes)]
                torch.distributed.all_gather(all_valid_samples, valid_samples_tensor)

                # Check if any process has fewer than threshold valid samples
                min_valid_samples = min([count.item() for count in all_valid_samples])

                if min_valid_samples < 8:  # Set minimum sample threshold to 8
                    print(f"Warning: Some processes have fewer than 8 valid samples (min: {min_valid_samples}), skipping step")
                    continue

            # Check for single GPU case
            elif valid_samples < 8:
                print(f"Warning: Too few valid samples ({valid_samples}), skipping step")
                continue

            try:
                stats, loss_ps, loss_vs, average_rewards = ppo_trainer.step(final_question_tensors,
                                                                            final_response_tensors, rewards, words,
                                                                            mask_loss=script_args.mask_loss)

                wandb.log({"train/loss_advantage": loss_ps}, step=wandb_step)
                wandb.log({"train/loss_value_kl": loss_vs}, step=wandb_step)
                wandb.log({"train/average_advantages": average_rewards.item()}, step=wandb_step)

                # Log detailed reward information for each batch
                if rewards:
                    # Calculate reward statistics
                    all_rewards = [r.tolist() for r in rewards]
                    flat_rewards = [item for sublist in all_rewards for item in sublist]

                    # Log reward distribution
                    wandb.log({
                        "rewards/mean": sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0,
                        "rewards/max": max(flat_rewards) if flat_rewards else 0,
                        "rewards/min": min(flat_rewards) if flat_rewards else 0,
                        "rewards/positive_ratio": sum(1 for r in flat_rewards if r > 0) / len(flat_rewards) if flat_rewards else 0,
                        "rewards/negative_ratio": sum(1 for r in flat_rewards if r < 0) / len(flat_rewards) if flat_rewards else 0,
                        "rewards/zero_ratio": sum(1 for r in flat_rewards if r == 0) / len(flat_rewards) if flat_rewards else 0,
                        "rewards/sample_count": len(flat_rewards)
                    }, step=wandb_step)

                    # Log histogram of rewards
                    wandb.log({"rewards/distribution": wandb.Histogram(flat_rewards)}, step=wandb_step)

                # Create a log_batch dictionary with the filtered data for log_stats
                log_batch = {"question": final_questions, "response": new_responses}
                ppo_trainer.log_stats(wandb_step, stats, log_batch, rewards)

                if step != 0 and step % script_args.save_freq == 0:
                    try:
                        # Only save on the main process to avoid conflicts
                        if ppo_trainer.accelerator.is_main_process:
                            save_path = os.path.join(script_args.output_dir, f"epoch_{epoch}_step_{step}")
                            os.makedirs(save_path, exist_ok=True)

                            # Save the model with proper error handling
                            print(f"Saving model to {save_path}...")
                            ppo_trainer.save_pretrained(save_path)

                            # Save optimizer state with proper error handling
                            optimizer_path = os.path.join(save_path, "optimizer.pt")
                            torch.save({
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
                            }, optimizer_path)

                            # Verify the saved files exist
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