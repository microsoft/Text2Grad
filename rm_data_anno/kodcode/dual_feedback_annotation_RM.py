from openai import OpenAI
import json
from tqdm import tqdm  
import time
from typing import Dict, Any, List, Optional
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import aiohttp

def analyze_test_results(example: Dict[str, Any]) -> str:
    """Generate prompts for code analysis"""
    problem = example.get('question', 'No problem description provided')
    solution = example.get('solution', 'No solution provided')
    log = example.get('log', {})

    passed = log.get('passed', False)
    stdout = log.get('stdout', '')
    stderr = log.get('stderr', '')
    # Ensure test_question comes from the main example dict if that's correct
    test_question = example.get('test', '')

    prompt = f"""Analyze the following code solution for the given problem:

Problem Description:
'''
{problem}
'''

Submitted Code:
'''
{solution}
'''

Test Results:
Passed: {passed}""" # Note: {passed} will be True/False (boolean)

    # Use the boolean directly for the condition
    if passed is False:
        prompt += f"\nTest Question:\n{test_question}\n"
        # Use stderr for error output if tests failed
        prompt += f"\nError Output:\n{stdout}"
    prompt += """

Please analyze the code and identify the following in JSON format:

1. Identify any error-causing code segments directly from the submitted solution.
2. Provide detailed feedback on the code's functionality, issues, and improvement suggestions.
    - First, understand what the code is trying to accomplish
    - Analyze the algorithm and approach used
    - Identify any logical errors or inefficiencies
    - Consider edge cases and potential improvements
3. Point out any code segments from the solution that work but could be improved.

Return your analysis in this JSON structure:
```json
{
    "Code Feedback": "Provide a detailed explanation of the code's functionality, any potential issues, and suggestions for improvement. Use markdown formatting for better readability.",
    "wrong_code": ["Extract ONLY the problematic code segments FROM THE SUBMITTED SOLUTION that cause failures. Must be exact quotes. Leave empty [] if none found."],
    "improvement_code": ["Extract ONLY the working but improvable code segments FROM THE SUBMITTED SOLUTION. Must be exact quotes. Leave empty [] if none needed."]
}
```
Note: For 'wrong_code' and 'improvement_code', only include direct quotes from the submitted code above, not suggested fixes.
"""


    return prompt

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_analysis_with_retry(prompt: str) -> Optional[Dict]:
    """Get analysis from AI with retry mechanism"""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=968,
            temperature=0.1,
        )
        analysis_str = response.choices[0].message.content.strip()
        
        json_str = ""
        
        if "```json" in analysis_str:
            json_str = analysis_str.split("```json")[-1].split("```")[0].strip()
        else:
            json_str = analysis_str

        try:
            analysis_json = json.loads(json_str)
        except json.JSONDecodeError:
            json_str = json_str.replace('\n', '').replace('\r', '').strip()
            analysis_json = json.loads(json_str)

        if not all(k in analysis_json for k in ["wrong_code", "improvement_code"]):
            raise ValueError("AI response missing required keys")

        return analysis_json
    except Exception as e:
        raise

async def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single example asynchronously"""
    try:
        prompt = analyze_test_results(example)
        analysis = await get_analysis_with_retry(prompt)
        example['analysis'] = analysis
    except Exception as e:
        example['analysis'] = {"error": f"Processing failed after retries: {e}"}
    return example

async def process_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of examples concurrently"""
    return await asyncio.gather(*[process_example(example) for example in batch])

def process_data(input_filepath: str, output_filepath: str, batch_size: int = 5):
    """
    Loads data from a single JSON file, processes in batches concurrently,
    and saves the augmented data with periodic temporary saves.
    """
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, list):
            data = [data]

    processed_data: List[Dict[str, Any]] = []
    save_interval = 2000
    temp_output_filepath = output_filepath.replace('.json', '_temp.json')
    if temp_output_filepath == output_filepath:
        temp_output_filepath += "_temp"

    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch = data[i:i + batch_size]
        batch_results = asyncio.run(process_batch(batch))
        processed_data.extend(batch_results)

        if (i + batch_size) % save_interval == 0 and i > 0:
            with open(temp_output_filepath, 'w', encoding='utf-8') as f_temp:
                json.dump(processed_data, f_temp, ensure_ascii=False, indent=4)

<<<<<<< HEAD
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
=======
    print(f"Saving final augmented data ({len(processed_data)} items) to: {output_filepath}")
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)
        print("Final data saved successfully.")
    except IOError as e:
        print(f"Error: Could not write final output file to {output_filepath}. Error: {e}")
>>>>>>> ead124ae444d74cef272025a2c8db2ae4ceeb16c

if __name__ == "__main__":
    INPUT_FILE = "./data/KodCode_RM_train.json"
    OUTPUT_FILE = "KodCode_RM_train_dual_feedback.json"
    process_data(INPUT_FILE, OUTPUT_FILE, batch_size=8)
