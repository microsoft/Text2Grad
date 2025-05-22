from openai import AsyncOpenAI
import json
from typing import Dict, List, Any
from tqdm import tqdm
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import random
import os

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

EVALUATION_PROMPT_TEMPLATE = """Compare and evaluate two different summaries of the same query. You must respond in valid JSON format.

Original Query:
{query}


{analysis_1_label}:
{response_1}

{analysis_2_label}:
{response_2}

Evaluation Criteria:
1. Accuracy (0-10):
   - Does it capture the main points correctly?
   - Is it faithful to the original content?
   - Are there any factual errors?

2. Completeness (0-10):
   - Are all key points included?
   - Is any important information missing?
   - Does it cover the core message?

3. Conciseness (0-10):
   - Is it clear and to the point?
   - Does it avoid unnecessary details?
   - Is the language efficient?

4. Coherence (0-10):
   - Is the summary well-organized?
   - Does it flow logically?
   - Is it easy to understand?

Compare both summaries and evaluate them. Respond ONLY with a JSON object in this exact format:
{{
    "{score_key_1}": {{
        "strengths": ["specific strength 1", "specific strength 2", ...],
        "weaknesses": ["specific weakness 1", "specific weakness 2", ...]
        "score": <overall score between 0-10>,
        "accuracy": <score between 0-10>,
        "completeness": <score between 0-10>,
        "conciseness": <score between 0-10>,
        "coherence": <score between 0-10>,
    }},
    "{score_key_2}": {{
        "strengths": ["specific strength 1", "specific strength 2", ...],
        "weaknesses": ["specific weakness 1", "specific weakness 2", ...]
        "score": <overall score between 0-10>,
        "accuracy": <score between 0-10>,
        "completeness": <score between 0-10>,
        "conciseness": <score between 0-10>,
        "coherence": <score between 0-10>,
    }}
}}"""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_evaluation(prompt: str) -> Dict:
    try:
        response = await client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a helpful text summarization assistant that always responds with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            raise

    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        print(f"Response content: {content}")
        raise

async def evaluate_single_example(example: Dict[str, Any], response1: str, response2: str) -> Dict[str, Any]:
    """Evaluate a single example with randomized position"""
    try:
        # Randomly decide order
        if random.random() < 0.5:
            response_1, response_2 = response1, response2
            score_key_1, score_key_2 = "response1", "response2"
            is_flipped = False
        else:
            response_1, response_2 = response2, response1
            score_key_1, score_key_2 = "response2", "response1"
            is_flipped = True

        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            query=example.get('query', ''),
            response_1=response_1,
            response_2=response_2,
            score_key_1=score_key_1,
            score_key_2=score_key_2,
            analysis_1_label="Summary A",
            analysis_2_label="Summary B"
        )

        evaluation = await get_evaluation(prompt)

        return {
            'query': example.get('query', ''),
            'evaluation': evaluation,
            'is_flipped': is_flipped,
            'responses': {
                'response1': response1,
                'response2': response2
            }
        }
    except Exception as e:
        print(f"Error evaluating example: {str(e)}")
        print(f"Query: {example.get('query', '')[:100]}...")
        return None

async def main():
    # Load data files
    print("Loading data files...")

    with open('./result/checkpoint-400-merge_samples.json', 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    with open('./result/epoch_2_step_200_merge_samples.json', 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    batch_size = 8
    all_evaluations = []

    common_examples = []
    for ex1, ex2 in zip(data1, data2):
        if ex1.get('query') == ex2.get('query'):
            common_examples.append({
                'example': ex1,
                'response1': ex1.get('response', ''),
                'response2': ex2.get('response', '')
            })

    print(f"Found {len(common_examples)} matching examples to evaluate")

    def print_statistics(evaluations):
        """Helper function to print statistics for a set of evaluations"""
        total = len(evaluations)
        if total == 0:
            return

        wins_1 = 0
        wins_2 = 0
        ties = 0
        total_score_1 = 0
        total_score_2 = 0

        for eval in evaluations:
            score_1 = eval['evaluation']['response1']['score']
            score_2 = eval['evaluation']['response2']['score']

            if eval['is_flipped']:
                score_1, score_2 = score_2, score_1

            total_score_1 += score_1
            total_score_2 += score_2

            if score_1 > score_2:
                wins_1 += 1
            elif score_2 > score_1:
                wins_2 += 1
            else:
                ties += 1

        print("\nCurrent Statistics:")
        print(f"Samples evaluated: {total}")
        print(f"Average score for first model: {total_score_1/total:.2f}")
        print(f"Average score for second model: {total_score_2/total:.2f}")
        print("\nWin Rate Analysis:")
        print(f"First model wins: {wins_1} ({(wins_1/total)*100:.1f}%)")
        print(f"Second model wins: {wins_2} ({(wins_2/total)*100:.1f}%)")
        print(f"Ties: {ties} ({(ties/total)*100:.1f}%)")
        print("-" * 50)

    for i in tqdm(range(0, len(common_examples), batch_size)):
        batch = common_examples[i:i + batch_size]
        batch_results = await asyncio.gather(*[
            evaluate_single_example(item['example'], item['response1'], item['response2'])
            for item in batch
        ])
        batch_results = [r for r in batch_results if r is not None]
        all_evaluations.extend(batch_results)

        if len(all_evaluations) % 30 == 0:
            print(f"\nProcessed {len(all_evaluations)} samples")
            print_statistics(all_evaluations)

        if (i + batch_size) % 10 == 0:
            with open('response_evaluation_results.json', 'w', encoding='utf-8') as f:
                json.dump(all_evaluations, f, indent=2, ensure_ascii=False)

    print("\nFinal Results:")
    print_statistics(all_evaluations)

if __name__ == "__main__":
    asyncio.run(main())