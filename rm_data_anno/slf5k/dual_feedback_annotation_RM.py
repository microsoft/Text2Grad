import json
import os
import time
from tqdm import tqdm
from openai import OpenAI
import argparse


client = OpenAI()

def generate_critiques(input_file, output_file, batch_size=10):
    """
    Read summaries from a file, generate critiques using the API, and save results
    :param input_file: Path to the input file containing summaries
    :param output_file: Path to save the output file with critiques
    :param batch_size: Number of items to process in each batch
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = []
        for key, value in data.items():
            value['id'] = key
            items.append(value)
        data = items

    total_items = len(data)

    for i in tqdm(range(0, total_items, batch_size), desc="Generating critiques"):
        batch = data[i:i + batch_size]

        for item in batch:
            post = item.get("post", "")
            generated_summary = item.get("generated_summary", "")

            prompt_template = f"""Please critique the following summary of a post and provide feedback in the specified JSON format:

            ---

            **Original Post:**
            {post}

            **Generated Summary:**
            {generated_summary}

            ---

            **Definitions:**
            - **good_spans**: 0-2 phrases from the summary that greatly improve its quality by accurately and concisely capturing the original post's core meaning or key details, as explained in 'textual_feedback'. Empty if none apply.
            - **poor_spans**: 0-2 phrases from the summary that noticeably harm its quality due to inaccuracy, redundancy, poor wording, or being less important and replaceable with more critical content, as explained in 'textual_feedback'. Empty if none apply.

            ---

            **Instructions:**
            1. Identify the summary's most essential strengths that reflect the original post accurately and its most critical weaknesses that misrepresent or confuse it.
            2. Select 0-2 of the most significant phrases for 'good_spans' and 'poor_spans', keeping them concise and impactful, with brief justifications. Include none if no phrases stand out.
            3. Ensure 'good_spans' and 'poor_spans' are directly supported by the analysis in 'textual_feedback'.

            ---
            
            **Chain of Thought:**
            First, carefully analyze both the original post and the generated summary:
            1. What are the key points of the original post?
            2. Which of these key points are accurately captured in the summary?
            3. What important information is missing from the summary?
            4. Are there any inaccuracies or misrepresentations in the summary?
            5. Which specific phrases in the summary represent its strongest elements?
            6. Which specific phrases in the summary represent its weakest elements?

            Based on this analysis, formulate your textual feedback and identify the good and poor spans.

            ---

            **Output Format:**
            Provide a concise, one-paragraph critique and the GOOD/POOR spans in this JSON structure:
            ```json
            {{
              "textual_feedback": "Your critique here summarizing key strengths and weaknesses in one paragraph.",
              "good_spans": ["phrase1", "phrase2"],  // 0-2 concise phrases from the generated summary, tied to textual_feedback, or [] if none
              "poor_spans": ["phrase1", "phrase2"]   // 0-2 concise phrases from the generated summary, tied to textual_feedback, or [] if none
            }}
            ```

            Focus on precision: include only the most impactful phrases of the generated summary, avoiding excessive or minor details."""

            try:
                formatted_prompt = prompt_template

                gpt4_response = client.chat.completions.create(
                    model="gpt-4o-20240513",
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=0.3,
                    max_tokens=1200,
                    top_p=0.95,
                )

                critique = gpt4_response.choices[0].message.content.strip()
                item["critique"] = critique

            except Exception as e:
                item["critique"] = "Error generating critique"

        if (i + batch_size) % 50 == 0 or (i + batch_size) >= total_items:
            output_data = data
            if 'id' in data[0]:
                output_data = {}
                for item in data:
                    key = item.pop('id')
                    output_data[key] = item

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate critiques for summaries.")
    parser.add_argument("input_file", type=str, help="Path to the input file containing summaries (JSON format).")
    parser.add_argument("output_file", type=str, help="Path to save the output file with critiques (JSON format).")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of items to process in each batch (default: 10).")

    args = parser.parse_args()

    # Call the function with arguments from the command line
    generate_critiques(args.input_file, args.output_file, args.batch_size)
