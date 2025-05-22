from datasets import load_dataset, Dataset
from collections import Counter
from tqdm import tqdm
import os
import sys
import tempfile
import subprocess
import random
import argparse

def parse_arguments():
    """
    Parse command line arguments for sample sizes
    """
    parser = argparse.ArgumentParser(description='Process KodCode datasets with configurable sample sizes')
    parser.add_argument('--train_samples', type=int, default=2000,
                        help='Number of samples to select from train split (default: 2000)')
    parser.add_argument('--target_total_size', type=int, default=10000,
                        help='Target total size for the final dataset (default: 10000)')
    parser.add_argument('--easy_samples', type=int, default=500,
                        help='Number of easy difficulty samples to select (default: 500)')
    parser.add_argument('--medium_samples', type=int, default=20000,
                        help='Number of medium difficulty samples to select (default: 20000)')
    parser.add_argument('--hard_samples', type=int, default=20000,
                        help='Number of hard difficulty samples to select (default: 20000)')
    parser.add_argument('--output_file', type=str, default="KodCode_RM_train.json",
                        help='Output file name (default: KodCode_RM_train.json)')
    return parser.parse_args()

def extract_missing_modules(error_text):
    """从错误信息中提取缺失的模块名称"""
    missing_modules = []
    for line in error_text.split('\n'):
        if "ModuleNotFoundError: No module named" in line:
            module = line.split("'")[1]
            missing_modules.append(module)
    return missing_modules

def run_tests_and_log_results(example):
    """
    Run tests and log results.

    Args:
        example: A dictionary containing 'solution' and 'test' keys

    Returns:
        The example with an added 'log' key containing test results and detailed stderr information
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create solution and test files
        solution_path = os.path.join(temp_dir, 'solution.py')
        test_path = os.path.join(temp_dir, 'test_solution.py')
        init_path = os.path.join(temp_dir, '__init__.py')

        # Write files
        with open(solution_path, 'w') as f:
            f.write(example['solution'])
        with open(test_path, 'w') as f:
            f.write(example['test'])
        with open(init_path, 'w') as f:
            f.write('')

        # 先尝试导入解决方案模块以捕获导入错误
        try:
            import_cmd = [
                sys.executable,
                '-c',
                'import sys; sys.path.insert(0, "."); import solution'
            ]
            import_result = subprocess.run(
                import_cmd,
                capture_output=True,
                text=True,
                cwd=temp_dir,
                timeout=10
            )

            if import_result.returncode != 0:
                full_error = f"""
Import Error Details:
Return Code: {import_result.returncode}
Stderr: {import_result.stderr}
Stdout: {import_result.stdout}
Missing Modules: {extract_missing_modules(import_result.stderr)}
"""
                return {
                    **example,
                    'log': {
                        'stdout': import_result.stdout,
                        'stderr': full_error,
                        'return_code': import_result.returncode,
                        'passed': False,
                        'error_type': 'import_error',
                        'missing_modules': extract_missing_modules(import_result.stderr)
                    }
                }

        except subprocess.TimeoutExpired as e:
            timeout_error = f"""
Import Timeout Error:
Command: {e.cmd}
Timeout: 10 seconds
Stdout: {e.stdout if e.stdout else 'None'}
Stderr: {e.stderr if e.stderr else 'None'}
"""
            return {
                **example,
                'log': {
                    'stdout': e.stdout if e.stdout else '',
                    'stderr': timeout_error,
                    'return_code': -1,
                    'passed': False,
                    'error_type': 'timeout'
                }
            }

        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', test_path, '-v'],
                capture_output=True,
                text=True,
                cwd=temp_dir,
                timeout=30
            )

            full_test_log = f"""
Test Execution Details:
Return Code: {result.returncode}
Stdout:
{result.stdout}

Stderr:
{result.stderr}
"""
            return {
                **example,
                'log': {
                    'stdout': result.stdout,
                    'stderr': full_test_log,
                    'return_code': result.returncode,
                    'passed': result.returncode == 0,
                    'error_type': 'test_failure' if result.returncode != 0 else None
                }
            }

        except subprocess.TimeoutExpired as e:
            timeout_error = f"""
Test Execution Timeout:
Command: {e.cmd}
Timeout: 30 seconds
Stdout: {e.stdout if e.stdout else 'None'}
Stderr: {e.stderr if e.stderr else 'None'}
"""
            return {
                **example,
                'log': {
                    'stdout': e.stdout if e.stdout else '',
                    'stderr': timeout_error,
                    'return_code': -1,
                    'passed': False,
                    'error_type': 'timeout'
                }
            }
        except Exception as e:
            execution_error = f"""
Execution Error Details:
Error Type: {type(e).__name__}
Error Message: {str(e)}
"""
            return {
                **example,
                'log': {
                    'stdout': '',
                    'stderr': execution_error,
                    'return_code': -1,
                    'passed': False,
                    'error_type': 'execution_error'
                }
            }

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Load the datasets
    train_dataset = load_dataset("KodCode/KodCode-V1-SFT-R1", split="train")
    incorrect_dataset = load_dataset("KodCode/KodCode-V1-SFT-R1", split="incorrect")
    
    # Randomly sample from train split based on argument
    train_samples = train_dataset.shuffle(seed=42).select(range(args.train_samples))
    
    # Group incorrect examples by difficulty
    difficulty_groups = {'easy': [], 'medium': [], 'hard': []}
    for example in incorrect_dataset:
        diff = example['gpt_difficulty']
        if diff in difficulty_groups:
            difficulty_groups[diff].append(example)
    
    target_counts = {
        'easy': args.easy_samples, 
        'medium': args.medium_samples, 
        'hard': args.hard_samples
    }
    incorrect_balanced_dataset = []
    for diff, examples in difficulty_groups.items():
        target = target_counts[diff]
        if len(examples) >= target:
            selected_examples = examples[:target]
            incorrect_balanced_dataset.extend(selected_examples)
            print(f"Selected {target} incorrect examples for {diff}")
        else:
            incorrect_balanced_dataset.extend(examples)
            print(f"Warning: Only {len(examples)} incorrect examples available for {diff}")
    
    balanced_dataset = list(train_samples) + incorrect_balanced_dataset
    
    print("\nRunning tests on balanced dataset...")
    processed_dataset = []
    error_types = {} 
    
    for i, example in enumerate(tqdm(balanced_dataset, desc="Running tests")):
        processed_example = run_tests_and_log_results(example)
        processed_dataset.append(processed_example)
        
        print(f"\nTest {i+1}/{len(balanced_dataset)}:")
        print(f"Difficulty: {processed_example['gpt_difficulty']}")
        print(f"Status: {'PASSED' if processed_example['log']['passed'] else 'FAILED'}")
        
        if not processed_example['log']['passed']:
            print(f"Error type: {processed_example['log']['error_type']}")
            if processed_example['log']['error_type'] == 'import_error':
                print(f"Missing modules: {processed_example['log']['missing_modules']}")
            print(f"Stderr:\n{processed_example['log']['stderr']}")
            
            error_type = processed_example['log']['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        if (i + 1) % 10 == 0:
            passed_so_far = sum(1 for ex in processed_dataset if ex['log']['passed'])
            print(f"\nProgress statistics (after {i+1} tests):")
            print(f"Passed: {passed_so_far}/{i+1} ({passed_so_far/(i+1)*100:.2f}%)")
            if error_types:
                print("Error type distribution:")
                for error_type, count in error_types.items():
                    print(f"  {error_type}: {count} cases")
    
    failed_cases = [ex for ex in processed_dataset if not ex['log']['passed']]
    passed_cases = [ex for ex in processed_dataset if ex['log']['passed']]
    
    target_total_size = args.target_total_size
    num_failed = len(failed_cases)
    num_passed_needed = max(0, target_total_size - num_failed)
    
    random.seed(42)
    selected_passed_cases = random.sample(passed_cases, min(num_passed_needed, len(passed_cases)))
    
    final_dataset_list = failed_cases + selected_passed_cases
    final_dataset = Dataset.from_list(final_dataset_list)
    final_dataset.to_json(args.output_file)
    
    print(f"\nFinal Dataset Statistics:")
    print(f"Total examples: {len(final_dataset_list)}")
    print(f"Failed cases: {len(failed_cases)}")
    print(f"Selected passed cases: {len(selected_passed_cases)}")
    print("Difficulty distribution:",
          Counter([ex['gpt_difficulty'] for ex in final_dataset_list]))
    
    print(f"\nFinal Statistics:")
    print(f"Total examples: {len(processed_dataset)}")
    print("Difficulty distribution:",
          Counter([ex['gpt_difficulty'] for ex in processed_dataset]))
    
    passing_examples = sum(1 for ex in processed_dataset if ex['log']['passed'])
    print(f"\nTest Results Summary:")
    print(f"Total examples tested: {len(processed_dataset)}")
    print(f"Passing examples: {passing_examples} ({passing_examples/len(processed_dataset)*100:.2f}%)")
    
    print("\nPassing examples by difficulty:")
    for diff in ['easy', 'medium', 'hard']:
        total_in_diff = sum(1 for ex in processed_dataset if ex['gpt_difficulty'] == diff)
        passed_in_diff = sum(1 for ex in processed_dataset
                            if ex['gpt_difficulty'] == diff and ex['log']['passed'])
        if total_in_diff > 0:
            print(f"{diff}:")
            print(f"  Total: {total_in_diff}")
            print(f"  Passed: {passed_in_diff} ({passed_in_diff/total_in_diff*100:.2f}%)")
    
    if error_types:
        print("\nError type distribution:")
        total_errors = sum(error_types.values())
        for error_type, count in error_types.items():
            print(f"  {error_type}: {count} cases ({count/total_errors*100:.2f}%)")