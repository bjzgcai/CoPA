import argparse
import pandas as pd
import json
import os
import tempfile
from llm_client import call_openai_api
from utils import parse_json
from tqdm import tqdm

_EVAL_PROMPT_SYSTEM = """
Role: You are a fair and insightful judge with exceptional reasoning and analytical abilities. 
Your task is to evaluate whether the response to the user's question aligns with the user's factor profile.

# Evaluation Criteria (The 6 Dimensions):
    - Cognitive Trust: What are the user's epistemic requirements regarding the credibility, reliability, and verifiability of the information?
    - Situational Anchoring: To what extent does the user require the response to be contextually aligned, practically applicable, or specific to a given scenario?
    - Schema Consistency: What is the nature of the user's existing prior knowledge and mental models (and how should the new information align with them)?
    - Cognitive Load Management: What are the user's constraints regarding information processing capacity and their tolerance for complexity?
    - Metacognitive Scaffolding: What are the user's requirements for structural guidance to facilitate higher-order understanding and self-regulated learning?
    - Affective and Motivational Resonance: What are the user's expectations regarding emotional engagement and motivational alignment within the response?

# Scoring Rubric (3-point Likert scale):
- 0 (Mismatch): The response actively violates the user's preference or completely ignores a high-priority requirement defined in the profile.
- 1 (Partial Match): The response addresses the requirement but lacks depth, or only partially aligns with the user's preference.
- 2 (Full Match): The response perfectly adapts to the user's constraints and preferences described in the profile.
"""

_EVAL_PROMPT_USER = """
# Input Data:
<user_factor_profile>
{factors_profile}
</user_factor_profile>

<question>
{question}
</question>

<response_to_evaluate>
{response}
</response_to_evaluate>


The response should be in JSON format:
{{
    "Cognitive Trust": {{
        "score": 0,
        "reasoning": "Brief explanation..."
    }},
    "Situational Anchoring": {{
        "score": 0,
        "reasoning": "Brief explanation..."
    }},
    "Schema Consistency": {{
        "score": 0,
        "reasoning": "Brief explanation..."
    }},
    "Cognitive Load Management": {{
        "score": 0,
        "reasoning": "Brief explanation..."
    }},
    "Metacognitive Scaffolding": {{
        "score": 0,
        "reasoning": "Brief explanation..."
    }},
    "Affective and Motivational Resonance": {{
        "score": 0,
        "reasoning": "Brief explanation..."
    }}
}}

Requirements:
1. Analyze the match between the <user_factor_profile> and <response_to_evaluate> for EACH factor.
2. Assign a score of 0, 1, or 2. **Important: The 'score' field must be a raw INTEGER type (int), do not use strings.** (e.g., output 1, not "1").
3. Provide a brief reasoning for your score.
4. Output the result in strict JSON format.
"""

def create_eval_prompt(question, response, factors_profile):
    system_prompt = _EVAL_PROMPT_SYSTEM
    user_prompt = _EVAL_PROMPT_USER.format(question=question,
                                            response=response, 
                                            factors_profile=factors_profile)
    return system_prompt, user_prompt


def get_score(question, response, factors_profile, max_retries=3): 
    system_prompt, user_prompt = create_eval_prompt(question, response, factors_profile)
    
    expected_factors = [
        "Cognitive Trust",
        "Situational Anchoring",
        "Schema Consistency",
        "Cognitive Load Management",
        "Metacognitive Scaffolding",
        "Affective and Motivational Resonance"
    ]
    
    retries = 0
    while retries < max_retries:
        retries += 1
        try:
            ai_result = call_openai_api(system_prompt=system_prompt, user_prompt=user_prompt)
            result_json = parse_json(ai_result['ai_response_text'])
            total_score = 0
            details = {}
            valid_parse = True
            
            for factor in expected_factors:
                if factor not in result_json:
                    print(f"Missing key: {factor}, retrying...")
                    valid_parse = False
                    break
                
                factor_data = result_json[factor]
                
                raw_score = factor_data.get('score', 0)
                try:
                    score = int(raw_score)
                except ValueError:
                    score = 0 
                
                score = max(0, min(2, score))
                
                reasoning = factor_data.get('reasoning', "No reasoning provided.")
                
                total_score += score
                
                details[factor] = {
                    "score": score,
                    "reasoning": reasoning
                }
            
            if not valid_parse:
                continue 
            
            normalized_score = total_score / (len(expected_factors) * 2)
            
            return {
                "score": normalized_score, 
                "details": details 
            }

        except Exception as e:
            print(f"Error logic processing (Round {retries}): {e}")
            if retries >= max_retries:
                return {
                    "score": 0,
                    "details": {},
                }
            continue
            
    return {"score": 0, "details": {}}


def load_existing_results(result_file):
    if not os.path.exists(result_file):
        return set(), []
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, dict):
            return set(), []
        
        per_question_scores = data.get('per_question_scores', [])
        # 提取已经存在的 ID
        processed_ids = {item['id'] for item in per_question_scores if 'id' in item}
        
        print(f"✅ Found {len(processed_ids)} evaluated questions. Resuming...")
        return processed_ids, per_question_scores

    except json.JSONDecodeError:
        print(f"⚠️ Warning: Could not parse {result_file}. Starting fresh.")
        return set(), []
    except Exception as e:
        print(f"⚠️ Warning: Error loading {result_file}: {e}. Starting fresh.")
        return set(), []


def save_eval_results_atomically(new_result_item, current_list, result_file):
    
    current_list.append(new_result_item)
    
    total_score = sum(item['score'] for item in current_list)
    avg_score = total_score / len(current_list) if len(current_list) > 0 else 0
    
    final_output = {
        "score": avg_score,
        "per_question_scores": current_list
    }
    
    
    temp_dir = os.path.dirname(result_file)
    if temp_dir and not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    fd, temp_path = tempfile.mkstemp('.tmp', text=True, dir=temp_dir)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        os.replace(temp_path, result_file)
    except Exception as e:
        print(f"✗ Error saving evaluation results: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def evaluator(df, outputs, result_file):
    processed_ids, per_question_scores = load_existing_results(result_file)
    
    print(f"Start evaluating. Total tasks: {len(df)}")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Questions"):
        user_id = str(row['user_id']) 
        
        if user_id in processed_ids:
            continue
            
        question = f"{row['question']} {row['question_text']}"
        
        if user_id in outputs and len(outputs[user_id]) > 0:
            output = outputs[user_id][0]['output']
        else:
            print(f"⚠️ Warning: No output found for user {user_id}, skipping evaluation.")
            
            continue 

        factors_profile = row['factor_labels']

       
        res = get_score(question, output, factors_profile)
        
        per_question_score = {
            'id': user_id,
            'score': res['score'],
            'details': res['details']
        }
        
        print(f"User {user_id}: Score {res['score']:.2f}")

        
        save_eval_results_atomically(per_question_score, per_question_scores, result_file)
        
    return {
        "score": sum(x['score'] for x in per_question_scores)/len(per_question_scores) if per_question_scores else 0,
        "per_question_scores": per_question_scores
    }

def main():
    parser = argparse.ArgumentParser(
        description="workflow for evaluating",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-file",
        type=str,
        default="",
        help="Path to the input Parquet file."
    )
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        default="",
        help="Path to the response JSON file (the answers to evaluate)."
    )
    parser.add_argument(
        "-r", "--result-file",
        type=str,
        default="",
        help="Path to save the evaluation results."
    )
    args = parser.parse_args()


    df = pd.read_parquet(args.input_file)
    with open(args.output_file) as f:
        outputs = json.load(f)
    

    final_results = evaluator(df, outputs, args.result_file)

    print(f"Evaluation complete. Final Average Score: {final_results.get('score', 0):.4f}")
    print(f"Results saved to {args.result_file}")


if __name__ == "__main__":
    main()