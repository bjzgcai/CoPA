import pandas as pd
import sys
import os
import json
import argparse
import tempfile
from tqdm import tqdm

from agent.prompt.prompt_formetters import get_prompt
from agent.utils.llm_client import call_openai_api, AppConfig
from agent.utils.utils import json_extract

def load_processed_users(json_path: str) -> set:
    if not os.path.exists(json_path):
        return set()

    if os.path.getsize(json_path) == 0:
        return set()

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        
        if isinstance(data, dict):
            processed_ids = set(data.keys())
            if processed_ids:
                print(f"✅ Found {len(processed_ids)} processed users in '{json_path}'. Resuming workflow.")
                return processed_ids
        
        print(f"⚠️ Warning: File format is not a dictionary or empty. Starting fresh/appending.")
        return set()

    except json.JSONDecodeError:
        print(f"⚠️ Warning: Could not parse the JSON file. Starting fresh.")
        return set()
    except Exception as e:
        print(f"⚠️ Warning: Error reading file: {e}. Starting fresh.")
        return set()
    

def save_results_atomically(data: dict, json_path: str):
    
    temp_dir = os.path.dirname(json_path)
    if temp_dir and not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # 1. 读取现有数据 (Read Phase)
    current_content = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                
                if isinstance(file_data, dict):
                    current_content = file_data
        except json.JSONDecodeError:
            current_content = {}

   
    current_content.update(data)

    
    fd, temp_path = tempfile.mkstemp('.tmp', text=True, dir=temp_dir)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(current_content, f, indent=4, ensure_ascii=False)
        
        os.replace(temp_path, json_path)
    except Exception as e:
        print(f"✗ Error saving results to {json_path}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def workflow(input_parquet_path: str,
            output_json_path: str,
            type: str,
            factor: bool = False,
            max_retries = 5
            ) -> bool:
    
    print("=== Response generate Workflow ===")
    print(f"Input file: {input_parquet_path}")
    print(f"Output file: {output_json_path}")
    
    # 1. Read data
    print("\n1. Reading data...")
    try:
        df = pd.read_parquet(input_parquet_path)
    except Exception as e:
        print(f"Error reading parquet: {e}")
        return False
        
    print(f"Results will be saved incrementally (merged into dict) to: {output_json_path}")

    
    processed_user_ids = load_processed_users(output_json_path)
    
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing each user"):
        user_id = str(row['user_id']) 
        if user_id in processed_user_ids:
            continue  
        
        
        historical_questions = ""
        current_question = f"{row['question']} {row['question_text']}"

        
        profile_list = row['profile']
        if type == "Rag-Personalization": 
            sorted_profiles = profile_list

        elif type == "Time-Personalization":
            sorted_profiles = sorted(profile_list, key=lambda p: p['creation_date'], reverse=True)
        else:  
            sorted_profiles = profile_list
            
        recent_profiles = sorted_profiles[:50]
        historical_questions_list = [p['question'] for p in recent_profiles]
        historical_questions = "\n".join([f"{i}. {q}" for i, q in enumerate(historical_questions_list, 1)])
            
        system_prompt, user_prompt = get_prompt(current_question=current_question,
                                               historical_questions=historical_questions,
                                                type=type,
                                               factor=factor,)
        retries = 0
        current_user_result = None 

        while retries < max_retries:
            retries += 1
            try:
                ai_result = call_openai_api(system_prompt=system_prompt, user_prompt=user_prompt)
            
                if ai_result['success'] is False:
                    print(f"✗ Error processing user {user_id}: {ai_result['error']}")
                
                if 'ai_response_text' in ai_result:
                    cleaned_text = json_extract(ai_result['ai_response_text'])
        
                if cleaned_text:
                    
                    current_user_result = {
                        user_id: [{"output": cleaned_text['answer']}]
                    }
                    break
            except Exception as e:
                print(f"Error logic processing (Round {retries}): {e}")
                
        
        if current_user_result is None:
            current_user_result = {
                user_id: [{"output": ""}]
            }
        
        
        save_results_atomically(current_user_result, output_json_path)
            
    print(f"\n✅ Workflow complete. Full results saved to {output_json_path}")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Baseline Response Generation Workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-i", "--input-file", type=str, required=True)
    parser.add_argument("-o", "--output-file", type=str, required=True)
    parser.add_argument("-t", "--type", type=str, default='No-Personalization',
                        choices=['No-Personalization', 'Rag-Personalization', 'Time-Personalization'])
    parser.add_argument("-f", "--factor", action='store_true')
    
    args = parser.parse_args()

    success = workflow(
        input_parquet_path=args.input_file,
        output_json_path=args.output_file,
        type=args.type,
        factor=args.factor
    )
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()