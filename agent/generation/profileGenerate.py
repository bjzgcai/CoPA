import pandas as pd
import sys
import os
import json
import argparse
import tempfile
from tqdm import tqdm

from agent.prompt.prompt_formetters import get_domain_extract_prompt,get_domain_profile_extract_prompt,get_domain_profile_synthesize_prompt,get_global_profile_generate_prompt
from agent.utils.llm_client import call_openai_api, AppConfig
from agent.utils.utils import json_extract

def domain_extract(question, max_retries = 5):
    system_prompt_template, user_prompt_template = get_domain_extract_prompt()
    if system_prompt_template is None or user_prompt_template is None:
        print("Error: Could not load domain extract prompts.")
        return "Unknown"
    system_prompt = system_prompt_template
    user_prompt = user_prompt_template.format(
        question=question,
    )
    retries = 0
    while retries < max_retries:
            retries += 1
            try:
                ai_result = call_openai_api(system_prompt=system_prompt, user_prompt=user_prompt)
            
                if ai_result['success'] is False:
                    print(f"✗ Error processing domain extract: {ai_result['error']}")
                    return "Unknown"
                
                if 'ai_response_text' in ai_result:
                    cleaned_text = json_extract(ai_result['ai_response_text'])
        
                if cleaned_text:
                    domain = cleaned_text['domain']
                    return domain
            except Exception as e:
                print(f"Error domain extract (Round {retries}): {e}")
                
    return "Unknown"

def domain_profile_generate(user_profile,domain,question, max_retries = 5):
    system_prompt_template, user_prompt_template = get_domain_profile_extract_prompt()
    if system_prompt_template is None or user_prompt_template is None:
        print("Error: Could not load domain extract prompts.")
        return "Unknown"
    system_prompt = system_prompt_template.format(domain=domain)
    user_prompt = user_prompt_template.format(
        question=question,
    )
    retries = 0
    while retries < max_retries:
            retries += 1
            try:
                ai_result = call_openai_api(system_prompt=system_prompt, user_prompt=user_prompt)
            
                if ai_result['success'] is False:
                    print(f"✗ Error processing domain profile: {ai_result['error']}")
                    return user_profile
                
                if 'ai_response_text' in ai_result:
                    cleaned_text = json_extract(ai_result['ai_response_text'])
        
                if cleaned_text:
                    profile = cleaned_text['profile']
                    break
                else:
                    print("No cleaned text obtained from domain profile generation.")
            except Exception as e:
                print(f"Error logic processing (Round {retries}): {e}")

    if domain not in user_profile:
        user_profile[domain] = profile
        return user_profile
    else:
        system_prompt_template, user_prompt_template = get_domain_profile_synthesize_prompt()
        if system_prompt_template is None or user_prompt_template is None:
            print("Error: Could not load domain extract prompts.")
            return "Unknown"
        system_prompt = system_prompt_template.format(domain=domain)
        user_prompt = user_prompt_template.format(
            history=user_profile[domain],
            current=profile
        )

        while retries < max_retries:
            retries += 1
            try:
                ai_result = call_openai_api(system_prompt=system_prompt, user_prompt=user_prompt)
            
                if ai_result['success'] is False:
                    print(f"✗ Error processing domain profile: {ai_result['error']}")
                    return user_profile
                
                if 'ai_response_text' in ai_result:
                    cleaned_text = json_extract(ai_result['ai_response_text'])
        
                if cleaned_text:
                    profile = cleaned_text['profile']
                    user_profile[domain] = profile
                    return user_profile
            except Exception as e:
                print(f"Error logic processing (Round {retries}): {e}")
    return user_profile


def global_profile_generate(user_profile,profile, max_retries = 5):
    system_prompt_template, user_prompt_template = get_global_profile_generate_prompt()
    if system_prompt_template is None or user_prompt_template is None:
        print("Error: Could not load domain extract prompts.")
        return "Unknown"
    
    if "global" not in user_profile:
        user_profile["global"] = profile
        return user_profile
    else:
        system_prompt = system_prompt_template
        user_prompt = user_prompt_template.format(
           global_profile = user_profile["global"],
           domain_profile = profile
        )
    retries = 0
    while retries < max_retries:
            retries += 1
            try:
                ai_result = call_openai_api(system_prompt=system_prompt, user_prompt=user_prompt)
            
                if ai_result['success'] is False:
                    print(f"✗ Error processing global profile: {ai_result['error']}")
                    return user_profile
                
                if 'ai_response_text' in ai_result:
                    cleaned_text = json_extract(ai_result['ai_response_text'])
        
                if cleaned_text:
                    user_profile["global"] = cleaned_text['profile']
                    return user_profile
                else:
                    print("No cleaned text obtained from global profile generation.")
            except Exception as e:
                print(f"Error logic processing (Round {retries}): {e}")
    return user_profile
