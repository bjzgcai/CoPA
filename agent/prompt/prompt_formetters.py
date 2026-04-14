from pathlib import Path
from typing import Tuple, Optional

_factors_prompt = """Your response must incorporate personalization by addressing the following six dimensions:
1. Cognitive Trust: Does the response align with the user's threshold for trust and credibility within this specific domain?
2. Situational Anchoring: Is the response precisely calibrated to the user's immediate context and specific problem?
3. Schema Consistency: Does the response integrate coherently with the user's prior knowledge and existing mental models?
4. Cognitive Load Management: Is the complexity of the response tailored to match the user's cognitive capacity?
5. Metacognitive Scaffolding: Does the response provide structural support that fosters the user's critical thinking skills?
6. Affective and Motivational Resonance: Does the response resonate with the user's current emotional state and motivational orientation?"""



def read_NoPersonalization_prompt(factor: bool) -> Optional[Tuple[str, str]]:
    try:
        base_path = Path(__file__).parent
        system_prompt_path = base_path / 'system' / 'No-Personalization.txt'
        user_prompt_path = base_path / 'user' / 'No-Personalization.txt' 
        with open(system_prompt_path, 'r', encoding='utf-8') as file:
            system_prompt_template = file.read()
        with open(user_prompt_path, 'r', encoding='utf-8') as file:
            user_prompt_template = file.read()
        if factor:
            system_prompt_template = f"{system_prompt_template}\n\n{_factors_prompt}"
        return system_prompt_template, user_prompt_template
    
    except FileNotFoundError:
        print(f"Error: A file was not found. Please check paths: 'agent/prompt/system/No-Personalization.txt'")
        return None
    except Exception as e:
        print(f"An error occurred while reading files: {e}")
        return None
    

def read_RagPersonalization_prompt(factor: bool) -> Optional[Tuple[str, str]]:
    try:
        base_path = Path(__file__).parent
        system_prompt_path = base_path / 'system' / 'Rag-Personalization.txt'
        user_prompt_path = base_path / 'user' / 'Rag-Personalization.txt' 
        with open(user_prompt_path, 'r', encoding='utf-8') as file:
            user_prompt_template = file.read()
        with open(system_prompt_path, 'r', encoding='utf-8') as file:
            system_prompt_template = file.read()
        if factor:
            system_prompt_template = f"{system_prompt_template}\n\n{_factors_prompt}"
        return system_prompt_template, user_prompt_template
    
    except FileNotFoundError:
        print(f"Error: A file was not found. Please check paths: 'agent/prompt/system/No-Personalization.txt'")
        return None
    except Exception as e:
        print(f"An error occurred while reading files: {e}")
        return None
    
def read_TimePersonalization_prompt(factor: bool) -> Optional[Tuple[str, str]]:
    try:
        base_path = Path(__file__).parent
        system_prompt_path = base_path / 'system' / 'Time-Personalization.txt'
        user_prompt_path = base_path / 'user' / 'Time-Personalization.txt' 

        with open(user_prompt_path, 'r', encoding='utf-8') as file:
            user_prompt_template = file.read()
        with open(system_prompt_path, 'r', encoding='utf-8') as file:
            system_prompt_template = file.read()
        if factor:
            system_prompt_template = f"{system_prompt_template}\n\n{_factors_prompt}"
        return system_prompt_template, user_prompt_template
    
    except FileNotFoundError:
        print(f"Error: A file was not found. Please check paths: 'agent/prompt/system/Time-Personalization.txt'")
        return None
    except Exception as e:
        print(f"An error occurred while reading files: {e}")
        return None

def get_prompt(historical_questions: str, current_question: str, type: str, factor: bool) -> Optional[Tuple[str, str]]:
    if type == "No-Personalization":
        system_prompt_template, user_prompt_template =  read_NoPersonalization_prompt(factor)
        system_prompt = system_prompt_template
        user_prompt = user_prompt_template.format(
            question=current_question,
        )
        return system_prompt, user_prompt
    elif type == "Rag-Personalization":
        system_prompt_template, user_prompt_template = read_RagPersonalization_prompt(factor)
        system_prompt = system_prompt_template
        user_prompt = user_prompt_template.format(
            Historical_questions=historical_questions,
            Current_question=current_question,
        )
        return system_prompt, user_prompt
    elif type == "Time-Personalization":
        system_prompt_template, user_prompt_template = read_RagPersonalization_prompt(factor)
        system_prompt = system_prompt_template
        user_prompt = user_prompt_template.format(
            Historical_questions=historical_questions,
            Current_question=current_question,
        )
        return system_prompt, user_prompt
    else:
        print(f"Error: Unknown prompt type '{type}'. Available types are 'No-Personalization' and 'Rag-Personalization'.")
        return None

def get_domain_extract_prompt() -> Optional[Tuple[str, str]]:
    try:
        base_path = Path(__file__).parent
        system_prompt_path = base_path / 'system'/'profileqa' / 'domain_extract.txt'
        user_prompt_path = base_path / 'user' /'profileqa' / 'domain_extract.txt' 

        with open(user_prompt_path, 'r', encoding='utf-8') as file:
            user_prompt_template = file.read()
        with open(system_prompt_path, 'r', encoding='utf-8') as file:
            system_prompt_template = file.read()
        return system_prompt_template, user_prompt_template
    
    except FileNotFoundError:
        print(f"Error: A file was not found. Please check paths: 'agent/prompt/system/Time-Personalization.txt'")
        return None
    except Exception as e:
        print(f"An error occurred while reading files: {e}")
        return None
    

def get_domain_profile_extract_prompt() -> Optional[Tuple[str, str]]:
    try:
        base_path = Path(__file__).parent
        system_prompt_path = base_path / 'system'/'profileqa' / 'domain_profile_extract.txt'
        user_prompt_path = base_path / 'user' /'profileqa' / 'domain_profile_extract.txt' 

        with open(user_prompt_path, 'r', encoding='utf-8') as file:
            user_prompt_template = file.read()
        with open(system_prompt_path, 'r', encoding='utf-8') as file:
            system_prompt_template = file.read()
        return system_prompt_template, user_prompt_template
    
    except FileNotFoundError:
        print(f"Error: A file was not found. Please check paths: 'agent/prompt/system/Time-Personalization.txt'")
        return None
    except Exception as e:
        print(f"An error occurred while reading files: {e}")
        return None
    

def get_domain_profile_synthesize_prompt() -> Optional[Tuple[str, str]]:
    try:
        base_path = Path(__file__).parent
        system_prompt_path = base_path / 'system'/'profileqa' / 'domain_profile_synthesize.txt'
        user_prompt_path = base_path / 'user' /'profileqa' / 'domain_profile_synthesize.txt' 

        with open(user_prompt_path, 'r', encoding='utf-8') as file:
            user_prompt_template = file.read()
        with open(system_prompt_path, 'r', encoding='utf-8') as file:
            system_prompt_template = file.read()
        return system_prompt_template, user_prompt_template
    
    except FileNotFoundError:
        print(f"Error: A file was not found. Please check paths: 'agent/prompt/system/Time-Personalization.txt'")
        return None
    except Exception as e:
        print(f"An error occurred while reading files: {e}")
        return None
    

def get_global_profile_generate_prompt() -> Optional[Tuple[str, str]]:
    try:
        base_path = Path(__file__).parent
        system_prompt_path = base_path / 'system'/'profileqa' / 'global_profile_generate.txt'
        user_prompt_path = base_path / 'user' /'profileqa' / 'global_profile_generate.txt' 
        with open(user_prompt_path, 'r', encoding='utf-8') as file:
            user_prompt_template = file.read()
        with open(system_prompt_path, 'r', encoding='utf-8') as file:
            system_prompt_template = file.read()
        return system_prompt_template, user_prompt_template
    
    except FileNotFoundError:
        print(f"Error: A file was not found. Please check paths: 'agent/prompt/system/Time-Personalization.txt'")
        return None
    except Exception as e:
        print(f"An error occurred while reading files: {e}")
        return None

def get_profileqa_answer_prompt(factor) -> Optional[Tuple[str, str]]:
    try:
        base_path = Path(__file__).parent
        system_prompt_path = base_path / 'system'/'profileqa' / 'answer_generation.txt'
        user_prompt_path = base_path / 'user' /'profileqa' / 'answer_generation.txt' 
        with open(user_prompt_path, 'r', encoding='utf-8') as file:
            user_prompt_template = file.read()
        with open(system_prompt_path, 'r', encoding='utf-8') as file:
            system_prompt_template = file.read()

        if factor:
            system_prompt_template = f"{system_prompt_template}\n\n{_factors_prompt}"
        return system_prompt_template, user_prompt_template
    
    except FileNotFoundError:
        print(f"Error: A file was not found. Please check paths: 'agent/prompt/system/Time-Personalization.txt'")
        return None
    except Exception as e:
        print(f"An error occurred while reading files: {e}")
        return None