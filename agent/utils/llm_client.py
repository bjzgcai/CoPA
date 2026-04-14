import os
import openai
import logging
import json
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Optional


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


load_dotenv() 

class AppConfig:
    """
    Centralizes application configuration loaded from environment variables.
    """
    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL")
    MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", 1000))
    TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.7))

def log_config_summary():
    """Logs the current configuration for verification at startup."""
    logging.info("--- OpenAI Configuration Summary ---")
    if not AppConfig.API_KEY:
        logging.warning("OPENAI_API_KEY not found in environment or .env file.")
    else:
        logging.info("OPENAI_API_KEY loaded successfully.")
    logging.info(f"Using Model: {AppConfig.MODEL}")
    logging.info(f"Using Base URL: {AppConfig.BASE_URL}")
    logging.info(f"Max Tokens: {AppConfig.MAX_TOKENS}")
    logging.info(f"Temperature: {AppConfig.TEMPERATURE}")
    logging.info("------------------------------------")


# --- 3. Core API Call Function (Refactored) ---

def call_openai_api(system_prompt: str, user_prompt: str, api_key_override: Optional[str] = None) -> Dict:
    """
    Calls the OpenAI Chat Completions API and returns a structured result.

    This function handles configuration, API key management, the API call itself,
    and robust error handling.

    Args:
        system_prompt (str): The system message to set the context for the AI.
        user_prompt (str): The user's query or instruction.
        api_key_override (Optional[str]): A specific API key for this call,
                                          overriding the environment configuration.

    Returns:
        Dict: A dictionary containing the result.
              On success: {'success': True, 'ai_response_text': str, 'model': str, 'usage': dict, ...}
              On failure: {'success': False, 'error': str, ...}
    """
    # Priority for API key: function argument > environment variable.
    final_api_key = api_key_override or AppConfig.API_KEY
    
    if not final_api_key:
        error_msg = "OpenAI API key is missing. Set it via the OPENAI_API_KEY environment variable or a .env file."
        logging.error(error_msg)
        return {'success': False, 'error': error_msg, 'timestamp': datetime.now().isoformat()}

    try:
        # Initialize the client using settings from AppConfig.
        client = openai.OpenAI(
            base_url=AppConfig.BASE_URL,
            api_key=final_api_key
        )
        
        logging.info(f"Sending request to model: {AppConfig.MODEL}...")
        
        # Call the OpenAI Chat Completions API.
        response = client.chat.completions.create(
            model=AppConfig.MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=AppConfig.MAX_TOKENS,
            temperature=AppConfig.TEMPERATURE
        )
        
        # Extract the response text and usage data.
        ai_response_text = response.choices[0].message.content
        usage_data = None
        if response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        logging.info("Successfully received response from AI.")
        
        return {
            'success': True,
            'ai_response_text': ai_response_text,
            'model': response.model,  # Get the model name from the response for accuracy.
            'usage': usage_data,      # Return token usage, which is useful for cost tracking.
            'timestamp': datetime.now().isoformat()
        }
            
    # Catch specific exceptions for clearer error messages.
    except openai.AuthenticationError as e:
        error_msg = f"OpenAI API authentication failed. Check if your API key is correct and valid. Details: {e}"
        logging.error(error_msg)
        return {'success': False, 'error': error_msg, 'timestamp': datetime.now().isoformat()}
    except openai.RateLimitError as e:
        error_msg = f"OpenAI API rate limit exceeded. Please wait and try again later. Details: {e}"
        logging.error(error_msg)
        return {'success': False, 'error': error_msg, 'timestamp': datetime.now().isoformat()}
    except openai.APIConnectionError as e:
        error_msg = f"Failed to connect to OpenAI API. Check your network and the Base URL. Details: {e}"
        logging.error(error_msg)
        return {'success': False, 'error': error_msg, 'timestamp': datetime.now().isoformat()}
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        logging.error(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }

# --- 4. Example Usage ---
if __name__ == "__main__":
    # Log the configuration at startup for debugging.
    log_config_summary()

    # Define example prompts for testing.
    test_system_prompt = "You are a helpful assistant that summarizes text."
    test_user_prompt = "Summarize the following text in one sentence: The quick brown fox jumps over the lazy dog."

    print("\n--- Calling API ---")
    # Call the API function.
    result = call_openai_api(test_system_prompt, test_user_prompt)

    print("\n--- API Result (Formatted) ---")
    # Use json.dumps for pretty-printing the result dictionary.
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Check the result and print a summary.
    if result.get('success'):
        print("\nAPI call was successful!")
        print(f"AI Response: {result.get('ai_response_text')}")
        print(f"Token Usage: {result.get('usage')}")
    else:
        print("\nAPI call failed!")
        print(f"Error: {result.get('error')}")