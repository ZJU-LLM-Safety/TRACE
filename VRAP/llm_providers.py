#!/usr/bin/env python3
"""
LLM Provider Management
Supports OpenAI and OpenRouter providers with unified interface
"""

import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMProvider:
    """Base LLM provider interface"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = None
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate chat completion
        
        Returns:
            Tuple[str, Dict[str, int]]: (response_content, token_usage)
            where token_usage contains 'prompt_tokens', 'completion_tokens', 'total_tokens'
        """
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # api_key = os.getenv('OPENAI_API_KEY')
        # if not api_key:
        #     raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key="sk-Y9sEZDrPHBFdgA9o2qV0wXkPUejydYJfE2mSl5y7jaArQ1XH", base_url="https://az.gptplus5.com/v1")
        if "deepseek" in model_name.lower(): # and "pro" in model_name.lower():
            self.client = OpenAI(api_key="sk-c38f2bcafba54ffdad2518d4626f27f5", base_url="https://api.deepseek.com")
        elif "gpt-5.3-chat" in model_name:
            self.client = OpenAI(api_key="sk-8bwuLBrYstPW7SDz30284771Ef274c7aB525Db7917640cBe", base_url="https://api.gpt.ge/v1")
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate chat completion using OpenAI"""
        # Extract temperature to avoid duplicate parameter
        temperature = kwargs.pop('temperature', 0)
        
        # Validate model name
        if not self.model_name:
            raise ValueError(f"Model name is empty or None: '{self.model_name}'")
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        
        # Extract token usage information
        token_usage = {
            'prompt_tokens': completion.usage.prompt_tokens,
            'completion_tokens': completion.usage.completion_tokens,
            'total_tokens': completion.usage.total_tokens
        }
        
        return completion.choices[0].message.content, token_usage

class OpenRouterProvider(LLMProvider):
    """OpenRouter provider implementation"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate chat completion using OpenRouter"""
        # Extract temperature to avoid duplicate parameter
        temperature = kwargs.pop('temperature', 0)
        
        # Validate model name
        if not self.model_name:
            raise ValueError(f"Model name is empty or None: '{self.model_name}'")
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        
        # Extract token usage information
        token_usage = {
            'prompt_tokens': completion.usage.prompt_tokens,
            'completion_tokens': completion.usage.completion_tokens,
            'total_tokens': completion.usage.total_tokens
        }
        
        return completion.choices[0].message.content, token_usage


def create_provider(provider_name: str, model_name: str) -> LLMProvider:
    """Factory function to create LLM provider"""
    if provider_name.lower() == 'openai':
        return OpenAIProvider(model_name)
    elif provider_name.lower() == 'openrouter':
        return OpenRouterProvider(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")