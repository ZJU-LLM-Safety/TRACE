import logging
import os
from random import uniform
from time import sleep
from typing import Dict, List, Optional, Tuple, Union

import aisuite as ai
import google.auth
import google.auth.transport.requests
import requests
from colorama import Fore, Style
from openai import OpenAI
from vertexai.generative_models._generative_models import ResponseValidationError


class APICallError(Exception):
    """Custom exception for API call failures"""

    pass


class BaseAgent:
    """Universal base agent for handling different LLM APIs.

    Configuration structure:
    {
        "provider": "openai|google|anthropic|sglang|ollama|together|openrouter",  # Required: API provider
        "model": "model-name",                                # Required: Model identifier
        "temperature": 0.7,                                   # Required: Temperature for sampling
        "max_retries": 3,                                     # Optional: Number of retries (default: 3)

        # Provider-specific configurations
        "project_id": "your-project",           # Required for Google
        "location": "us-central1",              # Required for Google
        "port": 30000,                          # Required for SGLang
        "base_url": "http://localhost:11434",   # Required for Ollama
        "api_key": "ollama",                    # Required for Ollama
        "response_format": {"type": "json_object"}  # Optional: For JSON responses (only for openai models) Make sure you include the word json in some form in the message
        "http_referer": "your-site-url",        # Optional for OpenRouter: Site URL for rankings
        "x_title": "your-site-name"             # Optional for OpenRouter: Site title for rankings
    }

    Usage:
        agent = BaseAgent(config)
        response = agent.call_api(messages, temperature=0.7)
        # Or with message inspection:
        response, messages = agent.call_api(messages, temperature=0.7, return_messages=True)
        # For JSON response (openai only):
        response = agent.call_api(messages, temperature=0.7, response_format={"type": "json_object"})
    """

    _transformers_tokenizers = {}
    _transformers_models = {}

    def __init__(self, config: Dict):
        """Initialize base agent with provider-specific setup."""
        self.config = config
        self.provider = config["provider"]
        self.max_retries = config.get("max_retries", 3)

        # Initialize client
        try:
            if self.provider == "openai":
                if (
                    os.getenv("OPENAI_BASE_URL")
                    or config.get("base_url")
                    or any(f"o{i}" in config["model"] for i in range(1, 6))
                    or config["model"].startswith("gpt-5")
                ):
                    self.client = OpenAI(
                        base_url=config.get("base_url") or os.getenv("OPENAI_BASE_URL"),
                        api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
                    )
                    self.model = config["model"]
                else:
                    self.client = ai.Client()
                    self.model = f"{self.provider}:{config['model']}"
            elif self.provider == "openrouter":
                # Initialize OpenRouter using OpenAI client with custom base URL
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY", ""),
                )
                self.model = config["model"]
                # Store optional headers for OpenRouter
                self.http_referer = config.get("http_referer")
                self.x_title = config.get("x_title")
            elif self.provider == "google":
                if "meta" in config["model"]:
                    self.project_id = config["project_id"]
                    self.location = config["location"]
                    self.model = config["model"]
                else:
                    os.environ["GOOGLE_PROJECT_ID"] = config["project_id"]
                    os.environ["GOOGLE_REGION"] = config["location"]
                    self.client = ai.Client()
                    self.model = f"{self.provider}:{config['model']}"
            elif self.provider == "sglang":
                # Initialize SGLang using OpenAI client
                self.client = OpenAI(
                    base_url=f"http://localhost:{config.get('port', 30000)}/v1",
                    api_key="None",  # SGLang doesn't require an API key
                )
                self.model = config["model"]
            elif self.provider == "transformers":
                self.client = None
                self.model = config["model"]
                self._ensure_transformers_model_loaded(config)
            elif self.provider == "ollama":
                self.client = ai.Client()
                self.model = f"{self.provider}:{config['model']}"
                self.client.configure(
                    {
                        "ollama": {
                            "timeout": 60,
                        }
                    }
                )
            elif self.provider == "together":
                self.client = ai.Client()
                self.model = f"{self.provider}:{config['model']}"
                self.client.configure(
                    {
                        "together": {
                            "timeout": 90,
                        }
                    }
                )
            else:
                # For all other providers
                self.client = ai.Client()
                self.model = f"{self.provider}:{config['model']}"

        except Exception as e:
            raise APICallError(f"Error initializing {self.provider} client: {str(e)}")

    def call_api(
        self,
        messages: List[Dict],
        temperature: float,
        response_format: Optional[Dict] = None,
        return_messages: bool = False,
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """Universal API call handler with retries.

        Args:
            messages: List of message dictionaries
            temperature: Float value for temperature
            response_format: Optional response format specifications
            return_messages: If True, returns tuple of (response, messages)

        Returns:
            Either string response or tuple of (response, messages) if return_messages=True
        """
        print(
            f"{Fore.GREEN}Model is {self.model}, temperature is {temperature}{Style.RESET_ALL}"
        )

        # breakpoint()

        # Provider-specific configurations
        provider_configs = {
            "google": {"base_delay": 1, "retry_delay": 3, "jitter": 2},
            "openai": {"base_delay": 1, "retry_delay": 3, "jitter": 1},
            "together": {"base_delay": 1, "retry_delay": 3, "jitter": 1},
            "anthropic": {"base_delay": 1, "retry_delay": 2, "jitter": 1},
            "ollama": {"base_delay": 0, "retry_delay": 0, "jitter": 0},
            "sglang": {"base_delay": 0, "retry_delay": 1, "jitter": 0.5},
            "transformers": {"base_delay": 0, "retry_delay": 1, "jitter": 0.5},
            "openrouter": {
                "base_delay": 1,
                "retry_delay": 3,
                "jitter": 1,
            },  # Add OpenRouter configuration
        }

        config = provider_configs[self.provider]

        # print(json.dumps(messages, indent=2, ensure_ascii=False))

        for attempt in range(self.max_retries):
            try:
                # Add retry delay if needed
                if attempt > 0:
                    delay = config["retry_delay"] * attempt + uniform(
                        0, config["jitter"]
                    )
                    print(
                        f"\nRetry attempt {attempt + 1}/{self.max_retries}. Waiting {delay:.2f}s..."
                    )
                    sleep(delay)

                # Get response based on provider
                if self.provider == "openai":
                    if any(
                        f"o{i}" in self.model for i in range(1, 6)
                    ):  # handles o1, o2, o3, o4, o5
                        response = self._call_openai_o1_model(messages)
                    else:
                        api_params = {
                            "model": self.model,
                            "messages": messages,
                            "temperature": temperature,
                        }
                        if response_format:
                            api_params["response_format"] = response_format

                        response = self.client.chat.completions.create(**api_params)
                        response = response.choices[0].message.content
                elif self.provider == "openrouter":
                    # Setup extra headers for OpenRouter if provided
                    extra_headers = {}
                    if self.http_referer:
                        extra_headers["HTTP-Referer"] = self.http_referer
                    if self.x_title:
                        extra_headers["X-Title"] = self.x_title

                    api_params = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                    }
                    if response_format:
                        api_params["response_format"] = response_format
                    if extra_headers:
                        api_params["extra_headers"] = extra_headers

                    response = self.client.chat.completions.create(**api_params)
                    response = response.choices[0].message.content
                elif self.provider == "google" and "meta" in self.model:
                    response = self._call_google_meta_api(messages, temperature)
                elif self.provider == "sglang":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=2048,
                    )
                    response = response.choices[0].message.content
                elif self.provider == "transformers":
                    response = self._call_transformers_model(messages, temperature)
                else:
                    api_params = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                    }
                    if response_format:
                        api_params["response_format"] = response_format

                    response = self.client.chat.completions.create(**api_params)
                    response = response.choices[0].message.content

                # Return based on return_messages flag
                return (response, messages) if return_messages else response

            except Exception as e:
                error_msg = str(e)
                if hasattr(e, "response"):
                    error_msg = f"Error code: {e.status_code} - {error_msg} - Response: {e.response}"

                logging.error(
                    f"BaseAgent: API call failed for {self.provider} (Attempt {attempt + 1}/{self.max_retries})",
                    exc_info=e,
                )

                if attempt == self.max_retries - 1:
                    # hack to prevent Anthropic error 400s from crashing the program
                    if self.provider == "anthropic" and e.status_code == 400:
                        return (
                            ("Sorry, I can't assist with that request.", messages)
                            if return_messages
                            else "Sorry, I can't assist with that request."
                        )
                    if self.provider == "google" and isinstance(
                        e, ResponseValidationError
                    ):
                        return (
                            ("Sorry, I can't assist with that request.", messages)
                            if return_messages
                            else "Sorry, I can't assist with that request."
                        )

                    raise APICallError(
                        f"BaseAgent: Failed to get response from {self.provider}: {error_msg}"
                    )
                continue

    def _ensure_transformers_model_loaded(self, config: Dict) -> None:
        model_path = config["model"]
        if model_path in BaseAgent._transformers_models:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=config.get("trust_remote_code", True),
        )
        load_kwargs = {
            "torch_dtype": "auto",
            "trust_remote_code": config.get("trust_remote_code", True),
        }
        device_map = config.get("device_map", "auto")
        if isinstance(device_map, str) and device_map.lower() in {"", "none", "null"}:
            device_map = None
        if device_map:
            load_kwargs["device_map"] = device_map

        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except Exception:
            load_kwargs.pop("device_map", None)
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            if torch.cuda.is_available():
                model = model.to("cuda")
        model.eval()

        BaseAgent._transformers_tokenizers[model_path] = tokenizer
        BaseAgent._transformers_models[model_path] = model

    def _call_transformers_model(
        self,
        messages: List[Dict],
        temperature: float,
    ) -> str:
        tokenizer = BaseAgent._transformers_tokenizers[self.model]
        model = BaseAgent._transformers_models[self.model]
        if getattr(tokenizer, "chat_template", None):
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if "enable_thinking" in self.config:
                template_kwargs["enable_thinking"] = bool(self.config["enable_thinking"])
            try:
                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                template_kwargs.pop("enable_thinking", None)
                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt += "\nassistant:"

        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            device = next(model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
        except StopIteration:
            pass

        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": int(self.config.get("max_new_tokens", 2048)),
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = float(self.config.get("top_p", 0.9))
        generated = model.generate(**inputs, **generation_kwargs)
        new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _call_google_meta_api(self, messages: List[Dict], temperature: float) -> str:
        """Handle Google-hosted Meta models."""
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)

        response = requests.post(
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/endpoints/openapi/chat/completions",
            headers={
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            },
        )

        if response.status_code != 200:
            raise APICallError(f"Error {response.status_code}: {response.text}")
        return response.json()["choices"][0]["message"]["content"]

    def _call_openai_o1_model(self, messages: List[Dict]) -> str:
        """Handle OpenAI o1 models which only accept user messages."""
        # Warning about message handling
        print(
            "\nWarning: OpenAI o1 model only accepts user messages. System messages will be ignored."
        )

        # Format message for o1 model
        formatted_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": messages[-1][
                            "content"
                        ],  # Just take the last user message
                    }
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model, messages=formatted_messages
        )
        return response.choices[0].message.content


def main():
    """Test different model configurations."""
    test_configs = [
        # OpenAI standard config
        {
            "provider": "openai",
            "model": "gpt-4o",
            "max_retries": 3,
            "temperature": 0,
            # "response_format": {"type": "json_object"}  # Correct format as a dictionary if json response is needed (only for openai models) Make sure you include the word json in some form in the message.
        },
        # SGLang config
        {
            "provider": "sglang",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "port": 30000,
            "max_retries": 3,
            "temperature": 0,
        },
        # OpenAI preview model config
        {
            "provider": "openai",
            "model": "o1-preview",
            "max_retries": 3,
            "temperature": 0,
        },
        # Google standard config
        {
            "provider": "google",
            "model": "gemini-1.5-pro-002",
            "project_id": "mars-lab-429920",
            "location": "us-central1",
            "max_retries": 3,
            "temperature": 0,
        },
        # Google-hosted Meta model config
        {
            "provider": "google",
            "model": "meta/llama-3.1-405b-instruct-maas",
            "project_id": "mars-lab-429920",
            "location": "us-central1",
            "max_retries": 3,
            "temperature": 0,
        },
        # SGLang config
        {
            "provider": "sglang",
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "port": 30000,
            "max_retries": 3,
            "temperature": 0,
        },
        # Ollama config
        {
            "provider": "ollama",
            "model": "llama3.1:latest",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "max_retries": 3,
            "temperature": 0,
        },
        # Anthropic config
        {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "max_retries": 3,
            "temperature": 0,
        },
        # OpenRouter config
        # other model anthropic/claude-3, anthropic/claude-3.7-sonnet, deepseek/deepseek-chat
        {
            "provider": "openrouter",
            "model": "deepseek/deepseek-r1",
            "max_retries": 3,
            "temperature": 0,
            # "http_referer": "https://yourapp.com",  # Optional
            # "x_title": "Your App Name"  # Optional
        },
    ]

    # Test message for all models
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    for config in test_configs:
        print("\n" + "=" * 50)
        print(f"Testing {config['provider']} with model {config['model']}")
        print("=" * 50)

        try:
            agent = BaseAgent(config)
            # Test with message inspection
            response, messages = agent.call_api(
                messages=test_messages,
                temperature=config["temperature"],
                # response_format=config.get('response_format'),  # Pass response_format from config
                return_messages=True,
            )
            print(f"\nMessages sent: {messages}")
            print(f"\nResponse: {response}")
            print("\nTest completed successfully!")

        except Exception as e:
            print(f"Error testing configuration: {str(e)}")

        print("-" * 50)
    breakpoint()


if __name__ == "__main__":
    main()
