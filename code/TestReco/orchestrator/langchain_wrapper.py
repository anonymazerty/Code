"""
LangChain wrapper for custom LLM compatibility.

This module provides a wrapper to make custom LLM instances compatible
with LangChain's Runnable interface and other LangChain components.
"""

import logging
from typing import Any, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult, Generation
from pydantic import Field

from generators.model import Message


class LangChainWrapper(BaseLanguageModel):
    """Wrapper to make custom LLM compatible with LangChain."""
    
    custom_llm: Any = Field(description="The custom LLM instance")
    
    def __init__(self, custom_llm):
        super().__init__(custom_llm=custom_llm)
        self.custom_llm = custom_llm
    
    def invoke(self, input, config=None, **kwargs):
        """Implement Runnable interface."""
        # Filter out LangChain-specific parameters that our custom LLM doesn't support
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['tools', 'tool_choice', 'functions', 'function_call']}
        
        # Handle different input types
        if isinstance(input, str):
            # Convert string to messages format
            messages = [Message(role="user", content=input)]
        elif isinstance(input, dict) and "messages" in input:
            # Handle ChatPromptTemplate output
            langchain_messages = input["messages"]
            messages = []
            for msg in langchain_messages:
                if isinstance(msg, SystemMessage):
                    messages.append(Message(role="system", content=msg.content))
                elif isinstance(msg, HumanMessage):
                    messages.append(Message(role="user", content=msg.content))
                else:
                    # Fallback for other message types
                    messages.append(Message(role="user", content=str(msg.content)))
        elif hasattr(input, 'text'):
            # StringPromptValue object
            messages = [Message(role="user", content=input.text)]
        elif isinstance(input, list):
            # List of prompt values
            content = " ".join([str(p.text) if hasattr(p, 'text') else str(p) for p in input])
            messages = [Message(role="user", content=content)]
        else:
            # Fallback
            messages = [Message(role="user", content=str(input))]
        
        # Extract max_tokens from config if available
        max_tokens = None
        if config and "max_tokens" in config:
            max_tokens = config["max_tokens"]
        elif config and "max_tokens" in config.get("configurable", {}):
            max_tokens = config["configurable"]["max_tokens"]
        
        # logging.info(f"Messages: {messages}")
        # Call the custom LLM with max_tokens if specified
        if max_tokens:
            response = self.custom_llm.generate_chat(messages, max_tokens=max_tokens)
        else:
            response = self.custom_llm.generate_chat(messages)
        
        return response
    
    def stream(self, input, config=None, **kwargs):
        """Stream implementation that handles LangChain parameters."""
        # Filter out LangChain-specific parameters that our custom LLM doesn't support
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['tools', 'tool_choice', 'functions', 'function_call']}
        
        response = self.invoke(input, config)
        yield response
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate response for LangChain compatibility."""
        # Filter out LangChain-specific parameters that our custom LLM doesn't support
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['tools', 'tool_choice', 'functions', 'function_call']}
        
        # Convert LangChain messages to our format
        converted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted_messages.append(Message(role="system", content=msg.content))
            elif isinstance(msg, HumanMessage):
                converted_messages.append(Message(role="user", content=msg.content))
            else:
                converted_messages.append(Message(role="user", content=str(msg.content)))
        
        response = self.custom_llm.generate_chat(converted_messages)
        
        # Handle different response types
        if isinstance(response, dict) and "content" in response:
            # Claude thinking model returns {"content": "...", "reasoning": "..."}
            text_content = response["content"]
        elif isinstance(response, str):
            text_content = response
        else:
            # Fallback for other types
            text_content = str(response)
        
        # Create proper Generation object
        generation = Generation(text=text_content)
        return LLMResult(generations=[[generation]])
    
    def generate_prompt(self, prompt, stop=None, callbacks=None, **kwargs):
        """Generate response for prompt."""
        # Handle different prompt types
        if hasattr(prompt, 'text'):
            # StringPromptValue object
            content = prompt.text
        elif isinstance(prompt, str):
            # Simple string
            content = prompt
        elif isinstance(prompt, list):
            # List of prompt values
            content = " ".join([str(p.text) if hasattr(p, 'text') else str(p) for p in prompt])
        else:
            # Fallback
            content = str(prompt)
        
        messages = [Message(role="user", content=content)]
        response = self.custom_llm.generate_chat(messages)
        
        # Handle different response types
        if isinstance(response, dict) and "content" in response:
            # Claude thinking model returns {"content": "...", "reasoning": "..."}
            text_content = response["content"]
        elif isinstance(response, str):
            text_content = response
        else:
            # Fallback for other types
            text_content = str(response)
        
        # Create proper Generation object
        generation = Generation(text=text_content)
        return LLMResult(generations=[[generation]])
    
    def predict(self, text, stop=None, callbacks=None, **kwargs):
        """Predict response for text."""
        messages = [Message(role="user", content=text)]
        response = self.custom_llm.generate_chat(messages)
        return response
    
    def predict_messages(self, messages, stop=None, callbacks=None, **kwargs):
        """Predict response for messages."""
        # Convert LangChain messages to our format
        converted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted_messages.append(Message(role="system", content=msg.content))
            elif isinstance(msg, HumanMessage):
                converted_messages.append(Message(role="user", content=msg.content))
            else:
                converted_messages.append(Message(role="user", content=str(msg.content)))
        
        response = self.custom_llm.generate_chat(converted_messages)
        return response
    
    def agenerate_prompt(self, prompt, stop=None, callbacks=None, **kwargs):
        """Async generate response for prompt."""
        return self.generate_prompt(prompt, stop, callbacks, **kwargs)
    
    def apredict(self, text, stop=None, callbacks=None, **kwargs):
        """Async predict response for text."""
        return self.predict(text, stop, callbacks, **kwargs)
    
    def apredict_messages(self, messages, stop=None, callbacks=None, **kwargs):
        """Async predict response for messages."""
        return self.predict_messages(messages, stop, callbacks, **kwargs) 