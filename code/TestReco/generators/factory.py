from generators.model import (
    ChatGPT4o,
    ChatGPT4o_mini,
    ChatGPT4_turbo,
    ChatGPT_3_5_turbo,
    Claude_3_7_Sonnet_thinking,
    Gemma_2_9B_IT,
    Gemma_2_9B_IT_Free,
    Gemma2_9B_Ollama,  # Add this import
    Llama3_8B,
    Mistral_24B_Instruct,
    ModelBase,
)


def model_factory(model_name: str) -> ModelBase:
    """
    Factory function to create model instances based on the model name.

    Args:
        model_name: The name of the model to create

    Returns:
        An instance of the appropriate model class

    Raises:
        ValueError: If the model name is not recognized
    """
    if model_name == "llama-3-8b":
        print("Using Llama 3 8B")
        return Llama3_8B()
    elif model_name == "gemma-2-9b-it-free":
        print("Using Gemma 2 9B IT Free")
        return Gemma_2_9B_IT_Free()
    elif model_name == "gemma-2-9b-it":
        print("Using Gemma 2 9B IT")
        return Gemma_2_9B_IT()
    elif model_name == "gemma2-9b-ollama":  # Add this case
        print("Using Gemma 2 9B (Ollama - Local)")
        return Gemma2_9B_Ollama()
    elif model_name == "chatgpt-4o":
        print("Using ChatGPT 4o")
        return ChatGPT4o()
    elif model_name == "chatgpt-4o-mini":
        print("Using ChatGPT 4o mini")
        return ChatGPT4o_mini()
    elif model_name == "chatgpt-4-turbo":
        print("Using ChatGPT 4 Turbo")
        return ChatGPT4_turbo()
    elif model_name == "chatgpt-3.5-turbo":
        print("Using ChatGPT 3.5 Turbo")
        return ChatGPT_3_5_turbo()
    elif model_name == "claude-3.7-sonnet-thinking":
        print("Using Claude 3.7 Sonnet thinking")
        return Claude_3_7_Sonnet_thinking()
    elif model_name == "mistral-24b-instruct":
        print("Using Mistral 24B Instruct")
        return Mistral_24B_Instruct()
    else:
        raise ValueError(f"Invalid model name: {model_name}")
