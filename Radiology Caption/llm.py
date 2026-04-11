from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig

def get_llm(model_name, access_token=None, use_peft=False):
    """
    Retrieves a language model from the Hugging Face Hub and optionally applies PEFT for LoRA-based fine-tuning.

    Parameters:
    - model_name: str, the name of the language model to load.
    - access_token: str, optional access token for authentication with Hugging Face Hub.
    - use_peft: bool, whether to apply PEFT using LoRA (default: False).

    Returns:
    - tokenizer: Tokenizer associated with the language model.
    - model: Pre-trained language model.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)

    # Apply PEFT if specified
    if use_peft:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM"
        )

        initial_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters before PEFT: {initial_trainable_params}")

        model.add_adapter(peft_config)

        updated_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        percent_trainable = (updated_trainable_params / initial_trainable_params) * 100
        print(f"Trainable Parameters after PEFT: {updated_trainable_params} ({percent_trainable:.2f}%)")
    else:
        # Freeze all parameters for inference mode
        for param in model.parameters():
            param.requires_grad = False

    return tokenizer, model

def get_hidden_size(tokenizer, model):
    """
    Determines the hidden size of the model's input embeddings.

    Parameters:
    - tokenizer: Tokenizer used with the model.
    - model: Pre-trained language model.

    Returns:
    - int, the hidden size of the input embeddings.
    """
    input_ids = tokenizer.encode('Boop', add_special_tokens=True, return_tensors="pt")
    embeddings = model.get_input_embeddings()(input_ids)
    return embeddings.size(-1)

if __name__ == '__main__':
    """
    Script entry point for testing the LLM functions.
    """
    # Load GPT-2 model and tokenizer for testing
    tokenizer, model = get_llm(
        "openai-community/gpt2", 
        access_token='GET_YOUR_OWN_TOKEN_FROM_HUGGINGFACE'
    )

    # Test text generation
    prompt = "Hiiii, what's up?"
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    generated_ids = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_length=30
    )
    output_text = tokenizer.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    print(f"Generated Response: {output_text}")

    # Test hidden size retrieval
    hidden_size = get_hidden_size(tokenizer, model)
    print(f"Hidden Size of Model Embeddings: {hidden_size}")
