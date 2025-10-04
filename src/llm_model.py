import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model

class Llm_:
    """
    A wrapper class for loading and preparing LLaMA models with quantization, 
    tokenization, and dataset preprocessing utilities.

    Attributes
    ----------
    path : str
        Path or Hugging Face model identifier to load the model and tokenizer.
    tokenizer : AutoTokenizer
        Tokenizer associated with the LLaMA model.
    """

    def __init__(self, path: str):
        """
        Initialize the Llama_ class with a tokenizer.

        Parameters
        ----------
        path : str
            Hugging Face model identifier or local path to the pretrained model.
        """
        self.path = path
     
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, use_auth_token=True)
        
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def create_model(self, attn_implementation: str = 'eager', dtype=torch.float16, device_map='auto', quantization_config=None):
        """
        Load a quantized LLaMA model with causal LM head.

        Parameters
        ----------
        attn_implementation : str, optional
            Attention implementation to use ('eager', 'flash_attention_2', etc.), by default 'eager'.
        dtype : torch.dtype, optional
            Data type for model weights (default: torch.float16).
        device_map : str or dict, optional
            Device mapping for model layers, by default 'auto'.
        quantization_config : BitsAndBytesConfig, optional
            Configuration for quantization. If None, defaults to 8-bit quantization.

        Returns
        -------
        model : AutoModelForCausalLM
            The quantized LLaMA model ready for training or inference.
        """
    
        

        
        model = AutoModelForCausalLM.from_pretrained(
            self.path,
            use_auth_token=True,
            device_map=device_map,
            attn_implementation=attn_implementation,
            quantization_config=quantization_config,
            torch_dtype=dtype
        )
        return model

    def tokenize_dataset(self, dataset, max_length: int = 512):
        """
        Tokenize a dataset for training with the model.

        Parameters
        ----------
        dataset : Dataset
            A Hugging Face Dataset object containing a "text" column.
        max_length : int, optional
            Maximum sequence length for tokenization, by default 512.

        Returns
        -------
        data_collator : DataCollatorWithPadding
            Data collator that dynamically pads inputs for batching.
        tokenizer : AutoTokenizer
            The tokenizer used for tokenization.
        tokenized_dataset : Dataset
            The dataset with tokenized inputs and labels.
        """
        def tokenize_function(examples):
       
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
          
            result["labels"] = result["input_ids"].copy()
            return result


        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        return data_collator, self.tokenizer, tokenized_dataset