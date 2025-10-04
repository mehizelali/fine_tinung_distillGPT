import os
import torch
import warnings
import argparse
from huggingface_hub import login
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
from src.llm_model import Llm_
from src.trainer import train_model
from src.data_loader import get_dataset

warnings.filterwarnings("ignore")

def run_training(
    hf_token: str,
    model_name: str = "distilgpt2",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    output_dir: str = "./outputs",
    seq_len: int = 512,
    num_epochs: int = 5,
    distributed_backend: str = "fsdp",
    dtype: str = "bfloat16",
    optimizer: str = "adamw_torch",
    use_lora: bool = False,
    quantize: str | None = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    learning_rate: float = 3e-4
):   
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"[Rank {local_rank}] Using device: {device}")
    # Login to Hugging Face
    login(hf_token)

    # Load dataset
    dataset = get_dataset(dataset_name, dataset_config)

    # Initialize model wrapper
    lama = Llm_(path=model_name)

    # Map string dtype → torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # Prepare quantization config
    quantization_config = None
    if quantize is not None:
        from transformers import BitsAndBytesConfig
        if quantize == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch_dtype
            )
        elif quantize == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        print(f"Using {quantize} quantization")
    device_map = None
    if quantize in ["8bit", "4bit"]:
        device_map = {"": device}
    # Create base model
    model = lama.create_model(
        attn_implementation="eager",
        dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config
    )

    # Tokenize
    data_collator, tokenizer, datasets = lama.tokenize_dataset(dataset)
    train_set, eval_set = datasets["train"], datasets["test"]

    # LoRA
    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    else:
        lora_config = None

    # Training arguments
    training_args_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=100,
        save_steps=500,
        save_total_limit=1,

        optim=optimizer,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
        save_safetensors=True,
        report_to=["tensorboard"],  
         logging_dir=f"{output_dir}/logs",
    )

    # Precision flags
    if dtype == "bfloat16":
        training_args_kwargs["bf16"] = True
        training_args_kwargs["fp16"] = False
    elif dtype == "float16":
        training_args_kwargs["fp16"] = True
        training_args_kwargs["bf16"] = False
    else:
        training_args_kwargs["fp16"] = False
        training_args_kwargs["bf16"] = False

    # Distributed backend
    if distributed_backend == "fsdp":
        training_args_kwargs.update({
            "fsdp": "full_shard auto_wrap",
            "fsdp_transformer_layer_cls_to_wrap": "GPT2Block",
            "fsdp_config": {
                "activation_checkpointing": True,
                "activation_cpu_offload": False
            },
        })
        print("Using FSDP for distributed training")
    elif distributed_backend == "ddp":
        training_args_kwargs.update({
            "ddp_find_unused_parameters": False
        })
        print("Using DDP for distributed training")
    else:
        print("Running without distributed backend (single GPU/CPU)")

    training_args = TrainingArguments(**training_args_kwargs)

    # Train
    train_model(
        model=model,
        training_args=training_args,
        train_set=train_set,
        eval_set=eval_set,
        tokenizer=tokenizer,
        peft_config=lora_config,
        max_seq_length=seq_len
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--hf_token",
    type=str,
    default="xxxx",  # default token
    help="Hugging Face token for authentication"
)
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--distributed_backend",
        type=str,
        default="ddp",
        choices=["ddp", "fsdp"]
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw_torch",
        choices=["adamw_torch", "adamw_hf", "adamw_apex", "adamw_bnb_8bit"]
    )
    parser.add_argument(
        "--lora_r", type=int, default=8
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16
    )

    args = parser.parse_args()

    #model variants
    variants = [
        {"name": "baseline_lora_quant_4bit", "use_lora": True, "quantize": "4bit"},
        {"name": "baseline_lora_quant", "use_lora": True, "quantize": "8bit"},
        {"name": "baseline_lora", "use_lora": True, "quantize": None},
        {"name": "baseline", "use_lora": False, "quantize": None},
        
        
        
        
    ]

    
    for variant in variants:
        print(f"\n=== Training variant: {variant['name']} ===\n")
        run_training(
            hf_token=args.hf_token,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            output_dir=f"{args.output_dir}/{variant['name']}",
            seq_len=args.seq_len,
            num_epochs=args.num_epochs,
            dtype=args.dtype,
            distributed_backend=args.distributed_backend,
            optimizer=args.optimizer,
            use_lora=variant["use_lora"],
            quantize=variant["quantize"],
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha
        )
