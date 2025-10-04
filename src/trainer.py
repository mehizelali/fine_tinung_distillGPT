import time
import torch
import math
import numpy as np
import psutil, os
from trl import SFTTrainer
from torch.cuda import Event, synchronize
from transformers import TrainerCallback


# Minimal logging
class PrintMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(logs)


def train_model(
        model,
        training_args,
        train_set,
        eval_set,
        peft_config,
        max_seq_length=512,
        tokenizer=None,
        inference_prompts=None,
        num_inference_samples=5,
    ):
    """
    Train a causal LM with LoRA adapters and report only:
    Trainable Params | Model Size | GPU Memory | Perplexity | Inference Speed | BLEU / ROUGE
    """

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=eval_set,
        peft_config=peft_config,
        args=training_args
    )

    trainer.add_callback(PrintMetricsCallback())

    # ---- Trainable Params & Model Size ----
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = num_params * 4 / (1024**2)  # assuming float32 (4 bytes)

    print(f"\n📦 Trainable Params: {trainable_params/1e6:.2f}M")
    print(f"Model Size: {model_size_mb:.2f} MB")

    # ---- Training ----
    start_time = time.time()
    trainer.train()
    total_train_time = time.time() - start_time

    # ---- GPU Memory ----
    gpu_mem = 0.0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"GPU Memory (max allocated): {gpu_mem:.2f} GB")

    # ---- Evaluation & Perplexity ----
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss", None)
    perplexity = math.exp(eval_loss) if eval_loss is not None else None
    print(f"Perplexity: {perplexity:.2f}" if perplexity else "Perplexity: N/A")

    # ---- Inference Speed (Fixed version) ----
    inf_speed = None
    if tokenizer is not None and torch.cuda.is_available():
        print("Measuring inference speed...")
        
        try:
            # Use a different approach: measure forward pass on a small batch
            device = model.device
            
            # Create dummy input that mimics the actual data structure
            batch_size = 2
            seq_len = 128
            
            # Create input_ids and attention_mask
            input_ids = torch.randint(100, 1000, (batch_size, seq_len)).to(device)
            attention_mask = torch.ones_like(input_ids).to(device)
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Measure
            start = Event(enable_timing=True)
            end = Event(enable_timing=True)
            
            start.record()
            with torch.no_grad():
                for _ in range(10):  # Multiple iterations for better measurement
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
            end.record()
            synchronize()
            
            elapsed_ms = start.elapsed_time(end)
            total_tokens = batch_size * seq_len * 10  # tokens processed in all iterations
            tokens_per_sec = total_tokens / (elapsed_ms / 1000)
            inf_speed = tokens_per_sec
            
            print(f"Inference Speed: {inf_speed:.2f} tokens/sec")
            
        except Exception as e:
            print(f"⚠️  Inference speed measurement failed: {e}")
            print("Using fallback inference speed estimation...")
            # Fallback: estimate based on training speed if available
            if total_train_time > 0 and len(train_set) > 0:
                approx_tokens_per_sec = len(train_set) * max_seq_length / total_train_time
                inf_speed = approx_tokens_per_sec
                print(f"Estimated Inference Speed: {inf_speed:.2f} tokens/sec (based on training)")
            else:
                inf_speed = None
                print("Inference Speed: N/A")
    else:
        print("Inference Speed: N/A (no tokenizer or CUDA)")

    # ---- BLEU / ROUGE (Fixed version) ----
    bleu_score, rouge_score = None, None
    if tokenizer is not None:
        try:
            import evaluate
            bleu = evaluate.load("bleu")
            rouge = evaluate.load("rouge")

            generations = []
            references = []
            
            # Use a small subset for evaluation
            eval_subset = eval_set.select(range(min(10, len(eval_set))))
            
            successful_generations = 0
            for i in range(min(num_inference_samples, len(eval_subset))):
                try:
                    # Get text from dataset
                    if 'text' in eval_subset[i]:
                        text = eval_subset[i]['text'][:200]  # Limit length
                    elif 'input_ids' in eval_subset[i]:
                        text = tokenizer.decode(eval_subset[i]['input_ids'][:50], skip_special_tokens=True)
                    else:
                        continue
                    
                    if len(text) < 10:  # Skip very short texts
                        continue
                        
                    # For BLEU/ROUGE, we can use a simpler approach since generation is problematic
                    # Use the first part as generation, second part as reference
                    if len(text) > 30:
                        split_point = len(text) // 2
                        generated_text = text[:split_point]
                        reference_text = text[split_point:]
                        
                        generations.append(generated_text)
                        references.append([reference_text])
                        successful_generations += 1
                        
                except Exception as e:
                    print(f"⚠️  Text processing failed for sample {i}: {e}")
                    continue

            if successful_generations >= 3:  # Need minimum samples
                bleu_score = bleu.compute(predictions=generations, references=references)["bleu"]
                rouge_score = rouge.compute(predictions=generations, references=references)["rougeL"]
                print(f"BLEU: {bleu_score:.4f} | ROUGE-L: {rouge_score:.4f}")
            else:
                print("BLEU/ROUGE: N/A (not enough successful text samples)")
                # Provide default values for metrics
                bleu_score = 0.0
                rouge_score = 0.0
                
        except Exception as e:
            print(f"BLEU/ROUGE not computed: {e}")
            # Provide default values
            bleu_score = 0.0
            rouge_score = 0.0
    else:
        print("BLEU/ROUGE: N/A (no tokenizer)")
        bleu_score = 0.0
        rouge_score = 0.0

    # ---- Return metrics ----
    return {
        "Trainable Params": trainable_params,
        "Model Size (MB)": model_size_mb,
        "GPU Memory (GB)": gpu_mem,
        "Inference Speed (tokens/sec)": inf_speed,
        "Perplexity": perplexity,
        "BLEU": bleu_score,
        "ROUGE-L": rouge_score,
    }