# Large Language Model Evaluation

This repository contains evaluation results for different training variants of a large language model (LLM). The table below summarizes key metrics including model size, trainable parameters, evaluation performance, inference speed, and GPU usage.  

| Model Name             | Trainable Params | Model Size | Eval Loss | Perplexity | Mean Token Accuracy | Eval Samples/sec | Inference Speed (tokens/sec) | GPU Memory (GB) |
|------------------------|-----------------|------------|-----------|------------|------------------|-----------------|-----------------------------|----------------|
| **base_lora_4Bit**   | 0.15M           | 232.03 MB  | 0.4632    | 1.59       | 0.9189           | 632.88          | 16,197 (estimated)          | 10.31          |
| **baseline_lora_8bit**| 0.15M           | 313.03 MB  | 0.4603    | 1.58       | 0.9192           | 529.13          | 21,061                       | 11.46          |
| **baseline_lora**      | 0.15M           | 313.03 MB  | 0.4597    | 1.58       | 0.9193           | 674.47          | 48,502–48,784                | 11.46          |
| **baseline**           | 81.91M          | 312.47 MB  | 0.4427    | 1.56       | 0.9216           | 671.70          | 50,856–51,251                | 11.46          |

## Key Observations
- **Perplexity** is very low across all models → high prediction accuracy for tokens.  
- **Mean token accuracy** is consistently high (~91–92%), with `baseline` performing best.  
- **Inference speed** increases with optimized DDP and full-precision models. LoRA variants show lower speed due to quantization and reduced parameter training.  
- **GPU memory usage** is similar (~11–11.5 GB) for larger models, slightly lower for quantized variants.  
- **BLEU/ROUGE metrics** were not computed due to insufficient text samples.  

## Notes
- `base_lora_4Bit` → LoRA + 4-bit quantization, smallest model size.  
- `baseline_lora_8Bit` → LoRA + 8-bit quantization.  
- `baseline_lora` → LoRA without quantization, uses DDP for distributed training.  
- `baseline` → Full model with 81.91M trainable parameters, highest performance and inference speed.  
