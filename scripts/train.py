"""Fine-tune Qwen3-0.6B on BMW press releases.

Loads cleaned articles, trains with SFTTrainer (packing enabled),
tracks everything in MLflow, and writes metrics + sample generations.

Usage:
    python scripts/train.py [configs/base.yaml]
"""

import json
import math
import os
import sys
import time
from pathlib import Path

# Single-GPU: prevent DataParallel on multi-GPU nodes
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import mlflow
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dict for MLflow param logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def generate_texts(model, tokenizer, prompts: list[str], config: dict) -> list[str]:
    """Generate text for a list of prompts."""
    device = next(model.parameters()).device
    model.eval()
    generations = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=config["evaluation"]["max_new_tokens"],
                temperature=config["evaluation"]["temperature"],
                top_p=config["evaluation"]["top_p"],
                do_sample=True,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        generations.append(text)
    return generations


def measure_latency(model, tokenizer, prompts: list[str], config: dict) -> float:
    """Measure average inference throughput (tokens/sec)."""
    device = next(model.parameters()).device
    model.eval()
    max_new_tokens = config["evaluation"]["max_new_tokens"]
    num_runs = config["evaluation"].get("num_latency_runs", 3)
    throughputs = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Warmup run
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            generated_tokens = output.shape[1] - inputs["input_ids"].shape[1]
            throughputs.append(generated_tokens / elapsed)

    return sum(throughputs) / len(throughputs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml"
    config = load_config(config_path)
    print(f"Config: {config_path}")

    # Seeds
    seed = config["training"]["seed"]
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # MLflow environment (before trainer creation)
    os.environ["MLFLOW_EXPERIMENT_NAME"] = config["mlflow"]["experiment_name"]
    os.environ["MLFLOW_FLATTEN_PARAMS"] = "TRUE"
    mlflow.enable_system_metrics_logging()

    # Tokenizer
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer: {model_name} (vocab_size={tokenizer.vocab_size})")

    # Model
    dtype = getattr(torch, config["model"]["torch_dtype"])
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    print(f"Model: {model_name} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")

    # Base model generations (before fine-tuning)
    prompts = config["evaluation"]["sample_prompts"]
    model.cuda()
    print("Generating base model samples...")
    base_generations = generate_texts(model, tokenizer, prompts, config)

    # Dataset
    dataset = load_dataset("json", data_files={
        "train": config["data"]["train_path"],
        "eval": config["data"]["eval_path"],
    })
    print(f"Dataset: {len(dataset['train'])} train, {len(dataset['eval'])} eval")

    # SFTTrainer
    tc = config["training"]
    sft_config = SFTConfig(
        output_dir=tc["output_dir"],
        max_length=tc["max_seq_length"],
        packing=tc["packing"],
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        per_device_eval_batch_size=tc.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        gradient_checkpointing=tc.get("gradient_checkpointing", False),
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        warmup_ratio=tc["warmup_ratio"],
        weight_decay=tc["weight_decay"],
        max_grad_norm=tc["max_grad_norm"],
        bf16=tc["bf16"],
        logging_steps=tc["logging_steps"],
        eval_strategy=tc["eval_strategy"],
        eval_steps=tc["eval_steps"],
        save_strategy=tc["save_strategy"],
        save_steps=tc["save_steps"],
        save_total_limit=tc["save_total_limit"],
        load_best_model_at_end=tc["load_best_model_at_end"],
        metric_for_best_model=tc["metric_for_best_model"],
        seed=tc["seed"],
        report_to="mlflow",
        run_name="qwen3-full",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=sft_config,
    )

    # Train
    print("Starting training...")
    torch.cuda.reset_peak_memory_stats()
    try:
        trainer.train()
    finally:
        train_peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
        train_peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
        if mlflow.active_run():
            mlflow.end_run()

    # Save model locally
    final_dir = tc["output_dir"] + "/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")

    # ------------------------------------------------------------------
    # Post-training: reopen MLflow run for custom metrics + artifacts
    # ------------------------------------------------------------------
    last_run_id = mlflow.last_active_run().info.run_id

    with mlflow.start_run(run_id=last_run_id):
        mlflow.set_tag("variant", "full")
        mlflow.log_params(flatten_dict(config))
        mlflow.log_dict(config, "config.yaml")

        # Perplexity
        eval_results = trainer.evaluate()
        eval_loss = eval_results["eval_loss"]
        perplexity = math.exp(eval_loss)
        mlflow.log_metric("eval_perplexity", perplexity)
        print(f"Eval loss: {eval_loss:.4f} | Perplexity: {perplexity:.2f}")

        # Inference latency
        print("Measuring inference latency...")
        torch.cuda.reset_peak_memory_stats()
        avg_throughput = measure_latency(model, tokenizer, prompts, config)
        inference_peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
        inference_peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
        mlflow.log_metric("inference_tokens_per_sec", avg_throughput)
        print(f"Inference throughput: {avg_throughput:.1f} tokens/sec")

        # GPU memory (separated by phase)
        mlflow.log_metrics({
            "train_peak_gpu_memory_allocated_mb": train_peak_allocated,
            "train_peak_gpu_memory_reserved_mb": train_peak_reserved,
            "inference_peak_gpu_memory_allocated_mb": inference_peak_allocated,
            "inference_peak_gpu_memory_reserved_mb": inference_peak_reserved,
        })
        print(f"Peak GPU memory — train: {train_peak_allocated:.0f} MB allocated, "
              f"{train_peak_reserved:.0f} MB reserved")
        print(f"Peak GPU memory — inference: {inference_peak_allocated:.0f} MB allocated, "
              f"{inference_peak_reserved:.0f} MB reserved")

        # Fine-tuned generations
        print("Generating fine-tuned samples...")
        finetuned_generations = generate_texts(model, tokenizer, prompts, config)

        mlflow.log_table(
            data={
                "prompt": prompts,
                "base_model": base_generations,
                "finetuned": finetuned_generations,
            },
            artifact_file="sample_generations.json",
        )

        # Log model to MLflow
        try:
            mlflow.transformers.log_model(
                transformers_model={"model": trainer.model, "tokenizer": tokenizer},
                task="text-generation",
                name="model",
            )
        except Exception as e:
            print(f"Warning: mlflow.transformers.log_model failed: {e}")
            print("Skipping MLflow model logging. Model is saved locally.")

    # ------------------------------------------------------------------
    # Write results to disk (for git)
    # ------------------------------------------------------------------
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    metrics = {
        "eval_loss": eval_loss,
        "eval_perplexity": perplexity,
        "inference_tokens_per_sec": avg_throughput,
        "train_peak_gpu_memory_allocated_mb": train_peak_allocated,
        "train_peak_gpu_memory_reserved_mb": train_peak_reserved,
        "inference_peak_gpu_memory_allocated_mb": inference_peak_allocated,
        "inference_peak_gpu_memory_reserved_mb": inference_peak_reserved,
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    samples = [
        {"prompt": p, "base_model": b, "finetuned": ft}
        for p, b, ft in zip(prompts, base_generations, finetuned_generations)
    ]
    with open(results_dir / "sample_generations.json", "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {results_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
