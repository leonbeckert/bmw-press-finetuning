"""Tests for training pipeline — validate config, data format, and helpers.

These tests catch issues before spending time on a GPU training run.
They do NOT test the training loop itself (that's framework code).
Designed to run without GPU or train dependencies (no torch/mlflow imports).
"""

import json
from pathlib import Path

import yaml

CONFIG_PATH = Path("configs/base.yaml")
TRAIN_PATH = Path("data/processed/train.jsonl")
EVAL_PATH = Path("data/processed/eval.jsonl")


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfig:

    def test_config_loads(self):
        config = _load_config(CONFIG_PATH)
        assert isinstance(config, dict)

    def test_required_top_level_keys(self):
        config = _load_config(CONFIG_PATH)
        for key in ["model", "training", "data", "mlflow", "evaluation"]:
            assert key in config, f"Missing top-level key: {key}"

    def test_model_config(self):
        config = _load_config(CONFIG_PATH)
        assert isinstance(config["model"]["name"], str)
        assert config["model"]["torch_dtype"] == "bfloat16"

    def test_training_types(self):
        config = _load_config(CONFIG_PATH)
        tc = config["training"]
        assert isinstance(tc["learning_rate"], float)
        assert isinstance(tc["num_train_epochs"], int)
        assert isinstance(tc["per_device_train_batch_size"], int)
        assert isinstance(tc["packing"], bool)
        assert isinstance(tc["bf16"], bool)

    def test_data_paths_exist(self):
        config = _load_config(CONFIG_PATH)
        assert Path(config["data"]["train_path"]).exists()
        assert Path(config["data"]["eval_path"]).exists()

    def test_sample_prompts(self):
        config = _load_config(CONFIG_PATH)
        prompts = config["evaluation"]["sample_prompts"]
        assert isinstance(prompts, list)
        assert len(prompts) >= 3
        assert all(isinstance(p, str) for p in prompts)


# ---------------------------------------------------------------------------
# Dataset format
# ---------------------------------------------------------------------------

class TestDatasetFormat:

    def test_train_jsonl_has_text_field(self):
        with open(TRAIN_PATH) as f:
            for i, line in enumerate(f):
                article = json.loads(line)
                assert "text" in article, f"Line {i}: missing 'text' field"
                assert len(article["text"]) > 0, f"Line {i}: empty text"

    def test_eval_jsonl_has_text_field(self):
        with open(EVAL_PATH) as f:
            for i, line in enumerate(f):
                article = json.loads(line)
                assert "text" in article, f"Line {i}: missing 'text' field"
                assert len(article["text"]) > 0, f"Line {i}: empty text"

    def test_no_empty_lines(self):
        for path in [TRAIN_PATH, EVAL_PATH]:
            with open(path) as f:
                for i, line in enumerate(f):
                    assert line.strip(), f"{path} line {i}: empty line"

    def test_train_eval_no_overlap(self):
        train_ids = set()
        with open(TRAIN_PATH) as f:
            for line in f:
                train_ids.add(json.loads(line)["article_id"])
        eval_ids = set()
        with open(EVAL_PATH) as f:
            for line in f:
                eval_ids.add(json.loads(line)["article_id"])
        assert train_ids.isdisjoint(eval_ids), "Train and eval sets overlap"


# ---------------------------------------------------------------------------
# Flatten dict helper
# ---------------------------------------------------------------------------

class TestFlattenDict:

    def test_flat(self):
        assert _flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested(self):
        result = _flatten_dict({"model": {"name": "qwen", "size": 0.6}})
        assert result == {"model.name": "qwen", "model.size": 0.6}

    def test_list_values(self):
        result = _flatten_dict({"prompts": ["a", "b"]})
        assert result == {"prompts": "['a', 'b']"}
