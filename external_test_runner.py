"""Run the independent external test across the nine paper models.

The initial supported scenario is the with-blue, all-layer-finetune experiment.
This runner does not train models, run grid search, edit notebooks, or estimate
normalization statistics from the external cohort.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as v2
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from training_toolbox import ResizeWithPadding, replace_classifier


SCRIPT_DIR = Path(__file__).resolve().parent

# =============================================================================
# USER CONFIGURATION
# Edit these values, then run this Python file directly from the IDE.
# =============================================================================
SCENARIO = "without_blue_linear_probe"
TEST_DATASET_PATH = Path(r"C:\cell_autoseg_test_additional_noblue")
MODELS_TO_TEST = [
    "dino",
    "dinov2",
    "dinov3",
    "vit",
    "densenet121",
    "efficientnet_v2_s",
    "resnet50",
    "vgg16",
    "convnext_base",
]
DEVICE_SETTING = "cpu"  # "cuda", "cpu", or "auto"
BATCH_SIZE = 32

CHECKPOINT_DIR = SCRIPT_DIR
OUTPUT_ROOT = SCRIPT_DIR / "external_test_results"
DINOV3_REPO = Path(r"C:\Users\Jiabang Chen\Desktop\dinov3\dinov3")
DINOV3_WEIGHTS = SCRIPT_DIR / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
# =============================================================================


SEED = 42
TEST_PERCENT = 0.15
FOLD_NUM = 5
CLASS_TO_IDX = {"cancer": 0, "normal": 1}
DEFAULT_MODELS = [
    "dino",
    "dinov2",
    "dinov3",
    "vit",
    "densenet121",
    "efficientnet_v2_s",
    "resnet50",
    "vgg16",
    "convnext_base",
]

# Dropout is kept at the training-time value for architectural consistency.
# model.eval() disables it during testing, so it does not change predictions.
DROPOUT = 0.3

BEST_SET_NUM_WITH_BLUE_FINETUNE = {
    "dino": 0,
    "dinov2": 1,
    "dinov3": 3,
    "vit": 6,
    "densenet121": 7,
    "efficientnet_v2_s": 4,
    "resnet50": 5,
    "vgg16": 6,
    "convnext_base": 6,
}

BEST_SET_NUM_WITH_BLUE_LINEAR_PROBE = {
    "dino": 7,
    "dinov2": 6,
    "dinov3": 6,
    "vit": 7,
    "densenet121": 3,
    "efficientnet_v2_s": 7,
    "resnet50": 7,
    "vgg16": 2,
    "convnext_base": 6,
}

BEST_SET_NUM_WITHOUT_BLUE_FINETUNE = {
    "dino": 3,
    "dinov2": 2,
    "dinov3": 3,
    "vit": 0,
    "densenet121": 5,
    "efficientnet_v2_s": 5,
    "resnet50": 4,
    "vgg16": 5,
    "convnext_base": 5,
}

BEST_SET_NUM_WITHOUT_BLUE_LINEAR_PROBE = {
    "dino": 7,
    "dinov2": 7,
    "dinov3": 6,
    "vit": 7,
    "densenet121": 7,
    "efficientnet_v2_s": 6,
    "resnet50": 6,
    "vgg16": 7,
    "convnext_base": 6,
}

# The values are fixed from the original 40x experiment. They must never be
# estimated from the independent external-test cohort.
SCENARIOS: dict[str, dict[str, Any]] = {
    "with_blue_finetune": {
        "finetune": True,
        "without_blue": False,
        "images_mean": [0.0868, 0.0924, 0.0216],
        "images_std": [0.1597, 0.1929, 0.0703],
        "default_test_path": Path(r"C:\cell_autoseg_test_additional"),
        "checkpoint_template": "{model}_best_model_parameter_{set_num}_{fold}.pth",
        "best_set_num": BEST_SET_NUM_WITH_BLUE_FINETUNE,
        "implemented": False,
    },
    "with_blue_linear_probe": {
        "finetune": False,
        "without_blue": False,
        "images_mean": [0.485, 0.456, 0.406],
        "images_std": [0.229, 0.224, 0.225],
        "default_test_path": Path(r"C:\cell_autoseg_test_additional"),
        "checkpoint_template": "{model}_best_model_parameter_{set_num}_{fold}.pth",
        "best_set_num": BEST_SET_NUM_WITH_BLUE_LINEAR_PROBE,
        "implemented": False,
    },
    "without_blue_finetune": {
        "finetune": True,
        "without_blue": True,
        "images_mean": [0.0868, 0.0924, 0.0],
        "images_std": [0.1597, 0.1929, 1.0],
        "default_test_path": Path(r"C:\cell_autoseg_test_additional_noblue"),
        "checkpoint_template": "{model}_best_model_parameter_{set_num}_{fold}.pth",
        "best_set_num": BEST_SET_NUM_WITHOUT_BLUE_FINETUNE,
        "implemented": False,
    },
    "without_blue_linear_probe": {
        "finetune": False,
        "without_blue": True,
        "images_mean": [0.485, 0.456, 0.406],
        "images_std": [0.229, 0.224, 0.225],
        "default_test_path": Path(r"C:\cell_autoseg_test_additional_noblue"),
        "checkpoint_template": "{model}_best_model_parameter_{set_num}_{fold}.pth",
        "best_set_num": BEST_SET_NUM_WITHOUT_BLUE_LINEAR_PROBE,
        "implemented": True,
    },
}

METRIC_NAMES = [
    "accuracy",
    "recall",
    "precision",
    "specificity",
    "auc_roc_macro",
    "auc_roc_micro",
    "auc_roc_weighted",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
]

PREDICTION_FIELDS = [
    "model",
    "fold",
    "sample_index",
    "filename",
    "source_path",
    "ground_truth_index",
    "ground_truth_class",
    "predicted_index",
    "predicted_class",
    "logit_cancer",
    "logit_normal",
    "probability_cancer",
    "probability_normal",
    "checkpoint",
]


class ModifiedTestModel(nn.Module):
    """Match the test-time wrapper used by the classification notebook."""

    def __init__(self, base_model: nn.Module, in_features: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(in_features, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.base_model(inputs)
        return self.head(features)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(
    checkpoint_dir: Path,
    checkpoint_template: str,
    model_name: str,
    set_num: int,
    fold: int,
) -> Path:
    return checkpoint_dir / checkpoint_template.format(
        model=model_name,
        set_num=set_num,
        fold=fold,
    )


def preflight(
    scenario: dict[str, Any],
    test_path: Path,
    models_to_test: list[str],
    batch_size: int,
    checkpoint_dir: Path,
    dinov3_repo: Path,
    dinov3_weights: Path,
) -> list[Path]:
    if TEST_PERCENT != 0.15:
        raise AssertionError(f"TEST_PERCENT must remain 0.15, got {TEST_PERCENT}")
    if not scenario["implemented"]:
        raise NotImplementedError(
            f"Scenario {SCENARIO!r} has normalization constants recorded, but its "
            "checkpoint mapping and external-test path have not been configured yet"
        )
    if batch_size <= 0:
        raise ValueError(f"batch size must be positive, got {batch_size}")
    if not models_to_test:
        raise ValueError("MODELS_TO_TEST must contain at least one model")
    unknown_models = sorted(set(models_to_test) - set(DEFAULT_MODELS))
    if unknown_models:
        raise ValueError(
            f"Unknown MODELS_TO_TEST entries: {unknown_models}; allowed models: {DEFAULT_MODELS}"
        )
    if len(models_to_test) != len(set(models_to_test)):
        raise ValueError(f"MODELS_TO_TEST contains duplicate entries: {models_to_test}")
    if not test_path.is_dir():
        raise FileNotFoundError(f"External-test directory does not exist: {test_path}")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    checkpoint_template = scenario["checkpoint_template"]
    best_set_num = scenario["best_set_num"]
    if not isinstance(checkpoint_template, str) or not checkpoint_template:
        raise ValueError(f"Scenario {SCENARIO!r} has no checkpoint template configured")
    if not isinstance(best_set_num, dict):
        raise ValueError(f"Scenario {SCENARIO!r} has no best-set mapping configured")
    missing_model_mappings = sorted(set(models_to_test) - set(best_set_num))
    if missing_model_mappings:
        raise ValueError(
            f"Scenario {SCENARIO!r} has no best-set mapping for: {missing_model_mappings}"
        )
    missing_checkpoints = []
    selected_checkpoints = []
    for model_name in models_to_test:
        set_num = best_set_num[model_name]
        for fold in range(FOLD_NUM):
            path = checkpoint_path(
                checkpoint_dir,
                checkpoint_template,
                model_name,
                set_num,
                fold,
            )
            selected_checkpoints.append(path)
            if not path.is_file() or path.stat().st_size == 0:
                missing_checkpoints.append(path)
    if missing_checkpoints:
        missing_text = "\n".join(f"  - {path}" for path in missing_checkpoints)
        raise FileNotFoundError(f"Missing or empty checkpoint files:\n{missing_text}")

    if "dinov3" in models_to_test:
        if not dinov3_repo.is_dir():
            raise FileNotFoundError(f"DINOv3 repository does not exist: {dinov3_repo}")
        if not dinov3_weights.is_file():
            raise FileNotFoundError(f"DINOv3 pretrained weights do not exist: {dinov3_weights}")
    return selected_checkpoints


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("external_test_runner")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        output_dir / "external_test.log",
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def build_dataset(
    test_path: Path,
    images_mean: list[float],
    images_std: list[float],
) -> torchvision.datasets.ImageFolder:
    transform = v2.Compose(
        [
            ResizeWithPadding((224, 224)),
            v2.ToTensor(),
            v2.Normalize(mean=images_mean, std=images_std),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(test_path, transform=transform)
    if dataset.class_to_idx != CLASS_TO_IDX:
        raise ValueError(
            f"Expected external-test class mapping {CLASS_TO_IDX}, got {dataset.class_to_idx}"
        )
    if len(dataset) == 0:
        raise ValueError(f"No external-test images were found under {test_path}")
    sample_image, _ = dataset[0]
    if tuple(sample_image.shape) != (3, 224, 224):
        raise ValueError(
            f"Expected transformed image shape (3, 224, 224), got {tuple(sample_image.shape)}"
        )
    return dataset


def build_model(
    model_name: str,
    device: torch.device,
    dinov3_repo: Path,
    dinov3_weights: Path,
) -> nn.Module:
    if model_name == "dino":
        base_model = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vitb16",
            pretrained=True,
        )
        num_features = base_model.embed_dim
    elif model_name == "dinov2":
        base_model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=True,
        )
        num_features = base_model.embed_dim
    elif model_name == "dinov3":
        base_model = torch.hub.load(
            str(dinov3_repo),
            "dinov3_vitb16",
            source="local",
            weights=str(dinov3_weights),
        )
        num_features = base_model.embed_dim
    elif model_name == "vit":
        base_model = torch.hub.load(
            "pytorch/vision",
            "vit_b_16",
            pretrained=True,
        )
        base_model, num_features = replace_classifier(base_model)
    else:
        base_model = torch.hub.load(
            "pytorch/vision",
            model_name,
            pretrained=True,
        )
        base_model, num_features = replace_classifier(base_model)

    return ModifiedTestModel(base_model, num_features).to(device)


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability_normal: np.ndarray,
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    auc_roc = float(roc_auc_score(y_true, probability_normal))
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "auc_roc_macro": auc_roc,
        "auc_roc_micro": auc_roc,
        "auc_roc_weighted": auc_roc,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_fold(
    model_name: str,
    fold: int,
    set_num: int,
    checkpoint: Path,
    model: nn.Module,
    dataset: torchvision.datasets.ImageFolder,
    data_loader: DataLoader,
    device: torch.device,
    prediction_writer: csv.DictWriter,
    prediction_file: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    probability_normal_values: list[float] = []
    sample_index = 0

    logger.info(
        "Starting model=%s fold=%d checkpoint=%s",
        model_name,
        fold,
        checkpoint.name,
    )
    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            outputs_cpu = outputs.detach().cpu()
            probabilities_cpu = probabilities.detach().cpu()
            predictions_cpu = predictions.detach().cpu()
            targets_cpu = targets.detach().cpu()

            for batch_index in range(len(targets_cpu)):
                target_class = int(targets_cpu[batch_index].item())
                predicted_class = int(predictions_cpu[batch_index].item())
                source_path = Path(dataset.samples[sample_index][0])
                logit_cancer = float(outputs_cpu[batch_index, 0].item())
                logit_normal = float(outputs_cpu[batch_index, 1].item())
                probability_cancer = float(probabilities_cpu[batch_index, 0].item())
                probability_normal = float(probabilities_cpu[batch_index, 1].item())

                record = {
                    "model": model_name,
                    "fold": fold,
                    "sample_index": sample_index,
                    "filename": source_path.name,
                    "source_path": str(source_path),
                    "ground_truth_index": target_class,
                    "ground_truth_class": dataset.classes[target_class],
                    "predicted_index": predicted_class,
                    "predicted_class": dataset.classes[predicted_class],
                    "logit_cancer": logit_cancer,
                    "logit_normal": logit_normal,
                    "probability_cancer": probability_cancer,
                    "probability_normal": probability_normal,
                    "checkpoint": checkpoint.name,
                }
                prediction_writer.writerow(record)
                logger.info(
                    "PREDICTION | %s",
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                )

                y_true.append(target_class)
                y_pred.append(predicted_class)
                probability_normal_values.append(probability_normal)
                sample_index += 1
            prediction_file.flush()

    if sample_index != len(dataset):
        raise RuntimeError(
            f"Evaluated {sample_index} samples for {model_name} fold {fold}; "
            f"expected {len(dataset)}"
        )

    metrics = compute_fold_metrics(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        np.asarray(probability_normal_values, dtype=np.float64),
    )
    result: dict[str, Any] = {
        "model": model_name,
        "fold": fold,
        "checkpoint": checkpoint.name,
        "set_num": set_num,
        "number_of_samples": len(dataset),
        **metrics,
    }
    logger.info("FOLD_METRICS | %s", json.dumps(result, separators=(",", ":")))

    return result


def summarize_models(fold_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for model_name in dict.fromkeys(row["model"] for row in fold_results):
        model_rows = [row for row in fold_results if row["model"] == model_name]
        model_set_nums = {row["set_num"] for row in model_rows}
        if len(model_set_nums) != 1:
            raise ValueError(
                f"Inconsistent set_num values for {model_name}: {sorted(model_set_nums)}"
            )
        summary: dict[str, Any] = {
            "model": model_name,
            "set_num": model_rows[0]["set_num"],
            "fold_count": len(model_rows),
            "number_of_samples_per_fold": model_rows[0]["number_of_samples"],
        }
        for metric_name in METRIC_NAMES:
            values = np.asarray(
                [row[metric_name] for row in model_rows],
                dtype=np.float64,
            )
            summary[f"{metric_name}_mean"] = float(values.mean())
            summary[f"{metric_name}_std"] = float(values.std())
        summaries.append(summary)
    return summaries


def main() -> None:
    if SCENARIO not in SCENARIOS:
        raise ValueError(f"Unknown SCENARIO {SCENARIO!r}; allowed scenarios: {sorted(SCENARIOS)}")
    scenario = SCENARIOS[SCENARIO]
    best_set_num = scenario["best_set_num"]
    test_path = TEST_DATASET_PATH.resolve()
    checkpoint_dir = CHECKPOINT_DIR.resolve()
    output_root = OUTPUT_ROOT.resolve()
    dinov3_repo = DINOV3_REPO.resolve()
    dinov3_weights = DINOV3_WEIGHTS.resolve()

    selected_checkpoints = preflight(
        scenario,
        test_path,
        MODELS_TO_TEST,
        BATCH_SIZE,
        checkpoint_dir,
        dinov3_repo,
        dinov3_weights,
    )
    device = resolve_device(DEVICE_SETTING)

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{SCENARIO}_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logger(output_dir)

    run_config = {
        "scenario": SCENARIO,
        "finetune": scenario["finetune"],
        "without_blue": scenario["without_blue"],
        "test_percent_provenance": TEST_PERCENT,
        "seed": SEED,
        "fold_num": FOLD_NUM,
        "dropout": DROPOUT,
        "images_mean": scenario["images_mean"],
        "images_std": scenario["images_std"],
        "normalization_source": "fixed values supplied from the original 40x experiment",
        "test_path": str(test_path),
        "checkpoint_dir": str(checkpoint_dir),
        "models": MODELS_TO_TEST,
        "best_set_num": {model: best_set_num[model] for model in MODELS_TO_TEST},
        "batch_size": BATCH_SIZE,
        "device": str(device),
        "selected_checkpoints": [str(path) for path in selected_checkpoints],
        "class_to_idx": CLASS_TO_IDX,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "normalization.json").write_text(
        json.dumps(
            {
                "images_mean": scenario["images_mean"],
                "images_std": scenario["images_std"],
                "source": run_config["normalization_source"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset = build_dataset(
        test_path,
        scenario["images_mean"],
        scenario["images_std"],
    )
    class_counts = {
        class_name: dataset.targets.count(class_index)
        for class_name, class_index in dataset.class_to_idx.items()
    }
    logger.info("RUN_CONFIG | %s", json.dumps(run_config, separators=(",", ":")))
    logger.info(
        "Loaded external test dataset: samples=%d class_counts=%s class_to_idx=%s",
        len(dataset),
        class_counts,
        dataset.class_to_idx,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    fold_results: list[dict[str, Any]] = []
    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as prediction_file:
        prediction_writer = csv.DictWriter(
            prediction_file,
            fieldnames=PREDICTION_FIELDS,
        )
        prediction_writer.writeheader()

        for model_name in MODELS_TO_TEST:
            model = build_model(
                model_name,
                device,
                dinov3_repo,
                dinov3_weights,
            )
            set_num = best_set_num[model_name]
            for fold in range(FOLD_NUM):
                checkpoint = checkpoint_path(
                    checkpoint_dir,
                    scenario["checkpoint_template"],
                    model_name,
                    set_num,
                    fold,
                )
                fold_result = evaluate_fold(
                    model_name,
                    fold,
                    set_num,
                    checkpoint,
                    model,
                    dataset,
                    data_loader,
                    device,
                    prediction_writer,
                    prediction_file,
                    logger,
                )
                fold_results.append(fold_result)
                write_csv(output_dir / "fold_metrics.csv", fold_results)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    model_summaries = summarize_models(fold_results)
    write_csv(output_dir / "model_summary.csv", model_summaries)
    for summary in model_summaries:
        logger.info("MODEL_SUMMARY | %s", json.dumps(summary, separators=(",", ":")))
    logger.info("External test complete. Results: %s", output_dir)


if __name__ == "__main__":
    main()
