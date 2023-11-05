import numpy as np
from datasets import load_metric


def prepare_dataset(batch, feature_extractor):
    audio_arr = batch["array"]
    input = feature_extractor(
        audio_arr, sampling_rate=16000, padding=True, return_tensors="pt"
    )

    batch[INPUT_FIELD] = input.input_values[0]
    batch[LABEL_FIELD] = batch[
        "label"
    ]  # colname MUST be labels as Trainer will look for it by default

    return batch


def compute_metrics(eval_pred):
    # DEFINE EVALUATION METRIC
    compute_accuracy_metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return compute_accuracy_metric.compute(predictions=predictions, references=labels)