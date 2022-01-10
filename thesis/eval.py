import json
import os
import timeit

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.cli import tqdm
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult

from load_data import load_and_cache_examples
from params import (
    device,
    do_lowercase,
    max_answer_length,
    n_best_size,
    null_score_diff_threshold,
    output_dir,
    per_gpu_eval_batch_size,
    version_2,
)
from utils import to_list


def evaluate(model, tokenizer):
    dataset, examples, features = load_and_cache_examples(
        tokenizer, evaluate=True, output_examples=True
    )

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=per_gpu_eval_batch_size
    )

    print("\nEvaluation:")

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            feature_indices = batch[3]
            outputs = model(**inputs, return_dict=False)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                output = [to_list(output[i]) for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    print(
        "  Evaluation done in total {} secs ({} sec per example)".format(
            evalTime, evalTime / len(dataset)
        )
    )

    output_prediction_file = os.path.join(output_dir, "predictions.json")
    output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")

    if version_2:
        output_null_log_odds_file = os.path.join(output_dir, "null_odds.json")
    else:
        output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lowercase,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,
        version_2,
        null_score_diff_threshold,
        tokenizer,
    )

    if output_null_log_odds_file is not None:
        filename = os.path.join(output_dir, "null_odds.json")
        null_odds = json.load(open(filename, "rb"))
    else:
        null_odds = None
    results = squad_evaluate(
        examples,
        predictions,
        no_answer_probs=null_odds,
        no_answer_probability_threshold=null_score_diff_threshold,
    )
    return results
