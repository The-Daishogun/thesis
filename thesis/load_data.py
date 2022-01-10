from transformers.data.processors.squad import (
    SquadResult,
    SquadV1Processor,
    SquadV2Processor,
)
from params import (
    version_2,
    max_seq_length,
    doc_stride,
    max_query_length,
    threads,
    input_prcnt,
)
import os
from transformers import squad_convert_examples_to_features
import torch


def load_and_cache_examples(tokenizer, evaluate=False, output_examples=False):
    input_dir = "data"
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}%".format("dev" if evaluate else "train", input_prcnt),
    )

    if os.path.exists(cached_features_file):
        print("Loading features from cached file {}".format(cached_features_file))
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        print("Creating features from dataset file at {}".format(input_dir))
        processor = SquadV2Processor() if version_2 else SquadV1Processor()
        if evaluate:
            examples = processor.get_dev_examples(input_dir, filename="dev.json")
        else:
            examples = processor.get_train_examples(input_dir, filename="train.json")
        examples_len = len(examples)
        used_examples = int(examples_len * (input_prcnt / 100))
        print(f"Using {used_examples} of {examples_len} examples")
        features, dataset = squad_convert_examples_to_features(
            examples=examples[:used_examples],
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=threads,
        )
        print("Saving features into cached file {}".format(cached_features_file))
        torch.save(
            {"features": features, "dataset": dataset, "examples": examples},
            cached_features_file,
        )

    if output_examples:
        return dataset, examples, features
    return dataset
