import os

import numpy as np
from tqdm.notebook import tqdm
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from eval import evaluate
from params import *
from train import do_train_setup, train
from utils import set_seed

if __name__ == "__main__":
    set_seed()

    global_step = ""

    model_file = os.path.join(output_dir, "pytorch_model.bin")
    do_train = not os.path.exists(model_file)
    if do_train:
        model, tokenizer, train_dataset, checkpoint = do_train_setup()
        model.to(device)
        global_step, tr_loss = train(train_dataset, model, tokenizer, checkpoint)
        print(" global_step = {}, average loss = {}".format(global_step, tr_loss))
        print("Saving model checkpoint to {}".format(output_dir))
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    result = {}
    print("Evaluate the last checkpoint")
    config = AutoConfig.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        output_dir, do_lower_case=do_lowercase, use_fast=False
    )
    model = AutoModelForQuestionAnswering.from_pretrained(output_dir, config=config)
    model.to(device)
    eval_result = evaluate(model, tokenizer)
    result = dict((k, v) for k, v in eval_result.items())

    print(result)
    print(
        "\nResult:\n\texact_match: {}\n\tf1: {}".format(result["exact"], result["f1"])
    )
