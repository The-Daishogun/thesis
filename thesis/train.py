import os

import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm.cli import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from load_data import load_and_cache_examples
from params import (
    device,
    do_lowercase,
    gradient_accumulation_steps,
    learning_rate,
    max_grad_norm,
    model_name_or_path,
    n_gpu,
    num_train_epochs,
    output_dir,
    per_gpu_train_batch_size,
    warmup_steps,
    weight_decay,
)
from utils import set_seed


def do_train_setup():
    checkpoints_dir = filter(
        lambda x: x.startswith("checkpoint_"), os.listdir(output_dir)
    )
    checkpoint = max(
        map(lambda x: int(x[x.find("_") + 1 :]), checkpoints_dir), default=0
    )
    if checkpoint == 0:
        config = AutoConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, do_lower_case=do_lowercase, use_fast=False
        )
        train_dataset = load_and_cache_examples(
            tokenizer, evaluate=False, output_examples=False
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path, config=config
        )
    else:
        checkpoint_output_dir = os.path.join(
            output_dir, "checkpoint_{}".format(checkpoint)
        )
        config = AutoConfig.from_pretrained(checkpoint_output_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_output_dir, do_lower_case=do_lowercase, use_fast=False
        )
        train_dataset = load_and_cache_examples(
            tokenizer, evaluate=False, output_examples=False
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            checkpoint_output_dir, config=config
        )
    return model, tokenizer, train_dataset, checkpoint


def train(train_dataset, model, tokenizer, start_epoch):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=per_gpu_train_batch_size
    )

    t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    if os.path.isfile(os.path.join(output_dir, "optimizer.pt")) and os.path.isfile(
        os.path.join(output_dir, "scheduler.pt")
    ):
        optimizer.load_state_dict(torch.load(os.path.join(output_dir, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(output_dir, "scheduler.pt")))

    print("\nTraining:")

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed()

    for epoch_idx in range(num_train_epochs):
        if epoch_idx < start_epoch:
            continue
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration in epoch {}".format(epoch_idx + 1)
        )
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
                "is_impossibles": batch[5],
            }
            outputs = model(**inputs)
            loss = outputs[0]
            if n_gpu > 1:
                loss = loss.mean()
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
        checkpoint_output_dir = os.path.join(
            output_dir, "checkpoint_{}".format(epoch_idx + 1)
        )
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(checkpoint_output_dir)
        tokenizer.save_pretrained(checkpoint_output_dir)
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    return global_step, tr_loss / global_step
