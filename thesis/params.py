import torch
"""Parameters"""

#model_name_or_path = 'bert-base-cased'
#model_name_or_path = 'm3hrdadfi/albert-fa-base-v2'
#model_name_or_path = 'HooshvareLab/bert-fa-base-uncased'
model_name_or_path = 'bert-base-multilingual-cased'
output_dir = 'model'
dataset_name = 'Sajad'
version_2 = True
null_score_diff_threshold = 0
max_seq_length = 384
max_query_length = 64
doc_stride = 128
do_lowercase = False
per_gpu_train_batch_size = 12
per_gpu_eval_batch_size = 8
learning_rate = 3e-5
gradient_accumulation_steps = 1
weight_decay = 0
max_grad_norm = 1
num_train_epochs = 2
warmup_steps = 0
n_best_size = 20
max_answer_length = 30
seed = 42
threads = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
"""Parameters"""