import torch
import numpy as np
import yaml
import time
from misc.utils import get_logger
from argument import parse_args

from models.s_mixup import s_mixup

def main():
    torch.set_num_threads(4)
    args = parse_args()

    with open(f"hyperparameter.yaml", "r") as f:
        hyperparams = yaml.safe_load(f)
        dataset = args.dataset
        if dataset in hyperparams:
            for k, v in hyperparams[dataset].items():
                setattr(args, k, v)
    
    logger = get_logger(args)    
    embedder = s_mixup(args)
    start_time = time.time()
    result = embedder.train()

    time_spent = time.time() - start_time
    logging_str = f'** S-Mixup\t\t** Train acc : {np.mean(result.acc_train_max)*100:.2f}+-{np.std(result.acc_train_max)*100:.2f}\t/ Test acc : {np.mean(result.best_acc)*100:.2f}+-{np.std(result.best_acc)*100:.2f} ({time_spent:.4f})'
    logger.info(logging_str)
    logger.info('')
        
if __name__ == "__main__":
    main()