import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--dataset', default='citeseer', help='[cora citeseer pubmed cs physics]')
    parser.add_argument('--embedder', default='s_mixup')

    parser.add_argument('--seed', default=0, type=int) 
    parser.add_argument('--n_runs', default=10, type=int) 
    parser.add_argument('--n_epochs', default=500, type=int) 
    parser.add_argument('--hidden_dim', default=16, type=int) 

    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout Rate')
    parser.add_argument('--wd', default=0.0001, type=float, help="weight decay or l2")
    parser.add_argument('--lr', default=0.01, type=float)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--target_ratio", type=float, default=0.3) # {0.1, 0.2, 0.3, 0.4, 0.5}
    parser.add_argument("--edge_ratio", type=float, default=0.1) # {0.1, 0.2, 0.3, 0.4, 0.5}
    parser.add_argument("--eta", type=float, default=0.5) # {0.1, ..., 0.9}
    parser.add_argument("--mixup_start", type=int, default=30)
    
    args = parser.parse_args()

    return args