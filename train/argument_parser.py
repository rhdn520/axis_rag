import argparse

def train_args_parser():
    parser = argparse.ArgumentParser(description="Train AxisRanker Model")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=60, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--l1_lambda", type=float, default=0.0, help="L1 regularization lambda"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=30,
        help="Hidden dimension size for AxisRanker",
    )
    parser.add_argument(
        "--input_dim", type=int, default=1024, help="Input embedding dimension size"
    )
    parser.add_argument(
        "--output_dim", type=int, default=1024, help="Output embedding dimension size"
    )
    parser.add_argument(
        "--hidden_layer_number", type=int, default=2, help="Number of hidden layers"
    )
    parser.add_argument(
        "--add_sigmoid", action="store_true", default=False, help="Whether to add sigmoid activation"
    )
    return parser.parse_args()
