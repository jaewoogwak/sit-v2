import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import TVRDataset
from model import GlobalPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Train global predictor on embedding sequences.")
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    dataset = TVRDataset(
        jsonl_path=args.jsonl_path,
        h5_path=args.h5_path,
        seq_len=args.seq_len,
        split=args.split,
        split_ratio=args.split_ratio,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GlobalPredictor(
        feature_dim=args.feature_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for batch in loader:
            seq, target = batch
            seq = seq.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            _, pred = model(seq)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * seq.size(0)

        avg_loss = epoch_loss / max(1, len(dataset))
        elapsed = time.time() - start_time
        print(f"epoch {epoch} loss {avg_loss:.6f} time {elapsed:.2f}s")

        ckpt_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )


if __name__ == "__main__":
    main()

