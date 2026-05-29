from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure simple NCCL all-reduce bandwidth.")
    parser.add_argument("--size-mb", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    numel = int(args.size_mb * 1024 * 1024 // 4)
    tensor = torch.ones(numel, device=f"cuda:{local_rank}", dtype=torch.float32)

    for _ in range(args.warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()
    for _ in range(args.iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - start

    bytes_per_rank = tensor.numel() * tensor.element_size()
    alg_bw = bytes_per_rank * args.iters / elapsed / 1e9
    # Ring all-reduce bus bandwidth approximation.
    bus_bw = alg_bw * (2 * (world_size - 1) / world_size)
    if rank == 0:
        print(
            f"world_size={world_size} size_mb={args.size_mb} iters={args.iters} "
            f"elapsed_sec={elapsed:.3f} alg_bw_GBps={alg_bw:.2f} bus_bw_GBps={bus_bw:.2f}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
