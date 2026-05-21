from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def load_matching_weights(
    module: torch.nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Load only checkpoint tensors that exactly match the target module shapes."""
    target_state = module.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped_missing = 0
    skipped_shape = 0
    matched_params = 0

    for key, tensor in source_state.items():
        target_tensor = target_state.get(key)
        if target_tensor is None:
            skipped_missing += 1
            continue
        if tuple(target_tensor.shape) != tuple(tensor.shape):
            skipped_shape += 1
            continue
        matched[key] = tensor
        matched_params += int(tensor.numel())

    module.load_state_dict(matched, strict=False)
    return {
        "matched_tensors": len(matched),
        "matched_params": matched_params,
        "target_tensors": len(target_state),
        "target_params": sum(int(tensor.numel()) for tensor in target_state.values()),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "new_or_unmatched_tensors": len(target_state) - len(matched),
    }


def format_partial_load_report(prefix: str, stats: Mapping[str, Any]) -> str:
    matched_params = int(stats["matched_params"])
    target_params = max(1, int(stats["target_params"]))
    matched_pct = matched_params * 100.0 / target_params
    return (
        f"{prefix} partial_init matched_tensors={stats['matched_tensors']}/{stats['target_tensors']} "
        f"matched_params={matched_params}/{target_params} ({matched_pct:.2f}%) "
        f"skipped_missing={stats['skipped_missing']} skipped_shape={stats['skipped_shape']} "
        f"new_or_unmatched_tensors={stats['new_or_unmatched_tensors']}"
    )
