from __future__ import annotations

from typing import Callable

import numpy as np

from services.combination_service import combo_to_mask, overlap_count
from services.scoring_service import score_sample


def build_portfolio(
    matrix: np.ndarray,
    *,
    ticket_count: int = 5,
    sample_size: int = 20000,
    seed: int = 42,
    max_overlap: int = 9,
    weights: dict | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[dict]:
    """
    Selecciona tickets con buen puntaje y baja similitud (máximo overlap).
    """
    if ticket_count < 1:
        raise ValueError("ticket_count debe ser >= 1")

    candidates = score_sample(
        matrix,
        sample_size=sample_size,
        seed=seed,
        weights=weights,
        top_k=max(ticket_count * 40, 100),
        progress_cb=progress_cb,
        should_cancel=should_cancel,
    )

    selected: list[dict] = []
    selected_masks: list[int] = []

    for item in candidates:
        if should_cancel and should_cancel():
            break
        mask = int(item["mask"])
        if any(overlap_count(mask, m) > max_overlap for m in selected_masks):
            continue
        selected.append(item)
        selected_masks.append(mask)
        if len(selected) >= ticket_count:
            break

    # Si no alcanza diversidad, completar con los mejores restantes
    if len(selected) < ticket_count:
        for item in candidates:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= ticket_count:
                break

    # Cobertura y similitud intra-portafolio
    all_nums = set()
    for item in selected:
        all_nums.update(item["numbers"])
    coverage = len(all_nums) / 25.0

    for i, item in enumerate(selected):
        if len(selected) == 1:
            sim = 0.0
        else:
            overlaps = []
            mask_i = int(item["mask"])
            for j, other in enumerate(selected):
                if i == j:
                    continue
                overlaps.append(overlap_count(mask_i, int(other["mask"])) / 14.0)
            sim = float(np.mean(overlaps)) if overlaps else 0.0
        item["rank"] = i + 1
        item["coverage"] = float(coverage)
        item["similarity"] = sim

    return selected
