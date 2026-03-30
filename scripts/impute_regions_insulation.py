#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import norm
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize



def resolve_higashi_root(script_path: Path) -> Path:
    for parent in script_path.resolve().parents:
        if (parent / "higashi" / "Higashi_wrapper.py").exists():
            return parent
    raise RuntimeError(f"could not resolve Higashi root from {script_path}")


HIGASHI_ROOT = resolve_higashi_root(Path(__file__))
REPO_ROOT = HIGASHI_ROOT.parent
if str(HIGASHI_ROOT) not in sys.path:
    sys.path.insert(0, str(HIGASHI_ROOT))

from higashi.Higashi_wrapper import Higashi  # noqa: E402
import higashi.Higashi_wrapper as higashi_wrapper_module  # noqa: E402
import higashi.Higashi_backend.Modules as backend_modules  # noqa: E402
from higashi.Higashi_backend.Modules import GraphSageEncoder_with_weights  # noqa: E402
from higashi.Impute import skip_start_end  # noqa: E402
from higashi.Higashi_analysis.Higashi_analysis import sqrt_norm  # noqa: E402
from higashi.Higashi_analysis.Higashi_TAD import insulation_score  # noqa: E402


@dataclass(frozen=True)
class RegionSpec:
    index: int
    group_name: str
    name: str
    original_chrom: str
    chrom: str
    bed_start_bp: int
    bed_end_bp: int
    target_start_bin: int
    target_end_bin: int
    context_start_bin: int
    context_end_bin: int
    chrom_index: int
    chrom_offset: int
    status: str
    coordinates: np.ndarray
    samples: np.ndarray
    target_start_idx: int
    target_end_idx: int
    bin_chrom: np.ndarray
    bin_start: np.ndarray
    bin_end: np.ndarray
    bad_bins: np.ndarray
    discard_rows: np.ndarray


@dataclass(frozen=True)
class ResolvedModel:
    mode: str
    model_path: Path
    weighted_info_path: Path | None
    stage: int
    is_full_model: bool


@dataclass(frozen=True)
class ChromRuntimeSpec:
    chrom: str
    chrom_index: int
    chrom_offset: int
    target_local_bin_ids: np.ndarray
    target_global_bin_ids: np.ndarray


def log(message: str) -> None:
    print(f"[impute_regions_insulation] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Impute Higashi contacts and compute insulation scores for BED regions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to the Higashi config JSON.")
    parser.add_argument("--bed", required=True, help="Path to a BED file of target regions.")
    parser.add_argument("--output", required=True, help="Output HDF5 path.")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=("auto", "neighbor", "no-neighbor"),
        help="Imputation mode to use.",
    )
    parser.add_argument(
        "--window-ins",
        type=int,
        default=500_000,
        help="Insulation window size in base pairs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=100_000,
        help="Batch size passed to model.predict().",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def resolve_runtime_device(device_arg: str) -> tuple[str, torch.device]:
    if device_arg == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"
    else:
        device_str = device_arg

    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested device {device_str} but CUDA is not available")
        if ":" in device_str:
            gpu_id = int(device_str.split(":", 1)[1])
        else:
            gpu_id = 0
            device_str = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        runtime_device = torch.device(device_str)
    elif device_str == "cpu":
        runtime_device = torch.device("cpu")
    else:
        raise RuntimeError(f"unsupported device: {device_arg}")

    backend_modules.device = runtime_device
    higashi_wrapper_module.device = runtime_device

    def _patched_get_free_gpu(num: int = 1, change_cur: bool = True):
        if runtime_device.type == "cuda":
            if change_cur:
                torch.cuda.set_device(runtime_device.index or 0)
            if num == 1:
                return str(runtime_device)
            return np.array([runtime_device.index or 0] * num)
        return "cpu"

    higashi_wrapper_module.get_free_gpu = _patched_get_free_gpu
    return str(runtime_device), runtime_device


def load_config(config_path: Path) -> dict:
    with config_path.open() as fh:
        return json.load(fh)


def resolve_config_paths(config_path: Path, config: dict) -> dict:
    normalized = dict(config)
    for key, value in config.items():
        if not isinstance(value, str):
            continue
        if not key.endswith(("_dir", "_path")):
            continue
        if value == "":
            continue
        path = Path(value)
        if not path.is_absolute():
            normalized[key] = str((config_path.parent / path).resolve())
    return normalized


def load_counts(temp_dir: Path, chrom_list: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    with h5py.File(temp_dir / "node_feats.hdf5", "r") as input_f:
        num = np.array(input_f["num"])
    num_list = np.cumsum(num)
    chrom_bin_counts = {chrom: int(num[idx + 1]) for idx, chrom in enumerate(chrom_list)}
    return num, num_list, chrom_bin_counts


def find_first_file(base_dirs: Iterable[Path], patterns: Iterable[str]) -> Path | None:
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for pattern in patterns:
            matches = sorted(base_dir.glob(pattern))
            if matches:
                return matches[0]
    return None


def read_cell_map(cell_map_path: Path) -> dict:
    cell_id: list[int] = []
    cell_label: list[str] = []
    with cell_map_path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if header[:2] != ["cell_id", "cell_label"]:
            raise RuntimeError(f"unsupported cell_map header in {cell_map_path}")
        for line in fh:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            cell_id.append(int(fields[0]))
            cell_label.append(fields[1])
    return {
        "cell_id": cell_id,
        "cell_label": cell_label,
        "cell_name_higashi": list(cell_label),
    }


def synthesize_label_info(cell_count: int) -> dict:
    labels = [f"cell_{idx}" for idx in range(cell_count)]
    return {
        "cell_id": list(range(cell_count)),
        "cell_label": labels,
        "cell_name_higashi": list(labels),
    }


def load_label_info(config_path: Path, config: dict, cell_count: int) -> dict:
    candidate_dirs = []
    data_dir_value = config.get("data_dir")
    if data_dir_value:
        candidate_dirs.append(Path(data_dir_value))
    candidate_dirs.append(config_path.parent)
    candidate_dirs.append(REPO_ROOT / "data")

    label_info_path = find_first_file(candidate_dirs, ("label_info.pickle", "*.label_info.pickle"))
    if label_info_path is not None:
        log(f"loading cell metadata from {label_info_path}")
        with label_info_path.open("rb") as fh:
            label_info = pickle.load(fh)
    else:
        cell_map_path = find_first_file(candidate_dirs, ("cell_map.tsv", "*.cell_map.tsv"))
        if cell_map_path is not None:
            log(f"loading cell metadata from {cell_map_path}")
            label_info = read_cell_map(cell_map_path)
        else:
            log("no label_info or cell_map found; synthesizing generic cell labels")
            label_info = synthesize_label_info(cell_count)

    if "cell_label" not in label_info:
        if "cell_name_higashi" in label_info:
            label_info["cell_label"] = list(label_info["cell_name_higashi"])
        else:
            raise RuntimeError("label_info is missing both cell_label and cell_name_higashi")
    if "cell_name_higashi" not in label_info:
        label_info["cell_name_higashi"] = list(label_info["cell_label"])
    if "cell_id" not in label_info:
        label_info["cell_id"] = list(range(len(label_info["cell_label"])))

    if len(label_info["cell_id"]) != cell_count:
        raise RuntimeError(
            f"label_info cell count mismatch: expected {cell_count}, found {len(label_info['cell_id'])}"
        )

    for key in ("batch_id", "library_id"):
        if key in config and config[key] not in label_info:
            raise RuntimeError(
                f"config requires label_info key {config[key]!r}, but it was not found in resolved metadata"
            )

    return label_info


def write_runtime_config(original_config: dict, label_info: dict) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    temp_data_dir = tempfile.TemporaryDirectory(prefix="higashi_runtime_data_")
    data_dir_path = Path(temp_data_dir.name)
    with (data_dir_path / "label_info.pickle").open("wb") as fh:
        pickle.dump(label_info, fh)
    with (data_dir_path / "cell_map.tsv").open("w") as fh:
        fh.write("cell_id\tcell_label\n")
        for cell_id, cell_label in zip(label_info["cell_id"], label_info["cell_label"]):
            fh.write(f"{cell_id}\t{cell_label}\n")

    runtime_config = dict(original_config)
    runtime_config["data_dir"] = str(data_dir_path)
    fd, runtime_config_path = tempfile.mkstemp(prefix="higashi_runtime_config_", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(runtime_config, fh)
    return Path(runtime_config_path), temp_data_dir


def resolve_model_artifacts(temp_dir: Path, mode: str) -> ResolvedModel:
    model_dir = temp_dir / "model"
    stage2_model = model_dir / "model.chkpt_stage2_model"
    stage3_model = model_dir / "model.chkpt_stage3_model"
    stage2_ckpt = model_dir / "model.chkpt_stage2"
    stage3_ckpt = model_dir / "model.chkpt_stage3"
    weighted_info = temp_dir / "weighted_info.npy"
    has_weighted_info = weighted_info.exists()

    log(
        "checking model artifacts in "
        f"{model_dir}: stage2_model={stage2_model.exists()}, "
        f"stage2_ckpt={stage2_ckpt.exists()}, "
        f"stage3_model={stage3_model.exists()}, "
        f"stage3_ckpt={stage3_ckpt.exists()}, "
        f"weighted_info={has_weighted_info}"
    )

    if mode == "auto":
        if stage3_model.exists() and has_weighted_info:
            resolved = ResolvedModel("neighbor", stage3_model, weighted_info, 3, True)
            log(f"auto-selected stage 3 full model: {resolved.model_path}")
            return resolved
        if stage3_ckpt.exists() and has_weighted_info:
            resolved = ResolvedModel("neighbor", stage3_ckpt, weighted_info, 3, False)
            log(f"auto-selected stage 3 raw checkpoint: {resolved.model_path}")
            return resolved
        if stage2_model.exists():
            resolved = ResolvedModel("no-neighbor", stage2_model, None, 2, True)
            log(f"auto-selected stage 2 full model: {resolved.model_path}")
            return resolved
        if stage2_ckpt.exists():
            resolved = ResolvedModel("no-neighbor", stage2_ckpt, None, 2, False)
            log(f"auto-selected stage 2 raw checkpoint: {resolved.model_path}")
            return resolved
    elif mode == "neighbor":
        if stage3_model.exists() and has_weighted_info:
            resolved = ResolvedModel("neighbor", stage3_model, weighted_info, 3, True)
            log(f"using requested stage 3 full model: {resolved.model_path}")
            return resolved
        if stage3_ckpt.exists() and has_weighted_info:
            resolved = ResolvedModel("neighbor", stage3_ckpt, weighted_info, 3, False)
            log(f"using requested stage 3 raw checkpoint: {resolved.model_path}")
            return resolved
        missing = "weighted_info.npy" if not has_weighted_info else "stage3 model/checkpoint"
        raise RuntimeError(f"neighbor mode requested but {missing} is missing in {temp_dir}")
    elif mode == "no-neighbor":
        if stage2_model.exists():
            resolved = ResolvedModel("no-neighbor", stage2_model, None, 2, True)
            log(f"using requested stage 2 full model: {resolved.model_path}")
            return resolved
        if stage2_ckpt.exists():
            resolved = ResolvedModel("no-neighbor", stage2_ckpt, None, 2, False)
            log(f"using requested stage 2 raw checkpoint: {resolved.model_path}")
            return resolved
        raise RuntimeError(f"no-neighbor mode requested but stage2 model/checkpoint is missing in {temp_dir}")
    else:
        raise RuntimeError(f"unsupported mode: {mode}")

    raise RuntimeError(
        "no imputation model found: stage3 requires model.chkpt_stage3[_model] plus weighted_info.npy; "
        "stage2 requires model.chkpt_stage2[_model]"
    )


def rebuild_stage2_runtime_model(
    config_path: Path,
    config: dict,
    label_info: dict,
    checkpoint_path: Path,
    runtime_device: str,
    torch_device: torch.device,
) -> torch.nn.Module:
    runtime_config_path, temp_data_dir = write_runtime_config(config, label_info)
    try:
        log(
            f"rebuilding stage {2 if 'stage2' in checkpoint_path.name else 3} runtime model from raw checkpoint "
            f"{checkpoint_path}"
        )
        log(f"temporary runtime config: {runtime_config_path}")
        higashi = Higashi(str(runtime_config_path))
        higashi.prep_model()
        node_embedding = higashi.node_embedding_init
        stage2_dynamic = GraphSageEncoder_with_weights(
            features=node_embedding,
            linear_features=node_embedding,
            feature_dim=higashi.dimensions,
            embed_dim=higashi.dimensions,
            num_sample=8,
            gcn=False,
            num_list=higashi.num_list,
            transfer_range=0,
            start_end_dict=higashi_wrapper_module.start_end_dict,
            pass_pseudo_id=False,
            remove=True,
            pass_remove=False,
        ).to(torch_device, non_blocking=True)
        higashi.higashi_model.encode1.dynamic_nn = stage2_dynamic
        checkpoint = torch.load(checkpoint_path, map_location=runtime_device)
        higashi.higashi_model.load_state_dict(checkpoint["model_link"])
        return higashi.higashi_model
    finally:
        runtime_config_path.unlink(missing_ok=True)
        temp_data_dir.cleanup()


def load_runtime_model(
    config_path: Path,
    config: dict,
    label_info: dict,
    resolved_model: ResolvedModel,
    runtime_device: str,
    torch_device: torch.device,
) -> torch.nn.Module:
    if resolved_model.is_full_model:
        log(f"loading serialized model from {resolved_model.model_path}")
        model = torch.load(resolved_model.model_path, map_location=runtime_device)
    else:
        model = rebuild_stage2_runtime_model(
            config_path=config_path,
            config=config,
            label_info=label_info,
            checkpoint_path=resolved_model.model_path,
            runtime_device=runtime_device,
            torch_device=torch_device,
        )
    model = model.to(torch_device)
    model.eval()
    if not isinstance(model.encode1.dynamic_nn, GraphSageEncoder_with_weights):
        raise RuntimeError("resolved model is not an imputation-ready stage2/stage3 model")
    log(
        f"loaded stage {resolved_model.stage} model in {resolved_model.mode} mode on {torch_device}; "
        f"full_model={resolved_model.is_full_model}"
    )
    return model


def canonical_chrom_name(chrom: str) -> str:
    if "chr" in chrom:
        return chrom[chrom.index("chr") :]
    return chrom


def build_chrom_alias_map(chrom_list: list[str]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for chrom in chrom_list:
        for alias in {chrom, canonical_chrom_name(chrom)}:
            existing = alias_map.get(alias)
            if existing is not None and existing != chrom:
                raise RuntimeError(f"ambiguous chromosome alias {alias!r} between {existing!r} and {chrom!r}")
            alias_map[alias] = chrom
    return alias_map


def parse_bed_line(line: str, index: int) -> tuple[str, int, int, str]:
    fields = line.strip().split()
    if len(fields) < 3:
        raise RuntimeError(f"BED line {index + 1} has fewer than 3 columns")
    chrom = fields[0]
    start = int(fields[1])
    end = int(fields[2])
    name = fields[3] if len(fields) > 3 and fields[3] else f"region_{index:06d}"
    return chrom, start, end, name


def get_impute_distance_bounds(config: dict, res: int) -> tuple[int, int]:
    if "minimum_impute_distance" in config:
        min_bin = int(config["minimum_impute_distance"] / res)
    else:
        min_distance = config["minimum_distance"]
        min_bin = 0 if min_distance < 0 else int(min_distance / res)

    if "maximum_impute_distance" in config:
        maximum_impute_distance = config["maximum_impute_distance"]
        max_bin = int(1e5) if maximum_impute_distance < 0 else int(maximum_impute_distance / res)
    else:
        max_distance = config["maximum_distance"]
        max_bin = int(1e5) if max_distance < 0 else int(max_distance / res)
    return min_bin, max_bin


def build_chrom_bad_bins(raw_bulk, config: dict, chrom: str) -> np.ndarray:
    bad_bins = np.asarray(raw_bulk.sum(axis=1)).reshape(-1) == 0
    skip_start, skip_end = skip_start_end(config, chrom)
    for start, end in zip(skip_start, skip_end):
        bad_bins[int(start) : int(end)] = True
    return bad_bins


def build_region_pairs(
    context_start_bin: int,
    context_end_bin: int,
    chrom_offset: int,
    bad_bins: np.ndarray,
    min_bin: int,
    max_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
    local_pairs: list[tuple[int, int]] = []
    global_pairs: list[tuple[int, int]] = []
    size = context_end_bin - context_start_bin
    if size < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

    for left in range(context_start_bin, context_end_bin):
        if bad_bins[left]:
            continue
        local_left = left - context_start_bin
        right_start = left + min_bin
        right_end = min(left + max_bin, context_end_bin)
        for right in range(right_start, right_end):
            if bad_bins[right]:
                continue
            local_right = right - context_start_bin
            local_pairs.append((local_left, local_right))
            global_pairs.append((chrom_offset + left + 1, chrom_offset + right + 1))

    return np.asarray(local_pairs, dtype=np.int64), np.asarray(global_pairs, dtype=np.int64)


def load_raw_bulk_cache(temp_dir: Path, chrom: str) -> np.ndarray:
    origin_sparse = np.load(temp_dir / "raw" / f"{chrom}_sparse_adj.npy", allow_pickle=True)
    return np.sum(origin_sparse, axis=0).astype("float32")


def build_region_specs(
    bed_path: Path,
    temp_dir: Path,
    config: dict,
    res: int,
    chrom_list: list[str],
    num_list: np.ndarray,
    chrom_bin_counts: dict[str, int],
    window_ins: int,
) -> list[RegionSpec]:
    alias_map = build_chrom_alias_map(chrom_list)
    chrom_to_index = {chrom: idx for idx, chrom in enumerate(chrom_list)}
    min_bin, max_bin = get_impute_distance_bounds(config, res)
    raw_bulk_cache: dict[str, np.ndarray] = {}
    bad_bins_cache: dict[str, np.ndarray] = {}
    regions: list[RegionSpec] = []

    with bed_path.open() as fh:
        for index, line in enumerate(fh):
            if not line.strip() or line.startswith("#"):
                continue
            original_chrom, bed_start_bp, bed_end_bp, name = parse_bed_line(line, index)
            normalized_chrom = alias_map.get(original_chrom) or alias_map.get(canonical_chrom_name(original_chrom))
            if normalized_chrom is None:
                raise RuntimeError(f"BED chromosome {original_chrom!r} does not map to config chrom_list")

            chrom_index = chrom_to_index[normalized_chrom]
            chrom_bins = chrom_bin_counts[normalized_chrom]
            target_start_bin = max(0, int(math.floor(bed_start_bp / res)))
            target_end_bin = min(chrom_bins, int(math.ceil(bed_end_bp / res)))
            context_pad = int(math.ceil(window_ins / res))
            context_start_bin = max(0, target_start_bin - context_pad)
            context_end_bin = min(chrom_bins, target_end_bin + context_pad)
            chrom_offset = int(num_list[chrom_index])
            if normalized_chrom not in raw_bulk_cache:
                raw_bulk_cache[normalized_chrom] = load_raw_bulk_cache(temp_dir, normalized_chrom)
            if normalized_chrom not in bad_bins_cache:
                bad_bins_cache[normalized_chrom] = build_chrom_bad_bins(
                    raw_bulk_cache[normalized_chrom], config, normalized_chrom
                )
            bad_bins = bad_bins_cache[normalized_chrom]

            status = "ok"
            coordinates = np.empty((0, 2), dtype=np.int64)
            samples = np.empty((0, 3), dtype=np.int64)

            if bed_end_bp <= bed_start_bp:
                status = "invalid_interval"
            elif target_end_bin - target_start_bin < 2:
                status = "empty_after_binning"
            elif context_end_bin - context_start_bin < 2:
                status = "empty_context"
            else:
                coordinates, global_pairs = build_region_pairs(
                    context_start_bin=context_start_bin,
                    context_end_bin=context_end_bin,
                    chrom_offset=chrom_offset,
                    bad_bins=bad_bins,
                    min_bin=min_bin,
                    max_bin=max_bin,
                )
                if len(global_pairs) == 0:
                    status = "empty_after_distance_filter"
                else:
                    samples = np.concatenate(
                        [np.ones((len(global_pairs), 1), dtype=np.int64), global_pairs],
                        axis=1,
                    )

            context_bins = np.arange(context_start_bin, context_end_bin, dtype=np.int64)
            bin_start = context_bins * res
            bin_end = (context_bins + 1) * res
            bin_chrom = np.asarray([normalized_chrom] * len(context_bins), dtype=object)

            raw_region = raw_bulk_cache[normalized_chrom][context_start_bin:context_end_bin, context_start_bin:context_end_bin]
            raw_region = np.asarray(raw_region.todense()) if hasattr(raw_region, "todense") else np.asarray(raw_region)
            local_bad = bad_bins[context_start_bin:context_end_bin]
            region_mask = np.ones_like(raw_region, dtype=np.float32)
            region_mask[local_bad, :] = 0.0
            region_mask[:, local_bad] = 0.0
            masked_raw_region = raw_region * region_mask
            if len(raw_region) == 0:
                discard_rows = np.array([], dtype=np.int64)
            else:
                row_threshold = 0.01 * np.sum(masked_raw_region) / max(len(masked_raw_region), 1)
                discard_rows = np.where(np.sum(masked_raw_region, axis=-1) <= row_threshold)[0].astype(np.int64)

            regions.append(
                RegionSpec(
                    index=len(regions),
                    group_name=f"region_{len(regions):06d}",
                    name=name,
                    original_chrom=original_chrom,
                    chrom=normalized_chrom,
                    bed_start_bp=bed_start_bp,
                    bed_end_bp=bed_end_bp,
                    target_start_bin=target_start_bin,
                    target_end_bin=target_end_bin,
                    context_start_bin=context_start_bin,
                    context_end_bin=context_end_bin,
                    chrom_index=chrom_index,
                    chrom_offset=chrom_offset,
                    status=status,
                    coordinates=coordinates,
                    samples=samples,
                    target_start_idx=target_start_bin - context_start_bin,
                    target_end_idx=target_end_bin - context_start_bin,
                    bin_chrom=bin_chrom,
                    bin_start=bin_start.astype(np.int64),
                    bin_end=bin_end.astype(np.int64),
                    bad_bins=local_bad.astype(bool),
                    discard_rows=discard_rows,
                )
            )
            region = regions[-1]
            log(
                f"region {region.group_name} ({region.name}): "
                f"{region.original_chrom}:{region.bed_start_bp}-{region.bed_end_bp} -> "
                f"{region.chrom} target_bins=[{region.target_start_bin}, {region.target_end_bin}) "
                f"context_bins=[{region.context_start_bin}, {region.context_end_bin}) "
                f"status={region.status} pairs={len(region.coordinates)}"
            )

    return regions


def prepare_cell_chrom_list(
    cell: int,
    chrom_runtime_specs: list[ChromRuntimeSpec],
    sparse_chrom_list,
    local_transfer_range: int,
    weighted_info,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[object]], list[int]]:
    def build_selected_row_block(adj, target_rows: np.ndarray, moving_range: int):
        adj = adj.tocsr()
        if len(target_rows) == 0:
            return csr_matrix((0, adj.shape[1]), dtype=np.float32)
        if moving_range <= 0:
            return adj[target_rows].copy()

        row_block = adj[target_rows].copy() * norm.pdf(0)
        for shift in range(1, moving_range * 2 + 1):
            before_rows = np.maximum(target_rows - shift, 0)
            after_rows = np.minimum(target_rows + shift, adj.shape[0] - 1)
            weight = norm.pdf(shift / moving_range)
            row_block = row_block + (adj[before_rows] + adj[after_rows]) * weight
        return row_block

    def compress_row_block(row_block, chrom_offset: int) -> tuple[list[object], np.ndarray]:
        row_block = row_block.tocsr().astype("float32")
        row_block.data = np.log1p(row_block.data)
        row_block = normalize(row_block, norm="l1", axis=1).tocsr().astype("float32")
        if row_block.nnz == 0:
            indices = torch.empty((2, 0), dtype=torch.long)
            values = torch.empty((0,), dtype=torch.float32)
            return [indices, values, (row_block.shape[0], 0)], np.empty((0,), dtype=np.int64)

        unique_cols, inverse = np.unique(row_block.indices, return_inverse=True)
        rows = np.repeat(np.arange(row_block.shape[0], dtype=np.int64), np.diff(row_block.indptr))
        cols = inverse.astype(np.int64, copy=False)
        indices = torch.from_numpy(np.vstack([rows, cols]))
        values = torch.from_numpy(row_block.data.astype(np.float32, copy=False))
        col_bin_ids = unique_cols.astype(np.int64) + chrom_offset + 1
        return [indices, values, (row_block.shape[0], len(unique_cols))], col_bin_ids

    route_nn_list = [spec.chrom_index + 1 for spec in chrom_runtime_specs]
    row_bin_ids = [spec.target_global_bin_ids for spec in chrom_runtime_specs]
    col_bin_ids: list[np.ndarray] = []
    cell_chrom_list: list[list[object]] = []
    weighted_adj = weighted_info is not None
    if weighted_adj:
        cell_neighbor_list, weight_dict = weighted_info

    for spec in chrom_runtime_specs:
        if weighted_adj:
            chrom_matrix = sparse_chrom_list[spec.chrom_index][cell - 1] * 0.0
        else:
            chrom_matrix = sparse_chrom_list[spec.chrom_index][cell - 1]
        if weighted_adj:
            for neighbor_cell in cell_neighbor_list[cell]:
                balance_weight = weight_dict[(neighbor_cell, cell)]
                chrom_matrix = chrom_matrix + balance_weight * sparse_chrom_list[spec.chrom_index][neighbor_cell - 1]

        row_block = build_selected_row_block(chrom_matrix, spec.target_local_bin_ids, local_transfer_range)
        sparse_input, chrom_col_bin_ids = compress_row_block(row_block, spec.chrom_offset)
        cell_chrom_list.append(sparse_input)
        col_bin_ids.append(chrom_col_bin_ids)

    return row_bin_ids, col_bin_ids, cell_chrom_list, route_nn_list


def build_chrom_runtime_specs(
    used_regions: list[RegionSpec],
) -> list[ChromRuntimeSpec]:
    chrom_to_bins: dict[str, list[np.ndarray]] = {}
    chrom_to_index: dict[str, int] = {}
    chrom_to_offset: dict[str, int] = {}
    chrom_order: list[str] = []
    specs: list[ChromRuntimeSpec] = []
    for region in sorted(used_regions, key=lambda item: item.chrom_index):
        if region.chrom not in chrom_to_bins:
            chrom_to_bins[region.chrom] = []
            chrom_order.append(region.chrom)
            chrom_to_index[region.chrom] = region.chrom_index
            chrom_to_offset[region.chrom] = region.chrom_offset
        chrom_to_bins[region.chrom].append(np.unique(region.samples[:, 1:].reshape(-1)))

    for chrom in chrom_order:
        target_global_bin_ids = np.unique(np.concatenate(chrom_to_bins[chrom])).astype(np.int64)
        chrom_offset = chrom_to_offset[chrom]
        target_local_bin_ids = target_global_bin_ids - chrom_offset - 1
        specs.append(
            ChromRuntimeSpec(
                chrom=chrom,
                chrom_index=chrom_to_index[chrom],
                chrom_offset=chrom_offset,
                target_local_bin_ids=target_local_bin_ids.astype(np.int64),
                target_global_bin_ids=target_global_bin_ids,
            )
        )
    return specs


def write_string_dataset(group: h5py.Group, name: str, values: Iterable[str]) -> None:
    dtype = h5py.string_dtype("utf-8")
    group.create_dataset(name, data=np.asarray(list(values), dtype=object), dtype=dtype)


def initialize_output(
    output_path: Path,
    args: argparse.Namespace,
    config_path: Path,
    resolved_model: ResolvedModel,
    config: dict,
    label_info: dict,
    regions: list[RegionSpec],
) -> tuple[h5py.File, dict[str, h5py.Group]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"creating output HDF5 at {output_path}")
    output_file = h5py.File(output_path, "w")
    output_file.attrs["config_path"] = str(config_path.resolve())
    output_file.attrs["resolved_model_path"] = str(resolved_model.model_path.resolve())
    output_file.attrs["resolved_mode"] = resolved_model.mode
    output_file.attrs["resolution"] = int(config["resolution"])
    output_file.attrs["window_ins"] = int(args.window_ins)

    cells_group = output_file.create_group("cells")
    cells_group.create_dataset("cell_id", data=np.asarray(label_info["cell_id"], dtype=np.int64))
    write_string_dataset(cells_group, "cell_label", [str(value) for value in label_info["cell_label"]])

    regions_group = output_file.create_group("regions")
    region_groups: dict[str, h5py.Group] = {}
    for region in regions:
        region_group = regions_group.create_group(region.group_name)
        region_group.attrs["name"] = region.name
        region_group.attrs["bed_chrom"] = region.original_chrom
        region_group.attrs["bed_start"] = int(region.bed_start_bp)
        region_group.attrs["bed_end"] = int(region.bed_end_bp)
        region_group.attrs["context_chrom"] = region.chrom
        region_group.attrs["context_start"] = int(region.context_start_bin * config["resolution"])
        region_group.attrs["context_end"] = int(region.context_end_bin * config["resolution"])
        region_group.attrs["status"] = region.status

        bins_group = region_group.create_group("bins")
        write_string_dataset(bins_group, "chrom", region.bin_chrom.tolist())
        bins_group.create_dataset("start", data=region.bin_start)
        bins_group.create_dataset("end", data=region.bin_end)

        matrix_group = region_group.create_group("matrix")
        matrix_group.create_dataset("coordinates", data=region.coordinates.astype(np.int64))

        insulation_group = region_group.create_group("insulation")
        insulation_group.create_dataset("target_bin_start_idx", data=np.asarray(region.target_start_idx, dtype=np.int64))
        insulation_group.create_dataset("target_bin_end_idx", data=np.asarray(region.target_end_idx, dtype=np.int64))
        if region.status != "ok":
            matrix_group.create_dataset("bulk_mean", data=np.empty((0,), dtype=np.float32))
            insulation_group.create_dataset("bulk_mean", data=np.empty((0,), dtype=np.float32))

        region_groups[region.group_name] = region_group
    return output_file, region_groups


def get_activation(mode: str):
    if mode == "classification":
        return torch.sigmoid
    return F.softplus


def to_dense_symmetric(size: int, coordinates: np.ndarray, values: np.ndarray) -> np.ndarray:
    dense = np.zeros((size, size), dtype=np.float32)
    if len(coordinates) > 0:
        dense[coordinates[:, 0], coordinates[:, 1]] = values.astype(np.float32)
    dense = dense + dense.T - np.diag(np.diag(dense))
    return dense


def compute_insulation_for_region(
    dense_matrix: np.ndarray,
    region: RegionSpec,
    window_ins: int,
    res: int,
) -> np.ndarray:
    mask = np.ones_like(dense_matrix, dtype=np.float32)
    if np.any(region.bad_bins):
        mask[region.bad_bins, :] = 0.0
        mask[:, region.bad_bins] = 0.0
    dense_matrix = dense_matrix * mask
    normalized = sqrt_norm(dense_matrix)
    score = insulation_score(normalized, windowsize=window_ins, res=res)
    if len(region.discard_rows) > 0:
        score[region.discard_rows] = 1.0
    if np.any(region.bad_bins):
        score[region.bad_bins] = 1.0
    return score[region.target_start_idx : region.target_end_idx].astype(np.float32)


def summarize_region_status(regions: list[RegionSpec]) -> str:
    status_counts: dict[str, int] = {}
    for region in regions:
        status_counts[region.status] = status_counts.get(region.status, 0) + 1
    return ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))


def main() -> int:
    args = parse_args()
    if args.predict_batch_size == 0:
        raise RuntimeError("predict-batch-size must be non-zero")
    if args.window_ins < 0:
        raise RuntimeError("window-ins must be non-negative")
    config_path = Path(args.config).resolve()
    bed_path = Path(args.bed).resolve()
    output_path = Path(args.output).resolve()
    log(
        f"starting with config={config_path}, bed={bed_path}, output={output_path}, "
        f"mode={args.mode}, window_ins={args.window_ins}, device={args.device}, "
        f"predict_batch_size={args.predict_batch_size}"
    )
    config = resolve_config_paths(config_path, load_config(config_path))
    temp_dir = Path(config["temp_dir"]).resolve()
    chrom_list = list(config["chrom_list"])
    res = int(config["resolution"])
    runtime_device, torch_device = resolve_runtime_device(args.device)
    log(f"resolved Higashi root: {HIGASHI_ROOT}")
    log(f"resolved runtime device: {runtime_device}")
    log(f"resolved temp_dir: {temp_dir}")
    log(f"chromosomes in config ({len(chrom_list)}): {', '.join(chrom_list)}")
    log(f"resolution: {res} bp")

    num, num_list, chrom_bin_counts = load_counts(temp_dir, chrom_list)
    cell_count = int(num[0])
    log(f"loaded node counts from {temp_dir / 'node_feats.hdf5'}")
    log(f"cell count: {cell_count}")
    log(
        "chromosome bin counts: "
        + ", ".join(f"{chrom}={chrom_bin_counts[chrom]}" for chrom in chrom_list)
    )
    label_info = load_label_info(config_path, config, cell_count)
    log(f"loaded {len(label_info['cell_id'])} cell labels")
    resolved_model = resolve_model_artifacts(temp_dir, args.mode)
    model = load_runtime_model(
        config_path=config_path,
        config=config,
        label_info=label_info,
        resolved_model=resolved_model,
        runtime_device=runtime_device,
        torch_device=torch_device,
    )

    regions = build_region_specs(
        bed_path=bed_path,
        temp_dir=temp_dir,
        config=config,
        res=res,
        chrom_list=chrom_list,
        num_list=num_list,
        chrom_bin_counts=chrom_bin_counts,
        window_ins=args.window_ins,
    )
    used_regions = [region for region in regions if region.status == "ok"]
    log(f"region summary: {summarize_region_status(regions)}")
    chrom_runtime_specs = build_chrom_runtime_specs(used_regions)
    if chrom_runtime_specs:
        log(
            "chromosomes used for imputation: "
            + ", ".join(spec.chrom for spec in chrom_runtime_specs)
        )
        log(
            "target bins per chromosome: "
            + ", ".join(f"{spec.chrom}={len(spec.target_global_bin_ids)}" for spec in chrom_runtime_specs)
        )
    output_file, region_groups = initialize_output(
        output_path=output_path,
        args=args,
        config_path=config_path,
        resolved_model=resolved_model,
        config=config,
        label_info=label_info,
        regions=regions,
    )

    if not used_regions:
        output_file.close()
        log("no imputable regions found; wrote metadata-only output")
        return 0

    local_transfer_range = int(config.get("local_transfer_range", 0))
    log(
        f"loading sparse chromosome graph from {temp_dir / 'sparse_nondiag_adj_nbr_1.npy'} "
        f"(local_transfer_range={local_transfer_range})"
    )
    sparse_chrom_list = np.load(temp_dir / "sparse_nondiag_adj_nbr_1.npy", allow_pickle=True)
    weighted_info = None
    if resolved_model.weighted_info_path is not None:
        log(f"loading neighbor weights from {resolved_model.weighted_info_path}")
        weighted_np = np.load(resolved_model.weighted_info_path, allow_pickle=True)
        weighted_info = (weighted_np[0], weighted_np[1])
    else:
        log("running without neighbor weights")

    all_samples = np.concatenate([region.samples for region in used_regions], axis=0)
    all_sample_chrom = np.concatenate(
        [np.full(len(region.samples), region.chrom_index, dtype=np.int64) for region in used_regions],
        axis=0,
    )
    region_slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for region in used_regions:
        region_slices[region.group_name] = (offset, offset + len(region.samples))
        offset += len(region.samples)
    log(f"total region triplets to predict per cell: {len(all_samples)}")
    log(f"usable regions: {len(used_regions)} of {len(regions)}")

    activation = get_activation(config["loss_mode"])
    log(f"loss mode: {config['loss_mode']}")
    model.eval()
    model.only_model = True
    embedding_init = model.encode1.static_nn
    embedding_init.off_hook()
    model.encode1.dynamic_nn.start_fix()
    model.encode1.dynamic_nn.forward = model.encode1.dynamic_nn.forward_off_hook

    bulk_accum = {region.group_name: np.zeros(len(region.coordinates), dtype=np.float64) for region in used_regions}
    verbose_every = int(config.get("impute_verbose", 10))
    log(f"per-cell progress logging interval: every {verbose_every} cells")
    start_time = time.time()

    try:
        for cell_idx in range(cell_count):
            cell_start_time = time.time()
            cell = cell_idx + 1
            cell_dataset_name = f"cell_{label_info['cell_id'][cell_idx]}"
            row_bin_ids, col_bin_ids, cell_chrom_list, route_nn_list = prepare_cell_chrom_list(
                cell=cell,
                chrom_runtime_specs=chrom_runtime_specs,
                sparse_chrom_list=sparse_chrom_list,
                local_transfer_range=local_transfer_range,
                weighted_info=weighted_info,
            )
            model.encode1.dynamic_nn.fix_cell_subset(
                cell,
                row_bin_ids=row_bin_ids,
                col_bin_ids=col_bin_ids,
                sparse_matrix=cell_chrom_list,
                local_transfer_range=local_transfer_range,
                route_nn_list=route_nn_list,
            )

            all_samples[:, 0] = cell
            predictions = model.predict(
                all_samples,
                all_sample_chrom,
                verbose=False,
                batch_size=args.predict_batch_size,
                activation=activation,
                extra_info=None,
            ).reshape(-1)

            for region in used_regions:
                slice_start, slice_end = region_slices[region.group_name]
                region_values = np.asarray(predictions[slice_start:slice_end], dtype=np.float32)
                region_group = region_groups[region.group_name]
                region_group["matrix"].create_dataset(
                    cell_dataset_name,
                    data=region_values,
                    compression="gzip",
                    compression_opts=6,
                )
                bulk_accum[region.group_name] += region_values

                dense = to_dense_symmetric(
                    size=region.context_end_bin - region.context_start_bin,
                    coordinates=region.coordinates,
                    values=region_values,
                )
                insulation = compute_insulation_for_region(dense, region, args.window_ins, res)
                region_group["insulation"].create_dataset(
                    cell_dataset_name,
                    data=insulation,
                    compression="gzip",
                    compression_opts=6,
                )

            if verbose_every > 0 and (cell_idx < 3 or cell_idx % verbose_every == 0 or cell_idx == cell_count - 1):
                elapsed = time.time() - cell_start_time
                total_elapsed = time.time() - start_time
                log(
                    f"processed cell {cell_idx + 1}/{cell_count} "
                    f"({cell_dataset_name}, triplets={len(all_samples)}, "
                    f"cell_time={elapsed:.2f}s, total_elapsed={total_elapsed:.2f}s)"
                )

        for region in used_regions:
            region_group = region_groups[region.group_name]
            bulk_mean = (bulk_accum[region.group_name] / cell_count).astype(np.float32)
            region_group["matrix"].create_dataset("bulk_mean", data=bulk_mean)

            dense_bulk = to_dense_symmetric(
                size=region.context_end_bin - region.context_start_bin,
                coordinates=region.coordinates,
                values=bulk_mean,
            )
            bulk_insulation = compute_insulation_for_region(dense_bulk, region, args.window_ins, res)
            region_group["insulation"].create_dataset("bulk_mean", data=bulk_insulation)
            log(
                f"wrote bulk outputs for {region.group_name} ({region.name}); "
                f"pairs={len(region.coordinates)}, target_bins={region.target_end_idx - region.target_start_idx}"
            )
    finally:
        output_file.close()
        embedding_init.on_hook()
        embedding_init.off_hook([0])
        model.only_model = False
        model.encode1.dynamic_nn.forward = model.encode1.dynamic_nn.forward_on_hook
        model.encode1.dynamic_nn.fix = False
        model.encode1.dynamic_nn.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"finished in {time.time() - start_time:.2f}s")
    log(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        raise SystemExit(fail(f"error: {exc}"))
    except FileNotFoundError as exc:
        missing = exc.filename if exc.filename is not None else str(exc)
        raise SystemExit(fail(f"error: missing file {missing}"))
