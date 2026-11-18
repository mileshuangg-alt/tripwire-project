from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import numpy as np
import torch


UPSTREAM_COMMIT = "c96f0559ea7a2504e797b9d65b9f1af1aa899ddd"

FROZEN_GRAPH_QUALIFIED_TSV = Path(
    "/scratch/user/mhuang8/actfound/chembl32/graph_production_20261113/"
    "actfound_chembl32_graph_qualified.tsv"
)

FROZEN_GRAPH_QUALIFIED_SHA256 = (
    "24d9fd297b735843284d82e7d31b85ccd46038f279931db8d5997fb9a096149e"
)

GRAPH_QUALIFIED_COLUMNS = (
    "assay_key",
    "activity_id",
    "assay_id",
    "record_id_unused",
    "molregno",
    "bao_endpoint",
    "standard_relation",
    "standard_value",
    "standard_units",
    "standard_type",
    "assay_type",
    "bao_format",
    "assay_chembl_id",
    "count_scratch",
    "canonical_smiles",
    "graph_pair_count",
)


@dataclass(frozen=True)
class ActFoundAssay:
    assay_id: str
    smiles: tuple[str, ...]
    y: np.ndarray
    x: Optional[np.ndarray] = None
    graph_pair_count: Optional[int] = None
    molecule_ids: Optional[tuple[str, ...]] = None
    observation_weights: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        y = np.asarray(self.y, dtype=np.float32)
        if len(self.smiles) != len(y):
            raise ValueError("SMILES must align exactly with assay labels")
        ids = (
            tuple(f"{self.assay_id}:{i}" for i in range(len(y)))
            if self.molecule_ids is None
            else tuple(str(value) for value in self.molecule_ids)
        )
        if len(ids) != len(y):
            raise ValueError("molecule_ids must align exactly with assay labels")
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "molecule_ids", ids)
        if self.observation_weights is not None:
            weights = np.asarray(self.observation_weights, dtype=np.float64)
            if weights.shape != y.shape:
                raise ValueError("observation_weights must align exactly with assay labels")
            observed = np.isfinite(y)
            invalid = observed & (~np.isfinite(weights) | (weights <= 0))
            if np.any(invalid):
                rows = [
                    (ids[i], float(y[i]), float(weights[i]))
                    for i in np.flatnonzero(invalid)
                ]
                raise ValueError(
                    f"Invalid uncertainty weights for observed ActFound labels: {rows[:50]}"
                )
            object.__setattr__(self, "observation_weights", weights)

    def with_features(self, x: np.ndarray) -> "ActFoundAssay":
        if len(x) != len(self.y):
            raise ValueError("feature rows must align with assay labels")
        return ActFoundAssay(
            assay_id=self.assay_id,
            smiles=self.smiles,
            y=self.y,
            x=np.asarray(x, dtype=np.float32),
            graph_pair_count=self.graph_pair_count,
            molecule_ids=self.molecule_ids,
            observation_weights=self.observation_weights,
        )


class ActFoundFrozenMetaDataset:
    """
    Minimal project-side equivalent of CHEMBLBDBMetaDataset for frozen TSV input.

    Split membership is supplied by the caller. This deliberately avoids
    reconstructing or changing the upstream chembl_split.json decision while
    preserving the in-memory representation and the upstream training duplicate
    rule: every train assay is repeated len(assay)//64 times in train_indices.
    """

    def __init__(
        self,
        assays: Sequence[ActFoundAssay],
        *,
        train_assays: Optional[set[str]] = None,
        val_assays: Optional[set[str]] = None,
        test_assays: Optional[set[str]] = None,
    ):
        self.Xs: list[Optional[np.ndarray]] = []
        self.ys: list[np.ndarray] = []
        self.smiles_all: list[tuple[str, ...]] = []
        self.assaes: list[str] = []
        self.train_indices: list[int] = []
        self.val_indices: list[int] = []
        self.test_indices: list[int] = []
        self.graph_pair_counts: list[Optional[int]] = []
        self.molecule_ids_all: list[tuple[str, ...]] = []
        self.observation_weights_all: list[Optional[np.ndarray]] = []

        train_assays = set(train_assays or ())
        val_assays = set(val_assays or ())
        test_assays = set(test_assays or ())

        for assay in assays:
            self._append_assay(assay)
            idx = len(self.assaes) - 1
            if assay.assay_id in train_assays:
                self.train_indices.append(idx)
                repeat_cnt = len(assay.y) // 64
                for repeat_idx in range(repeat_cnt):
                    repeated = ActFoundAssay(
                        assay_id=f"{assay.assay_id}_{repeat_idx}",
                        smiles=assay.smiles,
                        y=assay.y,
                        x=assay.x,
                        graph_pair_count=assay.graph_pair_count,
                        molecule_ids=assay.molecule_ids,
                        observation_weights=assay.observation_weights,
                    )
                    self._append_assay(repeated)
                    self.train_indices.append(len(self.assaes) - 1)
            elif assay.assay_id in val_assays:
                self.val_indices.append(idx)
            elif assay.assay_id in test_assays:
                self.test_indices.append(idx)

        self.data_length = {
            "train": len(self.train_indices),
            "val": len(self.val_indices),
            "test": len(self.test_indices),
            "train_weight": len(self.train_indices),
        }

    def _append_assay(self, assay: ActFoundAssay) -> None:
        self.Xs.append(assay.x)
        self.ys.append(assay.y)
        self.smiles_all.append(assay.smiles)
        self.assaes.append(assay.assay_id)
        self.graph_pair_counts.append(assay.graph_pair_count)
        self.molecule_ids_all.append(assay.molecule_ids)
        self.observation_weights_all.append(assay.observation_weights)

    @classmethod
    def from_tsv(
        cls,
        path: Path,
        *,
        train_assays: Optional[set[str]] = None,
        val_assays: Optional[set[str]] = None,
        test_assays: Optional[set[str]] = None,
        featurize: bool = False,
    ) -> "ActFoundFrozenMetaDataset":
        assays = []
        for assay in iter_frozen_graph_qualified_assays(path):
            if featurize:
                assay = assay.with_features(upstream_count_morgan_features(assay.smiles))
            assays.append(assay)
        return cls(
            assays,
            train_assays=train_assays,
            val_assays=val_assays,
            test_assays=test_assays,
        )

    def get_task_split(
        self,
        index: int,
        *,
        test_sup_num: Union[float, int],
        is_test: bool,
        rand_seed: int,
        include_metadata: bool = False,
    ):
        x = self.Xs[index]
        if x is None:
            raise ValueError("assay has no feature matrix; call from_tsv(..., featurize=True) or with_features")
        x_split, y_split, split, selected = make_episode_split(
            x,
            self.ys[index],
            test_sup_num=test_sup_num,
            is_test=is_test,
            rand_seed=rand_seed,
            return_indices=True,
        )
        if not include_metadata:
            return x_split, y_split, split
        weights = self.observation_weights_all[index]
        ids = self.molecule_ids_all[index]
        selected_weights = None if weights is None else np.asarray(weights)[selected].copy()
        selected_ids = tuple(ids[i] for i in selected)
        result = (x_split, y_split, split, selected_weights, selected_ids)
        return result


def _upstream_pic50(standard_value: str) -> float:
    return -math.log10(float(standard_value))


def _upstream_assay_descriptor(row: dict[str, str]) -> str:
    """Match dataset/load_dataset.py::read_chembl_assay assay_id construction."""
    return "{}_{}_{}".format(
        row["assay_chembl_id"],
        row["standard_units"],
        row["standard_type"],
    ).replace("/", "_")


def iter_frozen_graph_qualified_assays(path: Path) -> Iterator[ActFoundAssay]:
    """
    Stream the frozen graph-qualified TSV as ActFound assay tasks.

    This starts after the already-frozen projection/pregraph/graph stage. It
    preserves the model-side gates still present in upstream read_chembl_assay:
    skip percent units, require '=' relation, require at least 20 rows, and
    require per-assay pIC50 standard deviation > 0.2.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != GRAPH_QUALIFIED_COLUMNS:
            raise ValueError("unexpected graph-qualified TSV header")

        current_key: Optional[str] = None
        rows: list[dict[str, str]] = []

        def emit() -> Optional[ActFoundAssay]:
            if not rows:
                return None
            if len(rows) > 10000:
                return None
            from rdkit import Chem

            smiles: list[str] = []
            molecule_ids: list[str] = []
            y_values: list[float] = []
            pair_counts: set[int] = set()
            descriptor: Optional[str] = None
            for row in rows:
                if row["standard_units"] == "%":
                    continue
                if row["standard_relation"] != "=":
                    continue
                mol = Chem.MolFromSmiles(row["canonical_smiles"])
                if mol is None:
                    continue
                if descriptor is None:
                    descriptor = _upstream_assay_descriptor(row)
                elif descriptor != _upstream_assay_descriptor(row):
                    raise ValueError("frozen assay group contains multiple upstream descriptors")
                smiles.append(row["canonical_smiles"])
                molecule_ids.append(str(row["molregno"]))
                y_values.append(_upstream_pic50(row["standard_value"]))
                pair_counts.add(int(row["graph_pair_count"]))

            if len(y_values) < 20:
                return None
            y = np.asarray(y_values, dtype=np.float32)
            if float(np.std(y)) <= 0.2:
                return None
            if descriptor is None:
                return None
            graph_pair_count = next(iter(pair_counts)) if len(pair_counts) == 1 else None
            return ActFoundAssay(
                assay_id=descriptor,
                smiles=tuple(smiles),
                y=y,
                graph_pair_count=graph_pair_count,
                molecule_ids=tuple(molecule_ids),
            )

        for row in reader:
            assay_key = row["assay_key"]
            if current_key is None:
                current_key = assay_key
            if assay_key != current_key:
                assay = emit()
                if assay is not None:
                    yield assay
                rows = []
                current_key = assay_key
            rows.append(row)

        assay = emit()
        if assay is not None:
            yield assay


def upstream_count_morgan_features(smiles: Sequence[str]) -> np.ndarray:
    """Reproduce dataset.data_base.preprocess_assay fingerprint construction."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    x_tmp: list[np.ndarray] = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            continue
        fp = rdFingerprintGenerator.GetCountFPs(
            [mol], fpType=rdFingerprintGenerator.MorganFP
        )[0]
        arr = np.zeros((0,), np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        x_tmp.append(arr)
    return np.asarray(x_tmp, dtype=np.float32)


def make_episode_split(
    x_in: np.ndarray,
    y_in: np.ndarray,
    *,
    test_sup_num: Union[float, int],
    is_test: bool,
    rand_seed: int,
    datasource: str = "chembl",
    return_indices: bool = False,
):
    """Reproduce the existing sampler; optionally return selected source indices."""
    rng = np.random.RandomState(seed=rand_seed)
    x = np.asarray(x_in, dtype=np.float32)
    y = np.asarray(y_in, dtype=np.float32)
    if len(x) != len(y):
        raise ValueError("episode features and labels must have equal row counts")
    selected_indices = np.arange(len(x), dtype=np.int64)

    if len(x) > 64 and not is_test and datasource != "gdsc":
        subset_num = 64
        select_idx = [1] * subset_num + [0] * (len(x) - subset_num)
        rng.shuffle(select_idx)
        select_idx = np.nonzero(np.asarray(select_idx))[0]
        x = x[select_idx]
        y = y[select_idx]
        selected_indices = selected_indices[select_idx]

    sup_num: Union[float, int] = test_sup_num
    if sup_num <= 1:
        sup_num = sup_num * len(x)
    sup_num = int(sup_num)
    if not is_test:
        min_num = math.log10(max(10, int(0.3 * len(x))))
        max_num = math.log10(int(0.85 * len(x)))
        sup_num = math.floor(10 ** random.uniform(min_num, max_num))
    split = [1] * int(sup_num) + [0] * (len(x) - int(sup_num))
    rng.shuffle(split)
    split_array = np.asarray(split, dtype=np.int64)
    if return_indices:
        return x, y, split_array, selected_indices
    return x, y, split_array


def jaccard_count_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ActFound get_sim_matrix: count fingerprints are binarized at >0."""
    a_bool = (a > 0.0).float()
    b_bool = (b > 0.0).float()
    and_res = torch.mm(a_bool, b_bool.transpose(0, 1))
    or_res = a.shape[-1] - torch.mm((1.0 - a_bool), (1.0 - b_bool).transpose(0, 1))
    return and_res / or_res


def cossim_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_norm = a / torch.clamp(a.norm(dim=1)[:, None], min=eps)
    b_norm = b / torch.clamp(b.norm(dim=1)[:, None], min=eps)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def curve_fit_observation_weights(
    conf_low,
    conf_high,
    observed_mask,
    molecule_ids,
) -> np.ndarray:
    """Raw bounded inverse-width weights: 1 / (conf_high - conf_low + 0.25)."""
    low = np.asarray(conf_low, dtype=np.float64)
    high = np.asarray(conf_high, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=bool)
    ids = tuple(str(value) for value in molecule_ids)
    if low.ndim != 1 or high.shape != low.shape or mask.shape != low.shape:
        raise ValueError("curve-fit CI bounds and observed mask must be aligned vectors")
    if len(ids) != len(low):
        raise ValueError("molecule_ids must align with curve-fit observations")
    result = np.full(low.shape, np.nan, dtype=np.float64)
    bad = []
    for i in np.flatnonzero(mask):
        if not np.isfinite(low[i]) or not np.isfinite(high[i]):
            bad.append((ids[i], "missing/non-finite CI bound"))
            continue
        width = high[i] - low[i]
        if not np.isfinite(width) or width <= 0:
            bad.append((ids[i], "non-positive/non-finite CI width"))
            continue
        result[i] = 1.0 / (width + 0.25)
    if bad:
        raise ValueError(f"Invalid observed curve-fit uncertainty rows: {bad[:50]}")
    return result


def screen_observation_weights(
    log2fc_std_error,
    observed_mask,
    molecule_ids,
) -> np.ndarray:
    """Raw inverse reported-variance weights: 1 / log2fc_std_error squared."""
    se = np.asarray(log2fc_std_error, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=bool)
    ids = tuple(str(value) for value in molecule_ids)
    if se.ndim != 1 or mask.shape != se.shape:
        raise ValueError("screen SE and observed mask must be aligned vectors")
    if len(ids) != len(se):
        raise ValueError("molecule_ids must align with screen observations")
    result = np.full(se.shape, np.nan, dtype=np.float64)
    bad = []
    for i in np.flatnonzero(mask):
        if not np.isfinite(se[i]) or se[i] <= 0:
            bad.append((ids[i], "missing/non-finite/non-positive SE"))
            continue
        result[i] = 1.0 / (se[i] ** 2)
        if not np.isfinite(result[i]) or result[i] <= 0:
            bad.append((ids[i], "inverse-SE-squared weight is non-finite/non-positive"))
    if bad:
        raise ValueError(f"Invalid observed screening uncertainty rows: {bad[:50]}")
    return result


def harmonic_pair_weights(weights_a: torch.Tensor, weights_b: torch.Tensor) -> torch.Tensor:
    """Raw pair weights (w_i * w_j) / (w_i + w_j); no partition normalization."""
    if weights_a.ndim != 1 or weights_b.ndim != 1:
        raise ValueError("pair endpoint weights must be vectors")
    wa = weights_a.unsqueeze(1)
    wb = weights_b.unsqueeze(0)
    return (wa * wb) / (wa + wb)


def _weighted_direct_mse(predictions, targets, raw_weights):
    observed = torch.isfinite(targets)
    if not bool(observed.any()):
        raise ValueError("ActFound direct loss has no observed labels")
    errors = (predictions[observed] - targets[observed]) ** 2
    if raw_weights is None:
        result = torch.mean(errors)
        return result
    selected = raw_weights[observed]
    if not bool(torch.isfinite(selected).all()) or not bool((selected > 0).all()):
        raise ValueError("Observed direct-loss weights must be finite and positive")
    normalized = selected / selected.mean()
    result = torch.mean(errors * normalized)
    return result


def _selected_neighbor_prediction(preds_all, embed_sim, topk_idx, source_observed):
    selected_predictions = torch.gather(preds_all, 0, topk_idx)
    selected_similarity = torch.gather(embed_sim, 0, topk_idx)
    source_mask = source_observed.to(torch.bool).unsqueeze(1).expand_as(preds_all)
    selected_observed = torch.gather(source_mask, 0, topk_idx)
    if not bool(selected_observed.any(dim=0).all()):
        raise ValueError("An ActFound prediction has no observed selected source neighbour")
    logits = torch.where(
        selected_observed,
        selected_similarity,
        torch.full_like(selected_similarity, -1e9),
    )
    probabilities = torch.softmax(logits, dim=0)
    selected_predictions = torch.where(
        selected_observed,
        selected_predictions,
        torch.zeros_like(selected_predictions),
    )
    result = torch.sum(selected_predictions * probabilities, dim=0)
    return result


def robust_square_error(
    a: torch.Tensor,
    b: torch.Tensor,
    topk_idx: torch.Tensor,
    *,
    observation_weights_a: Optional[torch.Tensor] = None,
    observation_weights_b: Optional[torch.Tensor] = None,
    observed_a: Optional[torch.Tensor] = None,
    observed_b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Original robust error; optional harmonic weights are gathered after top-k."""
    rows, columns = a.shape
    if observed_a is None:
        observed_a = torch.ones(rows, dtype=torch.bool, device=a.device)
    if observed_b is None:
        observed_b = torch.ones(columns, dtype=torch.bool, device=a.device)
    pair_mask = observed_a.to(torch.bool).unsqueeze(1) & observed_b.to(torch.bool).unsqueeze(0)

    # Avoid NaN arithmetic for genuinely missing labels before applying the mask.
    difference = torch.where(pair_mask, a - b, torch.zeros_like(a))
    abs_diff = torch.abs(difference)
    square_mask = (abs_diff <= 2).to(dtype=a.dtype)
    linear_mask = 1.0 - square_mask
    square_error = difference ** 2
    linear_error = 1.0 * (abs_diff - 2) + 4
    loss = square_error * square_mask + linear_error * linear_mask

    selected_loss = torch.gather(loss, 0, topk_idx)
    selected_mask = torch.gather(pair_mask, 0, topk_idx)
    if not bool(selected_mask.any()):
        raise ValueError("ActFound pairwise reduction has no observed selected pairs")

    weighted = observation_weights_a is not None or observation_weights_b is not None
    if not weighted:
        if bool(selected_mask.all()):
            return torch.mean(selected_loss)
        result = torch.mean(selected_loss[selected_mask])
        return result

    if observation_weights_a is None:
        observation_weights_a = torch.ones(rows, dtype=a.dtype, device=a.device)
    if observation_weights_b is None:
        observation_weights_b = torch.ones(columns, dtype=a.dtype, device=a.device)
    if observation_weights_a.shape != (rows,) or observation_weights_b.shape != (columns,):
        raise ValueError("pair endpoint weights do not match pair-error axes")

    # Crucially, combine RAW endpoint weights first; only selected pairs are normalized.
    pair_raw = harmonic_pair_weights(observation_weights_a, observation_weights_b)
    selected_raw = torch.gather(pair_raw, 0, topk_idx)
    valid_raw = selected_raw[selected_mask]
    if not bool(torch.isfinite(valid_raw).all()) or not bool((valid_raw > 0).all()):
        raise ValueError("selected raw harmonic pair weights must be finite and positive")
    normalized = valid_raw / valid_raw.mean()
    result = torch.mean(selected_loss[selected_mask] * normalized)
    return result


@dataclass(frozen=True)
class PairMatrices:
    support_idx: torch.Tensor
    target_idx: torch.Tensor
    support_support_real: torch.Tensor
    support_target_real: torch.Tensor
    target_target_real: torch.Tensor
    support_support_topk_idx: torch.Tensor
    support_target_topk_idx: torch.Tensor
    target_target_topk_idx: torch.Tensor


def construct_pair_matrices(x: torch.Tensor, y: torch.Tensor, split: torch.Tensor) -> PairMatrices:
    """Expose the pair matrices constructed inside upstream net_forward."""
    split = split.to(dtype=torch.float32, device=x.device)
    sup_idx = torch.nonzero(split)[:, 0]
    tgt_idx = torch.nonzero(1.0 - split)[:, 0]
    sup_x = x[sup_idx]
    tgt_x = x[tgt_idx]
    sup_y = y[sup_idx]
    tgt_y = y[tgt_idx]

    sup_num = int(sup_idx.shape[0])
    tgt_num = int(tgt_idx.shape[0])
    if sup_num < 2 or tgt_num < 2:
        raise ValueError("ActFound pair construction requires at least two support and two target rows")

    sup_sim = jaccard_count_sim_matrix(sup_x, sup_x) - torch.eye(sup_num, device=x.device)
    _, sup_topk = torch.topk(sup_sim, dim=0, k=sup_sim.shape[0] // 2)

    cross_sim = jaccard_count_sim_matrix(sup_x, tgt_x)
    _, cross_topk = torch.topk(cross_sim, dim=0, k=int(cross_sim.shape[0] * 0.5))

    tgt_sim = jaccard_count_sim_matrix(tgt_x, tgt_x) - torch.eye(tgt_num, device=x.device)
    _, tgt_topk = torch.topk(tgt_sim, dim=0, k=tgt_sim.shape[0] // 2)

    return PairMatrices(
        support_idx=sup_idx,
        target_idx=tgt_idx,
        support_support_real=sup_y.unsqueeze(-1) - sup_y.unsqueeze(0),
        support_target_real=sup_y.unsqueeze(-1) - tgt_y.unsqueeze(0),
        target_target_real=tgt_y.unsqueeze(-1) - tgt_y.unsqueeze(0),
        support_support_topk_idx=sup_topk,
        support_target_topk_idx=cross_topk,
        target_target_topk_idx=tgt_topk,
    )


def actfound_loss_and_preds(
    x: torch.Tensor,
    y: torch.Tensor,
    split: torch.Tensor,
    embeddings: torch.Tensor,
    values: torch.Tensor,
    *,
    is_support: bool,
    is_training_phase: bool,
    temp: float = 0.2,
    observation_weights: Optional[torch.Tensor] = None,
    molecule_ids: Optional[Sequence[str]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ActFound loss with optional raw observation weights; pair topology is unchanged."""
    split = split.to(dtype=torch.float32, device=x.device)
    sup_idx = torch.nonzero(split)[:, 0]
    tgt_idx = torch.nonzero(1.0 - split)[:, 0]
    sup_x = x[sup_idx]
    tgt_x = x[tgt_idx]
    sup_y = y[sup_idx]
    tgt_y = y[tgt_idx]
    support_value = values[sup_idx]
    tgt_value = values[tgt_idx]
    support_features = embeddings[sup_idx]
    query_features = embeddings[tgt_idx]
    sup_num = int(sup_idx.shape[0])
    tgt_num = int(tgt_idx.shape[0])
    if sup_num < 2 or tgt_num < 2:
        raise ValueError("ActFound pair construction requires at least two support and two target rows")

    all_observed = torch.isfinite(y)
    raw_all = None
    if observation_weights is not None:
        raw_all = torch.as_tensor(observation_weights, dtype=y.dtype, device=y.device)
        if raw_all.shape != y.shape:
            raise ValueError("observation_weights must align with the full ActFound episode")
        ids = tuple(str(v) for v in molecule_ids) if molecule_ids is not None else tuple(str(i) for i in range(len(y)))
        if len(ids) != len(y):
            raise ValueError("molecule_ids must align with ActFound episode labels")
        invalid = all_observed & (~torch.isfinite(raw_all) | (raw_all <= 0))
        if bool(invalid.any()):
            rows = [
                (ids[i], float(y[i].detach().cpu()), float(raw_all[i].detach().cpu()))
                for i in torch.nonzero(invalid).flatten().tolist()
            ]
            raise ValueError(f"Invalid uncertainty for observed ActFound labels: {rows[:50]}")

    sup_observed = torch.isfinite(sup_y)
    tgt_observed = torch.isfinite(tgt_y)
    if int(sup_observed.sum()) < 2 or int(tgt_observed.sum()) < 2:
        raise ValueError("ActFound requires at least two observed support and query labels")
    sup_weights = None if raw_all is None else raw_all[sup_idx]
    tgt_weights = None if raw_all is None else raw_all[tgt_idx]

    softmax = torch.nn.Softmax(dim=0)
    observed_sup_y = sup_y[sup_observed]
    rescale = 1.0 if is_training_phase else max(0.2, float(observed_sup_y.std().detach().cpu()))

    if is_support:
        ddg_pred = (support_value.unsqueeze(-1) - support_value.unsqueeze(0)) * rescale
        sup_sim = jaccard_count_sim_matrix(sup_x, sup_x) - torch.eye(sup_num, device=x.device)
        _, topk_idx = torch.topk(sup_sim, dim=0, k=sup_sim.shape[0] // 2)
        embed_sim = cossim_matrix(support_features, support_features) / temp
        embed_sim = embed_sim - 1e9 * torch.eye(support_features.shape[0], device=x.device)
        preds_all = sup_y.unsqueeze(-1).repeat(1, sup_num) - ddg_pred
        if bool(sup_observed.all()):
            embed_select = softmax(torch.gather(embed_sim, 0, topk_idx))
            preds = torch.sum(torch.gather(preds_all, 0, topk_idx) * embed_select, dim=0)
        else:
            preds = _selected_neighbor_prediction(preds_all, embed_sim, topk_idx, sup_observed)
        loss = _weighted_direct_mse(preds, sup_y, sup_weights) / rescale ** 2
        return torch.sqrt(loss), preds

    ddg_pred_1 = support_value.unsqueeze(-1) - tgt_value.unsqueeze(0)
    ddg_real_1 = sup_y.unsqueeze(-1) - tgt_y.unsqueeze(0)
    ddg_pred_2 = tgt_value.unsqueeze(-1) - tgt_value.unsqueeze(0)
    ddg_real_2 = tgt_y.unsqueeze(-1) - tgt_y.unsqueeze(0)
    cross_sim = jaccard_count_sim_matrix(sup_x, tgt_x)
    _, topk_idx = torch.topk(cross_sim, dim=0, k=int(cross_sim.shape[0] * 0.5))
    embed_sim = cossim_matrix(support_features, query_features) / temp
    preds_all = sup_y.unsqueeze(-1).repeat(1, tgt_num) - ddg_pred_1 * rescale
    if bool(sup_observed.all()):
        preds = torch.sum(
            torch.gather(preds_all, 0, topk_idx)
            * softmax(torch.gather(embed_sim, 0, topk_idx)),
            dim=0,
        )
    else:
        preds = _selected_neighbor_prediction(preds_all, embed_sim, topk_idx, sup_observed)

    tgt_sim = jaccard_count_sim_matrix(tgt_x, tgt_x) - torch.eye(tgt_num, device=x.device)
    _, tgt_topk_idx = torch.topk(tgt_sim, dim=0, k=tgt_sim.shape[0] // 2)

    if raw_all is None and bool(all_observed.all()):
        # Preserve the exact original unweighted ActFound objective.
        loss_2 = robust_square_error(ddg_pred_2, ddg_real_2, tgt_topk_idx)
        loss_1 = robust_square_error(ddg_pred_1, ddg_real_1, topk_idx)
    else:
        loss_2 = robust_square_error(
            ddg_pred_2,
            ddg_real_2,
            tgt_topk_idx,
            observation_weights_a=tgt_weights,
            observation_weights_b=tgt_weights,
            observed_a=tgt_observed,
            observed_b=tgt_observed,
        )
        loss_1 = robust_square_error(
            ddg_pred_1,
            ddg_real_1,
            topk_idx,
            observation_weights_a=sup_weights,
            observation_weights_b=tgt_weights,
            observed_a=sup_observed,
            observed_b=tgt_observed,
        )
    loss_dg = _weighted_direct_mse(preds, tgt_y, tgt_weights)
    result = torch.sqrt(loss_2 * 0.25 + loss_1 * 0.5 + loss_dg)
    return result, preds


def select_actfound_inner_loop_params(
    named_parameters: Sequence[tuple[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Reproduce RegressorBase.get_inner_loop_parameter_dict name filtering."""
    return {
        name: param
        for name, param in named_parameters
        if param.requires_grad
        and ("layer_dict.linear.bias" in name or "layer_dict.linear.weights" in name)
    }


def linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = x
    values = torch.nn.functional.linear(x, weight, bias)[:, 0]
    return embeddings, values


def actfound_forward_with_regressor(
    regressor: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    split: torch.Tensor,
    *,
    weights: Optional[dict[str, torch.Tensor]],
    is_support: bool,
    is_training_phase: bool,
    training: bool = True,
    backup_running_statistics: bool = False,
    num_step: int = 0,
    observation_weights: Optional[torch.Tensor] = None,
    molecule_ids: Optional[Sequence[str]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal adapter for the real upstream FCNReLUNormNetworkQSAR forward path."""
    embeddings, values = regressor.forward(
        x=x,
        params=weights,
        training=training,
        backup_running_statistics=backup_running_statistics,
        num_step=num_step,
    )
    return actfound_loss_and_preds(
        x,
        y,
        split,
        embeddings,
        values,
        is_support=is_support,
        is_training_phase=is_training_phase,
        observation_weights=observation_weights,
        molecule_ids=molecule_ids,
    )


def continuous_inference_from_outputs(
    x: torch.Tensor,
    y: torch.Tensor,
    split: torch.Tensor,
    embeddings: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return actfound_loss_and_preds(
        x,
        y,
        split,
        embeddings,
        values,
        is_support=False,
        is_training_phase=False,
    )


def adapt_final_linear_params(
    x: torch.Tensor,
    y: torch.Tensor,
    split: torch.Tensor,
    initial_params: dict[str, torch.Tensor],
    *,
    num_steps: int,
    lr: float,
    is_training_phase: bool = False,
    create_graph: bool = False,
) -> tuple[dict[str, torch.Tensor], tuple[float, ...]]:
    """Adapt exactly the upstream final linear inner-loop parameter set."""
    unexpected = set(initial_params) - {
        "layer_dict.linear.weights",
        "layer_dict.linear.bias",
    }
    if unexpected:
        raise ValueError(f"unexpected ActFound inner-loop parameters: {sorted(unexpected)}")
    if "layer_dict.linear.weights" not in initial_params:
        raise ValueError("missing layer_dict.linear.weights")

    params = dict(initial_params)
    support_losses: list[float] = []
    for _ in range(num_steps):
        embeddings, values = linear_forward(
            x,
            params["layer_dict.linear.weights"],
            params.get("layer_dict.linear.bias"),
        )
        loss, _ = actfound_loss_and_preds(
            x,
            y,
            split,
            embeddings,
            values,
            is_support=True,
            is_training_phase=is_training_phase,
        )
        support_losses.append(float(loss.detach().cpu().item()))
        grads = torch.autograd.grad(
            loss,
            tuple(params.values()),
            create_graph=create_graph,
        )
        params = {
            name: value - lr * grad
            for (name, value), grad in zip(params.items(), grads)
        }
    return params, tuple(support_losses)


def adapt_final_linear_weight(
    x: torch.Tensor,
    y: torch.Tensor,
    split: torch.Tensor,
    initial_weight: torch.Tensor,
    *,
    num_steps: int,
    lr: float,
    is_training_phase: bool = False,
) -> tuple[torch.Tensor, tuple[float, ...]]:
    """Compatibility wrapper for QSAR ActFound, whose final linear has no bias."""
    params, support_losses = adapt_final_linear_params(
        x,
        y,
        split,
        {"layer_dict.linear.weights": initial_weight},
        num_steps=num_steps,
        lr=lr,
        is_training_phase=is_training_phase,
        create_graph=False,
    )
    return params["layer_dict.linear.weights"], support_losses


def continuous_inference(
    x: torch.Tensor,
    y: torch.Tensor,
    split: torch.Tensor,
    adapted_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings, values = linear_forward(x, adapted_weight)
    return continuous_inference_from_outputs(x, y, split, embeddings, values)


PROVENANCE = {
    "dataset_adapter": {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_file": "dataset/load_dataset.py; dataset/data_base.py; dataset/data_chemblbdb_Assay_reg.py",
        "upstream_lines": "load_dataset.py:260-327; data_base.py:78-110,134-233,235-280; data_chemblbdb_Assay_reg.py:22-167",
        "project_implementation": "cyp/scripts/actfound_model_side.py:ActFoundAssay, ActFoundFrozenMetaDataset, iter_frozen_graph_qualified_assays, upstream_count_morgan_features, make_episode_split",
        "qualification_test": "cyp/tests/test_actfound_model_side.py::test_dataset_adapter_preserves_upstream_representation_and_filters",
    },
    "pair_construction": {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_file": "learning_system/system_actfound.py",
        "upstream_lines": "138-225",
        "project_implementation": "cyp/scripts/actfound_model_side.py:construct_pair_matrices",
        "qualification_test": "cyp/tests/test_actfound_model_side.py::test_pair_construction_exact_counts_and_labels",
    },
    "loss": {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_file": "learning_system/system_base.py; learning_system/system_actfound.py",
        "upstream_lines": "system_base.py:71-88; system_actfound.py:138-228",
        "project_implementation": "cyp/scripts/actfound_model_side.py:robust_square_error, actfound_loss_and_preds, continuous_inference_from_outputs, actfound_forward_with_regressor",
        "qualification_test": "cyp/tests/test_actfound_model_side.py::test_pairwise_loss_behavior_matches_formula",
    },
    "sampler": {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_file": "dataset/data_base.py",
        "upstream_lines": "134-233,303-311,357-406",
        "project_implementation": "cyp/scripts/actfound_model_side.py:make_episode_split",
        "qualification_test": "cyp/tests/test_actfound_model_side.py::test_sampling_determinism_under_fixed_seeds",
    },
    "target_adaptation": {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_file": "learning_system/system_base.py; inner_loop_optimizers.py",
        "upstream_lines": "system_base.py:113-135,152-196; inner_loop_optimizers.py:58-141",
        "project_implementation": "cyp/scripts/actfound_model_side.py:select_actfound_inner_loop_params, adapt_final_linear_params, adapt_final_linear_weight",
        "qualification_test": "cyp/tests/test_actfound_model_side.py::test_target_adaptation_inputs_outputs",
    },
    "inference": {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_file": "main_reg.py; learning_system/system_base.py; learning_system/system_actfound.py",
        "upstream_lines": "main_reg.py:137-182,223-252; system_base.py:198-221,260-267; system_actfound.py:48-74,138-228",
        "project_implementation": "cyp/scripts/actfound_model_side.py:continuous_inference, continuous_inference_from_outputs, actfound_forward_with_regressor",
        "qualification_test": "cyp/tests/test_actfound_model_side.py::test_inference_output_semantics",
    },
}


def load_adjudicated_screen_assays(
    screen_path,
    identity_path,
    similarity_path,
    contract_path,
    *,
    featurize=False,
):
    """Load exact adjudicated screen observations as four enzyme-specific tasks."""
    import hashlib
    import json

    contract = json.loads(Path(contract_path).read_text())
    paths = {
        "screen": Path(screen_path),
        "identity": Path(identity_path),
        "similarity": Path(similarity_path),
    }
    for kind, path in paths.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = contract["sha256"][kind]
        if actual != expected:
            raise ValueError(
                f"{kind} input hash mismatch: {path}; "
                f"expected={expected}, actual={actual}"
            )

    def read_rows(path, delimiter):
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))

    def unique_adjudications(records):
        table = {}
        for row in records:
            name = row["molecule_name"]
            if name in table:
                raise ValueError(f"Duplicate adjudication identity: {name}")
            table[name] = row
        return table

    identity = unique_adjudications(read_rows(paths["identity"], "\t"))
    similarity = unique_adjudications(read_rows(paths["similarity"], "\t"))
    enzymes = tuple(contract["enzymes"])
    groups = {enzyme: [] for enzyme in enzymes}
    source_names = set()
    population = set()
    seen_pairs = set()

    for row_number, row in enumerate(read_rows(paths["screen"], ","), start=2):
        name = row["Molecule_Name"]
        smiles = row["SMILES"]
        enzyme = row["enzyme"]
        source_names.add(name)

        if name not in identity:
            raise ValueError(
                f"Missing identity adjudication: row={row_number}, molecule={name}"
            )
        adjudication = identity[name]
        if smiles != adjudication["smiles"]:
            raise ValueError(
                f"Source/adjudication SMILES mismatch: row={row_number}, molecule={name}"
            )

        identity_class = adjudication["class"]
        if identity_class in ("full_key_match", "largest_fragment_match"):
            continue
        if identity_class not in ("no_identity_link", "first_block_near_match_only"):
            raise ValueError(
                f"Unresolved identity adjudication: {name}, {identity_class}"
            )

        if name not in similarity:
            raise ValueError(f"Missing similarity adjudication: {name}")
        neighbor = similarity[name]
        if neighbor["identity_class"] != identity_class:
            raise ValueError(f"Identity/similarity adjudication mismatch: {name}")

        verdict = neighbor["cluster_verdict"]
        if verdict == "bridges_protected":
            continue
        if verdict != "cluster_clear":
            raise ValueError(f"Unresolved similarity adjudication: {name}, {verdict}")

        if enzyme not in groups:
            raise ValueError(
                f"Unexpected screen enzyme: row={row_number}, molecule={name}, enzyme={enzyme}"
            )
        pair = (name, enzyme)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate eligible compound/enzyme observation: {pair}")
        seen_pairs.add(pair)
        population.add(name)

        label_text = row["log2fc_estimate"].strip()
        if label_text.lower() in ("", "nan", "na", "null"):
            continue

        try:
            label = float(label_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid screen label: row={row_number}, molecule={name}, enzyme={enzyme}"
            ) from exc
        if not np.isfinite(label):
            raise ValueError(
                f"Non-finite screen label: row={row_number}, molecule={name}, enzyme={enzyme}"
            )

        try:
            se = float(row["log2fc_std_error"])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid observed SE: row={row_number}, molecule={name}, enzyme={enzyme}"
            ) from exc

        raw_weight = screen_observation_weights(
            [se],
            [True],
            [f"{name}:{enzyme}:row{row_number}"],
        )[0]
        groups[enzyme].append((name, smiles, label, raw_weight))

    if len(source_names) != contract["source_compounds"]:
        raise ValueError(
            f"Screen source compound count mismatch: {len(source_names)}"
        )
    missing_adjudications = source_names - set(identity)
    if missing_adjudications:
        raise ValueError(
            "Screen compounds missing from identity adjudications: "
            f"{sorted(missing_adjudications)}"
        )
    if len(population) != contract["eligible_compounds"]:
        raise ValueError(
            f"Eligible compound count mismatch: {len(population)}"
        )

    assays = []
    labelled_population = set()
    for enzyme in enzymes:
        records = groups[enzyme]
        if len(records) < 4:
            raise ValueError(
                f"Insufficient observed screening data for {enzyme}: n={len(records)}"
            )
        ids = tuple(record[0] for record in records)
        labelled_population.update(ids)
        assay = ActFoundAssay(
            assay_id=f"screen:{enzyme}",
            smiles=tuple(record[1] for record in records),
            y=np.asarray([record[2] for record in records], dtype=np.float32),
            molecule_ids=ids,
            observation_weights=np.asarray(
                [record[3] for record in records],
                dtype=np.float64,
            ),
        )
        if featurize:
            from rdkit import Chem

            bad = [
                name
                for name, smiles in zip(ids, assay.smiles)
                if Chem.MolFromSmiles(smiles) is None
            ]
            if bad:
                raise ValueError(f"Screen feature parse failures: {bad}")
            assay = assay.with_features(
                upstream_count_morgan_features(assay.smiles)
            )
        assays.append(assay)

    if labelled_population != population:
        raise ValueError(
            "Eligible compounds without any observed screening label: "
            f"{sorted(population - labelled_population)}"
        )

    result = tuple(assays)
    return result
