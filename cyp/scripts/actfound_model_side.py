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

    def with_features(self, x: np.ndarray) -> "ActFoundAssay":
        if len(x) != len(self.y):
            raise ValueError("feature rows must align with assay labels")
        return ActFoundAssay(
            assay_id=self.assay_id,
            smiles=self.smiles,
            y=self.y,
            x=np.asarray(x, dtype=np.float32),
            graph_pair_count=self.graph_pair_count,
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.Xs[index]
        if x is None:
            raise ValueError("assay has no feature matrix; call from_tsv(..., featurize=True) or with_features")
        return make_episode_split(
            x,
            self.ys[index],
            test_sup_num=test_sup_num,
            is_test=is_test,
            rand_seed=rand_seed,
        )


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce BaseMetaDataset.get_split for the ActFound ChEMBL path."""
    rng = np.random.RandomState(seed=rand_seed)
    x = np.asarray(x_in, dtype=np.float32)
    y = np.asarray(y_in, dtype=np.float32)

    if len(x) > 64 and not is_test and datasource != "gdsc":
        subset_num = 64
        select_idx = [1] * subset_num + [0] * (len(x) - subset_num)
        rng.shuffle(select_idx)
        select_idx = np.nonzero(np.asarray(select_idx))[0]
        x = x[select_idx]
        y = y[select_idx]

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
    return x, y, np.asarray(split, dtype=np.int64)


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


def robust_square_error(a: torch.Tensor, b: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
    abs_diff = torch.abs(a - b)
    square_mask = (abs_diff <= 2).float()
    linear_mask = 1.0 - square_mask
    square_error = (a - b) ** 2
    linear_error = 1.0 * (abs_diff - 2) + 4
    loss = square_error * square_mask + linear_error * linear_mask
    return torch.mean(torch.gather(loss, 0, topk_idx))


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce ActFoundRegressor.net_forward after model forward."""
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
    softmax = torch.nn.Softmax(dim=0)
    rescale = 1.0 if is_training_phase else max(0.2, sup_y.std())

    if is_support:
        ddg_pred = (support_value.unsqueeze(-1) - support_value.unsqueeze(0)) * rescale
        sup_sim = jaccard_count_sim_matrix(sup_x, sup_x) - torch.eye(sup_num, device=x.device)
        _, topk_idx = torch.topk(sup_sim, dim=0, k=sup_sim.shape[0] // 2)
        embed_sim = cossim_matrix(support_features, support_features) / temp
        embed_sim = embed_sim - 1e9 * torch.eye(support_features.shape[0], device=x.device)
        embed_select = softmax(torch.gather(embed_sim, 0, topk_idx))
        preds_all = sup_y.unsqueeze(-1).repeat(1, sup_num) - ddg_pred
        preds = torch.sum(torch.gather(preds_all, 0, topk_idx) * embed_select, dim=0)
        loss = torch.mean((preds - sup_y) ** 2) / rescale ** 2
        return torch.sqrt(loss), preds

    ddg_pred_1 = support_value.unsqueeze(-1) - tgt_value.unsqueeze(0)
    ddg_real_1 = sup_y.unsqueeze(-1) - tgt_y.unsqueeze(0)
    ddg_pred_2 = tgt_value.unsqueeze(-1) - tgt_value.unsqueeze(0)
    ddg_real_2 = tgt_y.unsqueeze(-1) - tgt_y.unsqueeze(0)
    cross_sim = jaccard_count_sim_matrix(sup_x, tgt_x)
    _, topk_idx = torch.topk(cross_sim, dim=0, k=int(cross_sim.shape[0] * 0.5))
    embed_sim = cossim_matrix(support_features, query_features) / temp
    preds_all = sup_y.unsqueeze(-1).repeat(1, tgt_num) - ddg_pred_1 * rescale
    preds = torch.sum(
        torch.gather(preds_all, 0, topk_idx) * softmax(torch.gather(embed_sim, 0, topk_idx)),
        dim=0,
    )
    tgt_sim = jaccard_count_sim_matrix(tgt_x, tgt_x) - torch.eye(tgt_num, device=x.device)
    _, tgt_topk_idx = torch.topk(tgt_sim, dim=0, k=tgt_sim.shape[0] // 2)
    loss_2 = robust_square_error(ddg_pred_2, ddg_real_2, tgt_topk_idx)
    loss_1 = robust_square_error(ddg_pred_1, ddg_real_1, topk_idx)
    loss_dg = torch.mean((preds - tgt_y) ** 2)
    return torch.sqrt(loss_2 * 0.25 + loss_1 * 0.5 + loss_dg), preds


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
