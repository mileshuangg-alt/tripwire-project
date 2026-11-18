from pathlib import Path
import ast

ROOT = Path.cwd()
MODEL = ROOT / "cyp/scripts/actfound_model_side.py"
RUNNER = ROOT / "cyp/scripts/actfound_production_train_v2.py"
TEST = ROOT / "cyp/tests/test_actfound_weighting.py"

for path in (MODEL, RUNNER):
    if not path.is_file():
        raise SystemExit(f"Run from the repository root; missing {path}")
if TEST.exists():
    raise SystemExit(f"Refusing to overwrite existing test file: {TEST}")

model = MODEL.read_text()
runner = RUNNER.read_text()

# Refuse to proceed unless the already-qualified CoreHPC penalty patch is present.
penalty_contract = (
    'ACTFOUND_LR_PENALTY_PARAMETER = "layer_dict.linear.weights"',
    "def add_inner_learning_rate_penalty(",
    "learned_lr_parameters: Mapping[str, Any]",
    "torch.sum(lr_param * lr_param) * args.lrloss_weight",
    "result = (meta_loss + penalty, penalty)",
    '"layer_dict.linear.weights": inner_lrs',
)
missing = [token for token in penalty_contract if token not in runner]
if missing:
    raise SystemExit(
        "The CoreHPC-qualified penalty implementation is not present verbatim; "
        f"refusing to patch runner. Missing: {missing}"
    )

# ---------------------------------------------------------------------------
# Model-side data representation and metadata-preserving episode sampling.
# ---------------------------------------------------------------------------
old_assay = '''@dataclass(frozen=True)
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
'''
new_assay = '''@dataclass(frozen=True)
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
'''
if model.count(old_assay) != 1:
    raise SystemExit("ActFoundAssay source did not match expected clean version")
model = model.replace(old_assay, new_assay, 1)

model = model.replace(
    '        self.graph_pair_counts: list[Optional[int]] = []\n',
    '        self.graph_pair_counts: list[Optional[int]] = []\n'
    '        self.molecule_ids_all: list[tuple[str, ...]] = []\n'
    '        self.observation_weights_all: list[Optional[np.ndarray]] = []\n',
    1,
)
model = model.replace(
    '                        graph_pair_count=assay.graph_pair_count,\n'
    '                    )',
    '                        graph_pair_count=assay.graph_pair_count,\n'
    '                        molecule_ids=assay.molecule_ids,\n'
    '                        observation_weights=assay.observation_weights,\n'
    '                    )',
    1,
)
model = model.replace(
    '        self.graph_pair_counts.append(assay.graph_pair_count)\n',
    '        self.graph_pair_counts.append(assay.graph_pair_count)\n'
    '        self.molecule_ids_all.append(assay.molecule_ids)\n'
    '        self.observation_weights_all.append(assay.observation_weights)\n',
    1,
)

old_get_split = '''    def get_task_split(
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
'''
new_get_split = '''    def get_task_split(
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
'''
if model.count(old_get_split) != 1:
    raise SystemExit("get_task_split source did not match expected clean version")
model = model.replace(old_get_split, new_get_split, 1)

# Carry ChEMBL molecule identity through its existing graph-qualified adapter.
model = model.replace(
    '            smiles: list[str] = []\n            y_values: list[float] = []\n',
    '            smiles: list[str] = []\n'
    '            molecule_ids: list[str] = []\n'
    '            y_values: list[float] = []\n',
    1,
)
model = model.replace(
    '                smiles.append(row["canonical_smiles"])\n'
    '                y_values.append(_upstream_pic50(row["standard_value"]))\n',
    '                smiles.append(row["canonical_smiles"])\n'
    '                molecule_ids.append(str(row["molregno"]))\n'
    '                y_values.append(_upstream_pic50(row["standard_value"]))\n',
    1,
)
model = model.replace(
    '                y=y,\n'
    '                graph_pair_count=graph_pair_count,\n'
    '            )',
    '                y=y,\n'
    '                graph_pair_count=graph_pair_count,\n'
    '                molecule_ids=tuple(molecule_ids),\n'
    '            )',
    1,
)

# ---------------------------------------------------------------------------
# Raw observation weights and harmonic selected-pair weighting.
# ---------------------------------------------------------------------------
weight_functions = '''def curve_fit_observation_weights(
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


'''
anchor = "def robust_square_error("
if model.count(anchor) != 1:
    raise SystemExit("Cannot find unique robust_square_error insertion point")
model = model.replace(anchor, weight_functions + anchor, 1)

old_robust = '''def robust_square_error(a: torch.Tensor, b: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
    abs_diff = torch.abs(a - b)
    square_mask = (abs_diff <= 2).float()
    linear_mask = 1.0 - square_mask
    square_error = (a - b) ** 2
    linear_error = 1.0 * (abs_diff - 2) + 4
    loss = square_error * square_mask + linear_error * linear_mask
    return torch.mean(torch.gather(loss, 0, topk_idx))
'''
new_robust = '''def robust_square_error(
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
'''
if model.count(old_robust) != 1:
    raise SystemExit("robust_square_error source did not match expected clean version")
model = model.replace(old_robust, new_robust, 1)

old_loss_start = "def actfound_loss_and_preds(\n"
old_loss_end = "\n\ndef select_actfound_inner_loop_params("
if model.count(old_loss_start) != 1 or model.count(old_loss_end) != 1:
    raise SystemExit("actfound_loss_and_preds source boundaries not unique")
la = model.index(old_loss_start)
lb = model.index(old_loss_end, la)
new_loss = '''def actfound_loss_and_preds(
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
'''
model = model[:la] + new_loss + model[lb:]

# Extend the existing model adapter without changing existing callers.
old_forward_sig = '''    num_step: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal adapter for the real upstream FCNReLUNormNetworkQSAR forward path."""'''
new_forward_sig = '''    num_step: int = 0,
    observation_weights: Optional[torch.Tensor] = None,
    molecule_ids: Optional[Sequence[str]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal adapter for the real upstream FCNReLUNormNetworkQSAR forward path."""'''
if model.count(old_forward_sig) != 1:
    raise SystemExit("actfound_forward_with_regressor signature mismatch")
model = model.replace(old_forward_sig, new_forward_sig, 1)
old_forward_call = '''        is_support=is_support,
        is_training_phase=is_training_phase,
    )


def continuous_inference_from_outputs'''
new_forward_call = '''        is_support=is_support,
        is_training_phase=is_training_phase,
        observation_weights=observation_weights,
        molecule_ids=molecule_ids,
    )


def continuous_inference_from_outputs'''
if model.count(old_forward_call) != 1:
    raise SystemExit("actfound_forward_with_regressor forwarding point mismatch")
model = model.replace(old_forward_call, new_forward_call, 1)

# Expose original row indices while preserving the existing 3-value sampler API.
old_sampler_start = "def make_episode_split(\n"
old_sampler_end = "\n\ndef jaccard_count_sim_matrix"
sa = model.index(old_sampler_start)
sb = model.index(old_sampler_end, sa)
new_sampler = '''def make_episode_split(
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
'''
model = model[:sa] + new_sampler + model[sb:]

# The assay/weight metadata and sampler changes above preserve old default API.
ast.parse(model)

# ---------------------------------------------------------------------------
# Runner: request metadata, pass it into existing forward calls, preserve D013
# penalty function/call/logging by patching only task_arrays and run_task body.
# ---------------------------------------------------------------------------
old_task_arrays = '''def task_arrays(dataset: Any, index: int, args: SimpleNamespace, set_name: str, data_epoch: int) -> tuple[Any, Any, Any]:
    seed_base = {
        "train": args.train_seed,
        "val": args.val_seed,
    }[set_name]
    is_test = set_name != "train"
    return dataset.get_task_split(
        index,
        test_sup_num=args.test_sup_num,
        is_test=is_test,
        rand_seed=seed_base + index + data_epoch,
    )
'''
new_task_arrays = '''def task_arrays(dataset: Any, index: int, args: SimpleNamespace, set_name: str, data_epoch: int):
    seed_base = {
        "train": args.train_seed,
        "val": args.val_seed,
    }[set_name]
    is_test = set_name != "train"
    return dataset.get_task_split(
        index,
        test_sup_num=args.test_sup_num,
        is_test=is_test,
        rand_seed=seed_base + index + data_epoch,
        include_metadata=True,
    )
'''
if runner.count(old_task_arrays) != 1:
    raise SystemExit("task_arrays source mismatch; no runner changes written")
runner = runner.replace(old_task_arrays, new_task_arrays, 1)

old_unpack = '''    x_np, y_np, split_np = arrays
    x_task = torch.tensor(x_np).float().cuda()
    y_task = torch.tensor(y_np).float().cuda()
    split = torch.tensor(split_np).cuda()
'''
new_unpack = '''    if len(arrays) == 3:
        x_np, y_np, split_np = arrays
        observation_weights_np = None
        molecule_ids = None
    elif len(arrays) == 5:
        x_np, y_np, split_np, observation_weights_np, molecule_ids = arrays
    else:
        raise ValueError("ActFound task arrays must contain 3 legacy or 5 metadata fields")

    x_task = torch.tensor(x_np).float().cuda()
    y_task = torch.tensor(y_np).float().cuda()
    split = torch.tensor(split_np).cuda()
    observation_weights = None
    if is_training_phase and observation_weights_np is not None:
        observation_weights = torch.as_tensor(
            observation_weights_np,
            dtype=y_task.dtype,
            device=y_task.device,
        )
'''
if runner.count(old_unpack) != 1:
    raise SystemExit("run_task unpack block mismatch; no runner changes written")
runner = runner.replace(old_unpack, new_unpack, 1)

# Add metadata to each of the three existing forward calls. This does not edit
# optimizer, per-step weighting, task_losses aggregation, or penalty code.
for old, new, label in (
    (
        '            num_step=num_step,\n        )',
        '            num_step=num_step,\n'
        '            observation_weights=observation_weights,\n'
        '            molecule_ids=molecule_ids,\n'
        '        )',
        "support forward call",
    ),
    (
        '                    num_step=num_step,\n                )',
        '                    num_step=num_step,\n'
        '                    observation_weights=observation_weights,\n'
        '                    molecule_ids=molecule_ids,\n'
        '                )',
        "multi-step target forward call",
    ),
    (
        '        num_step=num_steps - 1,\n    )',
        '        num_step=num_steps - 1,\n'
        '        observation_weights=observation_weights,\n'
        '        molecule_ids=molecule_ids,\n'
        '    )',
        "final target forward call",
    ),
):
    if runner.count(old) != 1:
        raise SystemExit(f"{label} source mismatch; no files written")
    runner = runner.replace(old, new, 1)

# Guard the frozen penalty call/logging and core meta-learning aggregation.
for token in (
    "add_inner_learning_rate_penalty(\n                meta_loss,",
    '"layer_dict.linear.weights": inner_lrs',
    '"inner_lr_penalty": float(inner_lr_penalty.detach().cpu().item())',
    "torch.sum(torch.stack(task_losses))",
    "loss_weights[num_step] * target_loss",
):
    if token not in runner:
        raise SystemExit(f"Required existing runner behavior disappeared: {token}")

ast.parse(runner)

# ---------------------------------------------------------------------------
# Focused standard-library qualification tests.
# ---------------------------------------------------------------------------
test_source = r'''import unittest
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cyp" / "scripts"))

from actfound_model_side import (
    ActFoundAssay,
    ActFoundFrozenMetaDataset,
    actfound_loss_and_preds,
    construct_pair_matrices,
    curve_fit_observation_weights,
    harmonic_pair_weights,
    robust_square_error,
    screen_observation_weights,
    _weighted_direct_mse,
)


class TestActFoundHarmonicWeighting(unittest.TestCase):
    def test_raw_curve_fit_and_screen_weights(self):
        ci = curve_fit_observation_weights(
            [2.0, 3.0, np.nan], [3.0, 5.0, np.nan],
            [True, True, False], ["M1", "M2", "M3"],
        )
        np.testing.assert_allclose(ci[:2], [1.0 / 1.25, 1.0 / 2.25])
        self.assertTrue(np.isnan(ci[2]))
        se = screen_observation_weights(
            [0.5, 1.0, np.nan], [True, True, False], ["M1", "M2", "M3"]
        )
        np.testing.assert_allclose(se[:2], [4.0, 1.0])
        self.assertTrue(np.isnan(se[2]))

    def test_invalid_observed_uncertainty_names_rows(self):
        with self.assertRaisesRegex(ValueError, "BAD_CI"):
            curve_fit_observation_weights([2.0], [2.0], [True], ["BAD_CI"])
        with self.assertRaisesRegex(ValueError, "BAD_SE"):
            screen_observation_weights([0.0], [True], ["BAD_SE"])

    def test_harmonic_cross_weights_and_selected_orientation(self):
        wa = torch.tensor([1.0, 2.0, 4.0])
        wb = torch.tensor([3.0, 6.0])
        raw = harmonic_pair_weights(wa, wb)
        expected_matrix = torch.tensor([
            [3.0 / 4.0, 6.0 / 7.0],
            [6.0 / 5.0, 12.0 / 8.0],
            [12.0 / 7.0, 24.0 / 10.0],
        ])
        torch.testing.assert_close(raw, expected_matrix)

        errors = torch.tensor([[1.0, 3.0], [4.0, 0.0], [2.0, 1.0]])
        zeros = torch.zeros_like(errors)
        idx = torch.tensor([[2, 0], [1, 2]])
        selected = torch.gather(raw, 0, idx)
        manual = torch.gather(torch.where(errors.abs() <= 2, errors.square(), errors.abs() + 2), 0, idx)
        normalized = selected / selected.mean()
        expected = torch.mean(manual * normalized)
        actual = robust_square_error(
            errors, zeros, idx,
            observation_weights_a=wa,
            observation_weights_b=wb,
        )
        torch.testing.assert_close(actual, expected)

    def test_query_query_harmonic_weights_and_selected_orientation(self):
        w = torch.tensor([2.0, 4.0, 8.0])
        raw = harmonic_pair_weights(w, w)
        expected = torch.tensor([
            [1.0, 4.0 / 3.0, 8.0 / 5.0],
            [4.0 / 3.0, 2.0, 8.0 / 3.0],
            [8.0 / 5.0, 8.0 / 3.0, 4.0],
        ])
        torch.testing.assert_close(raw, expected)
        errors = torch.tensor([[0.0, 1.0, 4.0], [2.0, 3.0, 0.0], [1.0, 5.0, 2.0]])
        idx = torch.tensor([[2, 0, 1], [1, 2, 0]])
        selected_w = torch.gather(raw, 0, idx)
        error_matrix = torch.where(errors.abs() <= 2, errors.square(), errors.abs() + 2)
        selected_e = torch.gather(error_matrix, 0, idx)
        expected_loss = torch.mean(selected_e * (selected_w / selected_w.mean()))
        actual = robust_square_error(
            errors, torch.zeros_like(errors), idx,
            observation_weights_a=w,
            observation_weights_b=w,
        )
        torch.testing.assert_close(actual, expected_loss)

    def test_unequal_partition_scales_are_not_normalized_before_pairing(self):
        support = torch.tensor([1.0, 100.0])
        query = torch.tensor([10.0, 10.0])
        combined = harmonic_pair_weights(support, query)
        separately_normalized = harmonic_pair_weights(
            support / support.mean(), query / query.mean()
        )
        self.assertFalse(torch.allclose(combined, separately_normalized))

    def test_uniform_weights_recover_original_support_and_query_losses(self):
        torch.manual_seed(412)
        x = torch.randint(0, 2, (8, 32)).float()
        y = torch.linspace(0.5, 4.0, 8)
        split = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
        embeddings = torch.randn(8, 12)
        values = torch.randn(8, requires_grad=True)
        for is_support in (True, False):
            base, base_preds = actfound_loss_and_preds(
                x, y, split, embeddings, values,
                is_support=is_support, is_training_phase=True,
            )
            weighted, weighted_preds = actfound_loss_and_preds(
                x, y, split, embeddings, values,
                is_support=is_support, is_training_phase=True,
                observation_weights=torch.ones_like(y),
                molecule_ids=tuple(f"M{i}" for i in range(len(y))),
            )
            torch.testing.assert_close(weighted, base, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(weighted_preds, base_preds)

    def test_uniform_weights_recover_each_direct_and_pair_reduction(self):
        predictions = torch.tensor([1.0, 3.0, 2.0])
        targets = torch.tensor([0.0, 1.0, 4.0])
        original_direct = torch.mean((predictions - targets) ** 2)
        weighted_direct = _weighted_direct_mse(
            predictions, targets, torch.ones_like(targets)
        )
        torch.testing.assert_close(weighted_direct, original_direct)

        cross_pred = torch.tensor([[0.0, 3.0], [2.0, 1.0], [4.0, 0.0]])
        cross_true = torch.tensor([[1.0, 0.0], [0.0, 3.0], [2.0, 1.0]])
        cross_idx = torch.tensor([[2, 0], [1, 2]])
        cross_base = robust_square_error(cross_pred, cross_true, cross_idx)
        cross_weighted = robust_square_error(
            cross_pred, cross_true, cross_idx,
            observation_weights_a=torch.ones(3),
            observation_weights_b=torch.ones(2),
        )
        torch.testing.assert_close(cross_weighted, cross_base)

        query_pred = torch.tensor([[0.0, 3.0, 2.0], [2.0, 1.0, 4.0], [4.0, 0.0, 1.0]])
        query_true = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0], [2.0, 1.0, 4.0]])
        query_idx = torch.tensor([[2, 0, 1], [1, 2, 0]])
        query_base = robust_square_error(query_pred, query_true, query_idx)
        query_weighted = robust_square_error(
            query_pred, query_true, query_idx,
            observation_weights_a=torch.ones(3),
            observation_weights_b=torch.ones(3),
        )
        torch.testing.assert_close(query_weighted, query_base)

    def test_metadata_alignment_and_sampling_are_unchanged(self):
        ids = tuple(f"M{i:02d}" for i in range(12))
        x = np.arange(12 * 32, dtype=np.float32).reshape(12, 32) % 2
        y = np.arange(12, dtype=np.float32)
        raw = np.arange(1, 13, dtype=np.float64)
        assay = ActFoundAssay(
            assay_id="A", smiles=tuple("C" for _ in ids), y=y, x=x,
            molecule_ids=ids, observation_weights=raw,
        )
        ds = ActFoundFrozenMetaDataset([assay], val_assays={"A"})
        first = ds.get_task_split(
            0, test_sup_num=0.5, is_test=True, rand_seed=919, include_metadata=True
        )
        second = ds.get_task_split(
            0, test_sup_num=0.5, is_test=True, rand_seed=919, include_metadata=True
        )
        self.assertEqual(len(first), 5)
        for left, right in zip(first[:4], second[:4]):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(first[4], second[4])
        xx, yy, split, ww, mm = first
        for row_x, row_y, row_w, molecule_id in zip(xx, yy, ww, mm):
            i = ids.index(molecule_id)
            np.testing.assert_array_equal(row_x, x[i])
            self.assertEqual(row_y, y[i])
            self.assertEqual(row_w, raw[i])

        p1 = construct_pair_matrices(torch.tensor(xx), torch.tensor(yy), torch.tensor(split))
        p2 = construct_pair_matrices(torch.tensor(xx), torch.tensor(yy), torch.tensor(split))
        torch.testing.assert_close(p1.support_target_topk_idx, p2.support_target_topk_idx)
        torch.testing.assert_close(p1.target_target_topk_idx, p2.target_target_topk_idx)

    def test_missing_labels_remain_masked_and_invalid_observed_weights_fail(self):
        assay = ActFoundAssay(
            assay_id="A", smiles=("C", "CC"),
            y=np.asarray([1.0, np.nan]),
            molecule_ids=("OBS", "MISSING"),
            observation_weights=np.asarray([2.0, np.nan]),
        )
        self.assertTrue(np.isnan(assay.y[1]))
        self.assertTrue(np.isnan(assay.observation_weights[1]))
        with self.assertRaisesRegex(ValueError, "BAD"):
            ActFoundAssay(
                assay_id="B", smiles=("C",), y=np.asarray([1.0]),
                molecule_ids=("BAD",), observation_weights=np.asarray([np.nan]),
            )

    def test_weighting_keeps_finite_model_gradients(self):
        torch.manual_seed(83)
        x = torch.randint(0, 2, (8, 32)).float()
        y = torch.linspace(0.5, 4.0, 8)
        split = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
        embeddings = torch.randn(8, 12, requires_grad=True)
        values = torch.randn(8, requires_grad=True)
        raw = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
        loss, _ = actfound_loss_and_preds(
            x, y, split, embeddings, values,
            is_support=False, is_training_phase=True,
            observation_weights=raw,
            molecule_ids=tuple(f"M{i}" for i in range(8)),
        )
        loss.backward()
        self.assertIsNotNone(embeddings.grad)
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertFalse(raw.requires_grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''

ast.parse(test_source)

# Final source guards and writes. No backups are created.
if "observation_weights: Optional[torch.Tensor] = None" not in model:
    raise SystemExit("model-side observation-weight API missing after patch construction")
if "harmonic_pair_weights" not in model or "selected_raw" not in model:
    raise SystemExit("harmonic selected-pair weighting missing after patch construction")
if "include_metadata=True" not in runner:
    raise SystemExit("runner metadata route missing after patch construction")

MODEL.write_text(model)
RUNNER.write_text(runner)
TEST.parent.mkdir(parents=True, exist_ok=True)
TEST.write_text(test_source)

print("UPDATED", MODEL.relative_to(ROOT))
print("UPDATED", RUNNER.relative_to(ROOT))
print("CREATED", TEST.relative_to(ROOT))
print("PRESERVED: CoreHPC-qualified ActFound penalty function, call, and logging")
print("PRESERVED: pair construction top-k selection, step importance, task aggregation")
print("No backup files created.")
