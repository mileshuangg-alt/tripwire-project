# D012 Arms B/C Qualification Package

This directory contains bounded local correctness and prepared runtime benchmark artifacts for D012 Arms B and C.

No production training, 150-cell matrix run, tuning, D011 change, or SLURM submission is authorized by these files.

Prepared first-cell production commands, after syncing this repository to `/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project` and ensuring `/scratch/user/mhuang8/envs/tripwire_cyp002` exists:

```bash
cd /scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project
sbatch qualification/cyp_external_task_qualification_20261109/d012_arm_qualification/corehpc/d012_first_cell_arm_b.slurm
sbatch qualification/cyp_external_task_qualification_20261109/d012_arm_qualification/corehpc/d012_first_cell_arm_c.slurm
```

Expected first-cell artifact directories:

`/scratch/user/mhuang8/d012_runtime_qualification_20261110/artifacts/d012_first_cell_arm_b_rep01_fold01_seed20261101`

`/scratch/user/mhuang8/d012_runtime_qualification_20261110/artifacts/d012_first_cell_arm_c_rep01_fold01_seed20261101`

Prepared remaining-production array commands. These exclude exactly the completed first cell `rep_01_fold_01 / 20261101` and cover the remaining 149 frozen cells per arm:

```bash
cd /scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project
sbatch qualification/cyp_external_task_qualification_20261109/d012_arm_qualification/corehpc/d012_remaining_arm_b_array.slurm
sbatch qualification/cyp_external_task_qualification_20261109/d012_arm_qualification/corehpc/d012_remaining_arm_c_array.slurm
```

Array artifact roots:

`/scratch/user/mhuang8/d012_runtime_qualification_20261110/artifacts/d012_remaining/arm_b/{manifest_id}_seed{training_seed}`

`/scratch/user/mhuang8/d012_runtime_qualification_20261110/artifacts/d012_remaining/arm_c/{manifest_id}_seed{training_seed}`

Restart contract: an array task skips only when both `completion.json` and `SHA256SUMS.txt` exist. Failed or incomplete cells are never counted complete, partial artifacts are preserved, and rerun requires explicit array task selection. No automatic retry changes scientific behavior.

Prepared CoreHPC staging/submission command, after syncing this repository to `/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project` and ensuring `/scratch/user/mhuang8/envs/tripwire_cyp002` exists:

```bash
cd /scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project
sbatch qualification/cyp_external_task_qualification_20261109/d012_arm_qualification/corehpc/d012_bounded_runtime_benchmark.slurm
```

Expected benchmark artifact:

`/scratch/user/mhuang8/d012_runtime_qualification_20261110/artifacts/d012_chemprop_dmpnn_complete_model_runtime_benchmark_rep01_fold01_seed20261101.json`

The benchmark uses the actual D012 Chemprop 2.3.1 D-MPNN path, not the bounded ECFP4 correctness model. It constructs Chemprop molecular graph datasets from frozen SMILES, uses depth-4 BondMessagePassing with hidden dimension 600, NormAggregation, 2000-hidden-unit scalar heads, dropout 0.4, and the D012-qualified B/C epoch plans.

The benchmark uses the established complete-update timing protocol: 10 continuous warmup updates followed by 100 measured updates, no optimizer/scheduler reset between warmup and measurement, CUDA synchronization around measured timing, and peak CUDA allocated/reserved memory capture. It records separate Arm B auxiliary, Arm B target-transfer, Arm B validation-scoring, Arm C auxiliary exact-ANIL, and Arm C target-adaptation/validation timing evidence and projects 50-epoch/150-cell workload cost from the D012-qualified step budgets.
