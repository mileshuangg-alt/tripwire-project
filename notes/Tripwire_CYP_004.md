# Tripwire-CYP-004 Session Log

**Date:** 2026-11-06  
**Status:** D008 completed and frozen

## 1. Session purpose

Session 004 began after authoritative OpenADMET CYP test labels were released.

The session's governing sequence was:

1. preserve the organizer-released labels unchanged;
2. freeze the external-evaluation scoring contract before label/prediction scoring;
3. qualify the external evaluator without exposing external performance;
4. execute the already-frozen D007-T121 predictions against the released labels once;
5. record the result without reopening or adapting D007.

D007 remained closed throughout Session 004.

## 2. Starting state

D007-T121 was already frozen as the preferred Chemprop development configuration.

Authoritative D007 development objective:

- D007-T121 O150: `0.701212576796`
- C0 O150: `0.718103374388`

The frozen T121 blind-inference ensemble contained exactly 150 checkpoints.

The frozen blinded challenge prediction artifact had already been generated before authoritative label release:

`cyp/artifacts/cyp_007/t121_blind_inference_v1/prediction_attempt_003/ensemble_predictions.csv`

SHA-256:

`51002c6dbe75041bf4c874f1a4dce4d59f293e82d9d8e4771a2132160da7e7ca`

The external evaluation therefore did not require or authorize inference regeneration.

## 3. Organizer label release preservation

The OpenADMET unblinded release was downloaded from the organizer-published OpenADMET Hugging Face dataset repository.

The release was preserved locally under:

`cyp/data/openadmet_unblinded_release_2026-11-06/`

The authoritative direct-inhibition label artifact was:

`cyp-challenge-TEST-UNBLINDED_inhibition.csv`

SHA-256:

`6551eba506c90cf51434bc1764424e928cfa19b45854148d692247b8eb49f74c`

Companion release artifacts were also preserved unchanged:

- `cyp-challenge-TEST-UNBLINDED_Emax.csv`
  - SHA-256 `104c6466382b61eaadb45c7a61c3906aa3e78cae9b7f5bf21183307564cd543c`
- `cyp-challenge-TEST-UNBLINDED_TDI.csv`
  - SHA-256 `b88b67961a1ae839be9eec25d4edc608a1ccb748dc61a4af75bd36810b994a3c`
- `openadmet-unblinded-structures.zip`
  - SHA-256 `ec5519262571c29929d935a15bfe70ab09cf2045a64ac9f6491116a8b8be0a9a`

Source/copy SHA-256 equality passed before scoring.

## 4. D008

Session 004 created and froze:

**D008 — Post-Deadline Blind External Evaluation of Frozen D007-T121**

D008 was frozen before external label/prediction scoring.

The pre-scoring `notes/Decisions.md` SHA-256 was:

`d27074594a7e57e4e93c739eee32369142019a0e9b0e9566c27fce492ef1e0fb`

D008 defined:

- primary external performance as full-test-set per-endpoint ST-RAE and the unweighted arithmetic mean across the four CYP endpoints;
- the existing qualified project `st_rae_metric` as the authoritative metric implementation;
- no new or alternative ST-RAE definition;
- organizer-compatible 1,000-resample bootstrap mean and standard deviation as secondary characterization;
- no transfer of the project's earlier paired cluster-bootstrap or paired-seed inference procedures to this absolute external-performance question;
- strict identity, missing-label, schema, and integrity handling;
- qualification before production scoring;
- zero authority for the external result to alter D007 or T121.

## 5. ST-RAE methodology clarification

Repository audit established that the existing Tripwire ST-RAE implementation already directly wraps the preserved upstream OpenADMET:

`rae_soft_threshold_absolute_error`

through:

`cyp/scripts/st_rae.py`

Therefore the project did not have a separate training-mean ST-RAE definition.

The earlier project-specific 10,000-draw bootstrap procedures were separate inferential procedures designed for paired model-comparison questions. They were not part of the ST-RAE metric itself and were not transferred to D008.

## 6. D008 implementation and qualification

D008 implementation proceeded incrementally using synthetic data only before external scoring.

Qualified behavior included:

1. molecular identity validation and row alignment;
2. endpoint-specific eligibility/routing;
3. endpoint scoring by delegation to the existing `st_rae_metric`;
4. unweighted four-endpoint macro ST-RAE aggregation;
5. organizer-compatible bootstrap characterization;
6. explicit-path production I/O and SHA/schema integrity validation.

Before production exposure:

- evaluator SHA-256:
  `ad2f38a60bc8ba620a4db94d23b83e6691201059c2da147d2dee4cad4543ebbb`
- tests SHA-256:
  `e812481978bb907d3e4d100b56a19b199dbe0fd4573bbcffe072344317ed225a`
- qualification:
  `58 passed`

No unblinded external performance was accessed during implementation qualification.

## 7. CoreHPC execution environment

D007 production artifacts remained on CoreHPC.

The frozen T121 ensemble prediction artifact was located at:

`/scratch/user/mhuang8/tripwire-d006/d006_bundle/cyp/artifacts/cyp_007/t121_blind_inference_v1/prediction_attempt_003/ensemble_predictions.csv`

The D008 evaluator, tests, and authoritative inhibition labels were transferred to CoreHPC and their frozen SHA-256 identities were verified after transfer.

Initial D008 execution attempts encountered environment/import failures before scientific evaluation:

- system Python lacked NumPy;
- an attempted disposable venv inherited CoreHPC system Python 3.9.25;
- Python 3.9 could not evaluate the preserved upstream scorer's modern union type annotations.

These attempts did not produce external performance.

CoreHPC's CBI software stack was then loaded using:

`module load CBI`
`module load python/3.14.8`

The actual interpreter was:

`/mnt/software/cbi/software/python-3.14.8/bin/python`

A dedicated D008 evaluation venv was created from Python 3.14.8 and qualified against the frozen D008 tests before production evaluation.

## 8. Attempt 001 serialization issue

The first scientifically executing production attempt completed the frozen D008 calculation in memory but the ad hoc output wrapper failed while applying `json.dump()` directly to the complete evaluator result because that result contained pandas `DataFrame` objects.

No external-performance value from Attempt 001 was printed or inspected.

The incomplete serialization artifact was preserved as:

`D008_EXTERNAL_EVALUATION_SERIALIZATION_FAILED.json.partial`

The issue was operational output serialization, not a change to the scientific calculation.

A serialization/reporting layer was then implemented and qualified. It does not recalculate ST-RAE, routing, macro aggregation, or bootstrap quantities.

Final serialization-capable identities:

- evaluator SHA-256:
  `bdb934b03a2ece685aea7c021e3d4d1221c4f28c6e51a2acc4e8d408bd52ccde`
- tests SHA-256:
  `68d4853897a46935090a90021bcf8a3c1f39f4c3c84170a452f655e8f16efbb9`
- local qualification:
  `59 passed`
- CoreHPC qualification:
  `59 passed`

The recovery execution repeated the identical deterministic scientific calculation solely to persist the result through the qualified serialization layer.

## 9. D008 primary external result

The authoritative full-test-set external results were:

| Endpoint | n scored | n missing | ST-RAE |
|---|---:|---:|---:|
| CYP1A2 | 748 | 2 | `0.7728950609716935` |
| CYP2C9 | 750 | 0 | `0.5558588275060217` |
| CYP2D6 | 748 | 2 | `1.217061477884744` |
| CYP3A4 | 748 | 2 | `0.5078206616488673` |

Primary macro ST-RAE:

`0.7634090070028317`

The result is heterogeneous across endpoints. CYP2C9 and CYP3A4 had the lowest external ST-RAE, CYP1A2 was intermediate, and CYP2D6 was the weakest endpoint and exceeded `1.0`.

D008 assigns no causal explanation to this endpoint heterogeneity.

## 10. Organizer-compatible bootstrap characterization

Frozen bootstrap contract:

- iterations: `1000`
- RNG: `np.random.default_rng(seed=0)`
- standard deviation: NumPy population SD (`ddof=0`)

Results:

| Endpoint | Mean | SD |
|---|---:|---:|
| CYP1A2 | `0.7745453137726171` | `0.02728147930356748` |
| CYP2C9 | `0.5571939927014358` | `0.020433515126421893` |
| CYP2D6 | `1.2172319746112334` | `0.05332503578021169` |
| CYP3A4 | `0.5084294889718826` | `0.02156381142675119` |
| Macro | `0.7643501925142921` | `0.016520750512073138` |

These are secondary organizer-compatible characterizations, not the project's historical cluster-bootstrap confidence intervals.

## 11. Frozen D008 result

Authoritative serialized result:

`d008_external_evaluation_v1/D008_EXTERNAL_EVALUATION.json`

SHA-256:

`b9f9beeaaa36b6e9bedd65740f966b67bd1958e9c09299c05926dbb6558e3b30`

This is the authoritative D008 result.

## 12. Claim boundary and closure

The result is a:

**post-deadline blind external evaluation of the frozen D007-T121 ensemble on the released OpenADMET CYP direct-inhibition test set.**

It is not an official OpenADMET challenge submission, official challenge score, or prospective leaderboard result.

D008 does not reopen D007.

No residual analysis, subgroup analysis, alternative metric, recalibration, endpoint-specific rescue analysis, or other result-driven post-hoc investigation was performed under D008.

The D007 O150 development objective and D008 external macro ST-RAE are different estimands under different evaluation structures and are not interpreted as a paired generalization-gap statistic.

D008 is complete and frozen.

The final `notes/Decisions.md` SHA-256 after D008 completion was:

`7e23056e328af45d1a3dbf48de0e657bbb36919365c937953dd9784fa6cadb6a`

## 13. Prospective exploratory branch — not yet executed

After D008 closure, a separate exploratory diagnostic branch was proposed but not executed.

Its purpose is to distinguish among three broad explanations using existing artifacts first:

1. implementation/integrity problem;
2. pre-existing weak or unstable development performance, particularly endpoint-specific weakness;
3. external distribution shift.

The branch has no authority to change D007 or D008.

Proposed evidence order:

1. reproduce/check implementation invariants using frozen existing artifacts;
2. examine already-produced D007 per-endpoint development evidence;
3. perform labels-independent chemistry/domain-shift characterization using existing development and external structures/predictions;
4. only after a separate prospective plan, consider analyses involving the now-revealed external labels.

Any future diagnostic work must state in advance what decision each check can change. External-label-driven diagnosis cannot retrospectively acquire D008 validation authority.


## 15. Post-D008 diagnostic Lane 1 — Check 1 result

**Status:** Partial pass; numerical component unresolved; I1 not cleared

Lane-1 Check 1 audited the preserved D007-T121 blind-inference ensemble using existing artifacts only.

The historically recorded prediction artifacts were present on CoreHPC and matched their recorded SHA-256 identities:

- `member_predictions.csv`:
  `f7e683c7ab8573762d19231e092fe7d48e708c9e572d526db12a4c502c02e8cb`
- `ensemble_predictions.csv`:
  `51002c6dbe75041bf4c874f1a4dce4d59f293e82d9d8e4771a2132160da7e7ca`

The preserved checkpoint inventory and construction/freeze evidence were also available.

### Membership

**PASS**

The frozen inventory contained exactly 150 intended unique `(repetition, fold, seed)` identities.

The preserved member-prediction artifact contained exactly the same 150 unique member identities:

- intended unique members: `150`
- observed unique members: `150`
- missing intended members: `0`
- unintended members: `0`
- duplicate molecule/member records: `0`

Membership was established by exact identity against the frozen inventory, not by member count alone.

### Molecule and endpoint coverage

**PASS**

The preserved ensemble contained `750` unique molecule identities.

Every intended member covered exactly the same 750-molecule ensemble population. No missing or unexpected molecule coverage was detected for any member.

The member and ensemble artifacts contained the four required endpoint columns:

- `CYP1A2_pIC50_direct_inhibition`
- `CYP2C9_pIC50_direct_inhibition`
- `CYP2D6_pIC50_direct_inhibition`
- `CYP3A4_pIC50_direct_inhibition`

No coverage failure was detected.

### Numerical member-to-ensemble consistency

**UNRESOLVED**

The preserved construction evidence establishes that the original ensemble mean was calculated in memory from the 150-member prediction cube before the member and ensemble artifacts were independently serialized.

However, preserved provenance does not establish enough numerical detail to derive a justified quantitative criterion relating a mean reconstructed from the separately serialized member values to the separately serialized ensemble values.

Relevant unresolved provenance includes the original prediction-tensor dtype and sufficient information to bound separately:

- member-value conversion/serialization effects;
- arithmetic error of the original NumPy mean relative to the exact mathematical mean;
- arithmetic error of any reconstruction relative to the same exact mathematical mean;
- ensemble-value serialization effects.

No empirical tolerance or permissive dtype assumption was introduced.

No production prediction values were compared numerically.

The numerical component is therefore closed for this check as:

`UNRESOLVED — preserved provenance insufficient to justify a quantitative comparison criterion.`

### Check-1 interpretation

Check 1 is a **partial pass**, not a full integrity clearance.

The preserved evidence supports:

- exact frozen-member membership;
- complete member/molecule coverage;
- required endpoint-schema coverage.

It does not support a quantitative member-to-ensemble reconstruction claim.

Therefore:

- no membership or coverage defect was demonstrated;
- no material defect requiring adjudication was demonstrated by the evidence-supported portions of Check 1;
- numerical consistency remains unresolved;
- Check 1 does not clear overall Lane-1 decision I1;
- I1 remains `NOT CLEARED`.

This unresolved numerical component is not interpreted as evidence that the original D008 calculation was wrong, and it is not silently converted into a pass.

No new inference, training, fitting, calibration, model selection, or external-label analysis was performed for Check 1.
