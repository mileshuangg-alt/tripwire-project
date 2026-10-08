# Tripwire-CYP-002 — Phase A Design Audit

**Status:** APPROVED / signed off by Miles on 2026-10-08
**Purpose:** Artifact-handling record for Step 2 before any held-out test access
**Test access during Phase A:** NONE
**Training/inference during Phase A:** NONE

## 1. Amended CYP-002 experiment contract

- Isoforms: CYP1A2, CYP2C9, CYP2D6, CYP3A4.
- Loss variants: stock masked multitask Chemprop and task-balanced masked multitask Chemprop.
- Sensitivity grid: C0-C7 x 2 variants, all 16 validation cells complete.
- Grid sensitivity statistic: `k/8 = 0.125`.
- Grid role: validation-only sensitivity analysis; it does not modify the headline C0 experiment.
- Headline experiment: C0 only.
- Headline test spend: exactly one held-out test evaluation for C0 / Variant 1 and exactly one for C0 / Variant 2.
- Metric: macro ST-RAE; lower is better.
- Headline uncertainty: paired cluster-bootstrap interval on the C0 test difference.
- Test set: remains untouched until explicit GO after Phase-B freeze.
- Decision burden: task-balanced must demonstrate both statistical separation and a material gain; otherwise stock remains headline.

## 2. Complete artifact/provenance audit

| Cell | Run path | Checkpoint SHA-256 | Metrics SHA-256 |
|---|---|---|---|
| C0 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C0-stock-seed20261001-r2/` | `89a3f37657e439258d3d6a8c650aceae4d4014951084baffaa4e536449d8b8f4` | `9a18a5bd8e5422defebf9d7f9f0b33441f7a557ee0dd98029b47fd6c2b8772bb` |
| C0 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C0-task-balanced-seed20261001-r1/` | `f082d67d3d7e2d3e243a148fa1b8a34326faecc239ec4c1b78b9c0dd39b3ef97` | `d823331fe0e3d4db6fa5e09c7a3756434bc4e56de40646929f0e45f0c8d08c6e` |
| C1 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C1-stock-seed20261001-r1/` | `4ac8f9749828da0df09a33f791806619957859a570e19482f23cf8fe53cfcd7f` | `3357c7f714c66b2fefbce110b33e485fc3c101a9212b0268ec9f5f889d5999ee` |
| C1 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C1-task-balanced-seed20261001-r1/` | `427ee45786750761cab437cdf00a5ab50d92a1699d8364b49868650e135fb412` | `86109a251964dbe6fc76e219ce47b5541cf33454ae7c878749fd6ee5ebba9fc5` |
| C2 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C2-stock-seed20261001-r1/` | `294fb2ee625559df473e8b95d5f1644b442b810c135ddbb8c36e90bd483a78bf` | `2e5f3446a64ca2e3f508848b91e58c763ff54a833923d94ddd83caf7548c0556` |
| C2 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C2-task-balanced-seed20261001-r1/` | **UNKNOWN** | **UNKNOWN** |
| C3 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C3-stock-seed20261001-r1/` | `92e2941a6863ac097de0aec870362ef59cf1a7e215a0266e1099e4cffa57498c` | `2c60a699df5fc2803c9bbff35fc62abc95f2eb9a4fb152d6ec62cfa2f3e899f2` |
| C3 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C3-task-balanced-seed20261001-r1/` | `2ac5d158da421310bb9d4bd797b16bbdb01bdae1329e38dd4cc0004899d736aa` | `f8aac1f304fc5a8fbd452794b7c4fbbf17316633f5d708b64b2e5c0e363ee43f` |
| C4 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C4-stock-seed20261001-r1/` | `8f04144035a2928e9c1ea44706ef74afcf5d893daa380f9d445a95aeaf37ddb6` | `916566a7d92e7b8518e8c880d54f4dc600a5c2aa9501607407451177ad7943fb` |
| C4 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C4-task-balanced-seed20261001-r1/` | `009f85b7c7f860b325c6f8739902caa8cd8361e190f2ea5af3408e089a30d2ee` | `f52455b435ff1bc7b7b6230c44eb6c3d6ddf09d738b51f8400d1473c5554a13b` |
| C5 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C5-stock-seed20261001-r1/` | `2057a211f4b63def828bd8e46ee6f3b82af0a32964c181d87309b8d037ef10a4` | `4235d876c274ddf57267abf161d595569fb31fccc31d8f8981f4803b247bfee5` |
| C5 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C5-task-balanced-seed20261001-r1/` | `246b48e776f2a22631f1fdd7b373fe73715bb81963ac8ebdcae1980ade927a74` | `6803e2af7605a3d363454984e58cb8d0bd3bf41c4118e7f578610e00f9fd31df` |
| C6 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C6-stock-seed20261001-r1/` | `3ed8bc60f1fd3cc13b8170915d314c47a8b8f9f0baf826c05a4ea643775e2117` | `b8dd17c94e02a44f490f4cc83ab9a8edc80dc1396e71d81175f66a59eb55bc01` |
| C6 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C6-task-balanced-seed20261001-r1/` | `288310353a77de7326f78168e257eab95ecd1180fbc3067af473efa7f09063f5` | `65c4f20facf123ecb01462efb8323d57b944cc456a42de39a5f0a3a604d59b0a` |
| C7 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C7-stock-seed20261001-r1/` | `2a2ed527cdb34cbee633cd29ffb209f670ed700796d90896dba2a243087a50d7` | `c033638b601190dd772b0dbec5094c18d08941e6af1b357516b95e3fadb62d6d` |
| C7 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C7-task-balanced-seed20261001-r1/` | `788bff18c199a85151554e8b3929e8c73f68fdcfd04eb31c1ad5ae833b63e740` | `81676743fa6fbec9a760d13021ad5d28a74fffb30e3b3d74578c6ae1cd2f438e` |

C2 / Variant 2 checkpoint and metrics hashes remain explicitly `UNKNOWN` and must not be inferred.

## 3. Assumptions and soft spots

- **Single seed:** all CYP-002 configurations use seed `20261001`; seed-to-seed variability is not characterized.
- **One-shot test spend:** exactly one C0 held-out evaluation per variant; no reruns and no second test evaluation.
- **Step-1 classification:** weak CYP1A2/CYP2D6 performance is more consistent with broad assay/label uncertainty than insufficient chemical coverage under the tested ECFP4 nearest-neighbor diagnostic. The basis is higher raw MAE in wide uncertainty bands (+0.134 for CYP1A2; +0.058 for CYP2D6) and a null nearest-neighbor chemical-distance relationship. Lower wide-band soft-thresholded error is mechanically expected under ST-RAE because wider intervals absorb more absolute error and is not evidence for either hypothesis.
- **Blind-set distribution shift:** measured prospectively by the held-out evaluation; not eliminated by the internal validation design.
- **Material gain:** required by the decision rule, but no numeric materiality threshold is introduced in this audit.
- **Paired cluster bootstrap:** the headline comparison must use the already-established paired cluster-bootstrap implementation; no new bootstrap implementation choices are introduced during the test phase.

## 4. Precommitted decision rule

Task-balanced takes the C0 headline only if the paired cluster-bootstrap interval on the C0 test difference excludes zero **and** the gain is material.

A tie or inconclusive interval keeps stock, because the challenger bears the burden of proof.

Operationally:

- Interval excludes zero and gain is material -> **Task-balanced**.
- Interval includes zero -> **Stock**.
- Statistically separated but non-material gain -> **Stock**.
- Tie / inconclusive -> **Stock**.

## 5. Phase-A sign-off

**Approved by Miles:** 2026-10-08.

Phase B may proceed: freeze the completed 16-cell grid summary, the Phase-A soft-spot list, and the precommitted decision rule into the project run record.

**Held-out test access remains prohibited until Phase B is written and Miles separately says GO.**
