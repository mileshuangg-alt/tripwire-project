"""Tripwire PXR-001 compact data audit of the preserved OpenADMET PXR release.

Read-only. The script reads the preserved raw files, computes their SHA-256,
and writes a plain-text report to stdout. It modifies nothing and it applies
no exclusion, filter or transformation to the data.

The held-out label files are read for IDENTITY COLUMNS AND ROW COUNTS ONLY.
No label value is read, aggregated or printed by this script.

Usage (from the repository root):
    python3 -I pxr/scripts/pxr001_data_audit.py \
        pxr/data/openadmet_pxr_release_2026-11-15 \
        pxr/data/openadmet_pxr_heldout_labels_2026-11-15
"""
import hashlib
import sys
from collections import Counter

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

DEV_FILES = [
    "pxr-challenge_TRAIN.csv",
    "pxr-challenge_TEST_BLINDED.csv",
    "pxr-challenge_counter-assay_TRAIN.csv",
    "pxr-challenge_single_concentration_TRAIN.csv",
    "pxr-challenge_htchem-libraries_TRAIN.csv",
    "pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv",
    "pxr-challenge_DRC_plot_data_TRAIN.parquet",
]
HELDOUT_FILES = [
    "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    "pxr-challenge_TEST_PHASE_2_UNBLINDED.csv",
]
# identity columns read from the firewalled files; no label column appears here
HELDOUT_IDENTITY_COLS = ["Molecule Name", "SMILES", "phase"]

PEC50 = "pEC50"
EMAX = "Emax_estimate (log2FC vs. baseline)"
EMAX_POS = "Emax.vs.pos.ctrl_estimate (dimensionless)"
SE = "pEC50_std.error (-log10(molarity))"
CI_LO = "pEC50_ci.lower (-log10(molarity))"
CI_HI = "pEC50_ci.upper (-log10(molarity))"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path, **kw):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, **kw)


def hdr(title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def structural_keys(df):
    """Canonical SMILES, InChIKey and InChIKey connectivity block; None on parse failure."""
    canon, ikey, block, fails = [], [], [], []
    for idx, smi in enumerate(df["SMILES"]):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            fails.append(idx)
            canon.append(None)
            ikey.append(None)
            block.append(None)
            continue
        canon.append(Chem.MolToSmiles(mol))
        key = Chem.MolToInchiKey(mol)
        ikey.append(key)
        block.append(key.split("-")[0] if key else None)
    return (pd.Series(canon, index=df.index), pd.Series(ikey, index=df.index),
            pd.Series(block, index=df.index), fails)


def main(dev_dir, heldout_dir):
    print("Tripwire PXR-001 compact data audit")
    print("Read-only. No exclusion, filter or transformation is applied by this script.")

    hdr("0. INPUT FILES AND SHA-256")
    for name in DEV_FILES:
        print(f"  {sha256(f'{dev_dir}/{name}')}  {dev_dir}/{name}")
    print("\n  Held-out label files (identity columns and row counts only):")
    for name in HELDOUT_FILES:
        print(f"  {sha256(f'{heldout_dir}/{name}')}  {heldout_dir}/{name}")

    train = read(f"{dev_dir}/pxr-challenge_TRAIN.csv")
    test = read(f"{dev_dir}/pxr-challenge_TEST_BLINDED.csv")
    counter = read(f"{dev_dir}/pxr-challenge_counter-assay_TRAIN.csv")
    single = read(f"{dev_dir}/pxr-challenge_single_concentration_TRAIN.csv")
    crude = read(f"{dev_dir}/pxr-challenge_htchem-libraries_TRAIN.csv")
    semi = read(f"{dev_dir}/pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv")
    drc = read(f"{dev_dir}/pxr-challenge_DRC_plot_data_TRAIN.parquet")
    p1 = read(f"{heldout_dir}/{HELDOUT_FILES[0]}", usecols=HELDOUT_IDENTITY_COLS)
    p2 = read(f"{heldout_dir}/{HELDOUT_FILES[1]}", usecols=HELDOUT_IDENTITY_COLS)

    hdr("1. SCHEMA OF THE PRIMARY TABLE")
    print(f"  pxr-challenge_TRAIN.csv  rows={len(train)}  cols={train.shape[1]}")
    for c in train.columns:
        s = train[c]
        print(f"    {c:<44} {str(s.dtype):<9} nonnull={s.notna().sum():>6}/{len(train)}  nunique={s.nunique():>6}")
    print(f"\n  pxr-challenge_TEST_BLINDED.csv  rows={len(test)}  columns={list(test.columns)}")
    print(f"  TRAIN 'Split' column values: {sorted(train['Split'].unique())}")
    print("  The release carries no internal train/validation partition; the Split column is constant.")

    hdr("2. ROW AND IDENTITY COUNTS")
    rows = []
    for label, df, idcols in [
        ("TRAIN (primary pEC50/Emax)", train, ["Molecule Name", "OCNT_ID", "SMILES"]),
        ("TEST_BLINDED", test, ["Molecule Name", "SMILES"]),
        ("counter-assay TRAIN", counter, ["Molecule Name", "OCNT_ID", "SMILES"]),
        ("single-concentration TRAIN", single, ["Molecule Name", "OCNT_ID", "SMILES"]),
        ("htchem crudes TRAIN", crude, ["OCNT_ID", "SMILES"]),
        ("semi-pure TRAIN", semi, ["OCNT_ID", "SMILES"]),
        ("DRC curves TRAIN", drc, ["Molecule Name", "OCNT_ID", "SMILES"]),
    ]:
        r = {"file": label, "rows": len(df)}
        for c in ["Molecule Name", "OCNT_ID", "SMILES"]:
            r[f"uniq {c}"] = df[c].nunique() if c in idcols else "-"
            r[f"null {c}"] = int(df[c].isna().sum()) if c in idcols else "-"
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    hdr("3. STRUCTURE PARSE (RDKit) AND STRUCTURAL KEYS")
    keys = {}
    for label, df in [("TRAIN", train), ("TEST_BLINDED", test), ("counter-assay", counter),
                      ("htchem crudes", crude), ("semi-pure", semi),
                      ("PHASE_1 (identity only)", p1), ("PHASE_2 (identity only)", p2)]:
        canon, ikey, block, fails = structural_keys(df)
        keys[label] = (canon, ikey, block)
        print(f"  {label:<26} n={len(df):<6} parse failures={len(fails)}")
        for i in fails[:10]:
            print(f"      FAIL row {i}: {df['SMILES'].iloc[i]!r}")

    hdr("4. PRIMARY TARGET: pEC50 IN TRAIN")
    y = train[PEC50]
    print(f"  non-null {y.notna().sum()}/{len(train)}   min {y.min()}   max {y.max()}   "
          f"mean {y.mean():.4f}   median {y.median()}")
    print(f"  distinct values {y.nunique()};  values equal to their own 2-dp rounding: "
          f"{(y.round(2) == y).sum()}/{len(y)}")
    print("  quantiles:")
    for q, v in y.quantile([0, .1, .25, .5, .75, .9, 1]).items():
        print(f"    {q:>5.2f}  {v}")
    print(f"  rows below 4.0: {(y < 4).sum()}   below 4.5: {(y < 4.5).sum()}   above 6.0: {(y > 6).sum()}")
    print(f"  rows at the minimum: {(y == y.min()).sum()}   rows at the maximum: {(y == y.max()).sum()}")
    print("  (no pile-up at a single value; no censoring floor apparent)")
    print(f"\n  {EMAX}: non-null {train[EMAX].notna().sum()}/{len(train)}  "
          f"min {train[EMAX].min()}  median {train[EMAX].median()}  max {train[EMAX].max()}")
    print(f"  {EMAX_POS}: min {train[EMAX_POS].min()}  median {train[EMAX_POS].median()}  "
          f"max {train[EMAX_POS].max()}")
    print("  Every row carries a fitted pEC50 and a positive Emax: the primary table contains no")
    print("  inactives or non-responders. This is an observation about the released data.")
    print(f"\n  fit quality: {SE} min {train[SE].min()} median {train[SE].median()} max {train[SE].max()}")
    print(f"  rows with pEC50 95% CI width > 2.0 log units: {((train[CI_HI] - train[CI_LO]) > 2).sum()}")

    hdr("5. DUPLICATES AND CONFLICTS WITHIN TRAIN")
    tr_canon, tr_ikey, tr_block = keys["TRAIN"]
    for label, ser in [("Molecule Name", train["Molecule Name"]), ("OCNT_ID", train["OCNT_ID"]),
                       ("raw SMILES", train["SMILES"]), ("canonical SMILES", tr_canon),
                       ("InChIKey", tr_ikey), ("InChIKey connectivity block", tr_block)]:
        print(f"  duplicated {label:<30} {ser.duplicated().sum()}")
    marked = train.assign(_block=tr_block, _ikey=tr_ikey)
    groups = marked[marked["_block"].duplicated(keep=False)].groupby("_block")
    print(f"\n  connectivity-block groups with more than one member: {groups.ngroups}")
    for block, g in groups:
        spread = g[PEC50].max() - g[PEC50].min()
        print(f"\n  connectivity {block}   members {len(g)}   distinct full InChIKeys {g['_ikey'].nunique()}"
              f"   pEC50 spread {spread:.4f}")
        print(g[["Molecule Name", "OCNT_ID", PEC50, "SMILES"]].to_string(index=False))
    print("\n  These are observed near-duplicate structures. Whether they are genuine stereochemical")
    print("  or tautomeric distinctions or representation inconsistencies is NOT established here.")
    print("  They matter because a holdout that separates members of a group would leak.")

    hdr("6. TRAIN / TEST OVERLAP")
    print("  Identity rule precedent found in notes/Decisions.md is CYP-scoped: exact Molecule_Name")
    print("  string, with canonicalization explicitly forbidden for identity matching. The structural")
    print("  comparisons below are DIAGNOSTIC and are not an established PXR identity rule.\n")
    te_canon, te_ikey, te_block = keys["TEST_BLINDED"]
    for label, a, b in [
        ("Molecule Name (project rule precedent)", set(train["Molecule Name"]), set(test["Molecule Name"])),
        ("raw SMILES string", set(train["SMILES"]), set(test["SMILES"])),
        ("canonical SMILES [diagnostic]", set(tr_canon.dropna()), set(te_canon.dropna())),
        ("InChIKey [diagnostic]", set(tr_ikey.dropna()), set(te_ikey.dropna())),
        ("InChIKey connectivity [diagnostic]", set(tr_block.dropna()), set(te_block.dropna())),
    ]:
        print(f"  TRAIN n TEST by {label:<42} {len(a & b)}")

    hdr("7. HELD-OUT LABEL COVERAGE (identity only)")
    n1, n2, blind = set(p1["Molecule Name"]), set(p2["Molecule Name"]), set(test["Molecule Name"])
    print(f"  PHASE_1 rows {len(p1)}  unique names {len(n1)}  phase column {sorted(set(p1['phase']))}")
    print(f"  PHASE_2 rows {len(p2)}  unique names {len(n2)}  phase column {sorted(set(p2['phase']))}")
    print(f"  PHASE_1 n PHASE_2                     {len(n1 & n2)}")
    print(f"  PHASE_1 u PHASE_2                     {len(n1 | n2)}")
    print(f"  TEST_BLINDED unique names             {len(blind)}")
    print(f"  union is a subset of the blinded set  {(n1 | n2) <= blind}")
    print(f"  blinded names with no released label  {len(blind - (n1 | n2))}")
    print(f"  TRAIN n released-label names          {len(set(train['Molecule Name']) & (n1 | n2))}")

    hdr("8. AUXILIARY TABLES vs THE PRIMARY TRAIN SET")
    tr_names, tr_ocnt = set(train["Molecule Name"]), set(train["OCNT_ID"])
    for label, df, col in [("counter-assay", counter, "OCNT_ID"), ("single-concentration", single, "OCNT_ID"),
                           ("htchem crudes", crude, "OCNT_ID"), ("semi-pure", semi, "OCNT_ID"),
                           ("DRC curves", drc, "OCNT_ID")]:
        s = set(df[col].dropna())
        print(f"  {label:<22} unique OCNT_ID {len(s):>6}   in TRAIN {len(s & tr_ocnt):>6}   "
              f"not in TRAIN {len(s - tr_ocnt):>6}")
    print(f"\n  counter-assay {PEC50} non-null: {counter[PEC50].notna().sum()}/{len(counter)} "
          f"(nulls are compounds with no fitted counter-assay curve)")

    print("\n  Structural match of an auxiliary table into the blinded test set [diagnostic]:")
    te_canon_map = dict(zip(te_canon, test["Molecule Name"]))
    for label, df in [("htchem crudes", crude), ("semi-pure", semi)]:
        canon, _, _ = keys[label]
        hits = [(df["OCNT_ID"].iloc[i], c, te_canon_map[c]) for i, c in enumerate(canon)
                if c is not None and c in te_canon_map]
        print(f"    {label:<16} {len(hits)} match(es)")
        for ocnt, smi, name in hits:
            print(f"      {ocnt}  ==  blinded molecule {name}")
            print(f"        canonical SMILES {smi}")
    print("    A structural match of this kind is not caught by a name-based identity rule.")

    hdr("9. DRC CURVE COVERAGE AND SINGLE-CONCENTRATION SHAPE")
    print(f"  DRC rows {len(drc)}  unique OCNT_ID {drc['OCNT_ID'].nunique()}  TRAIN molecules {len(tr_ocnt)}")
    print(f"  TRAIN molecules with no DRC curve {len(tr_ocnt - set(drc['OCNT_ID']))}")
    print(f"  DRC OCNT_IDs absent from TRAIN    {len(set(drc['OCNT_ID']) - tr_ocnt)}")
    print(f"  curves per molecule: {sorted(Counter(drc['OCNT_ID'].value_counts().values).items())}")
    print(f"\n  single-concentration rows {len(single)}  unique OCNT_ID {single['OCNT_ID'].nunique()}  "
          f"unique batches {single['OCNT Batch'].nunique()}")
    print(f"  Molecule Name null in {single['Molecule Name'].isna().sum()} of {len(single)} rows")
    print(f"  concentration_M values: {sorted(single['concentration_M'].unique())}")
    print(f"  compound_class: {single['compound_class'].value_counts().to_dict()}")
    print(f"  n_replicates:   {single['n_replicates'].value_counts().to_dict()}")
    print("  Endpoint is log2_fc_estimate, a single-concentration response, not a potency.")

    hdr("10. WHAT THIS AUDIT DOES NOT ESTABLISH")
    for line in [
        "No exclusion, eligibility rule or filter is proposed or frozen by this audit.",
        "The PXR identity rule is not established; the rule applied above is a CYP-scoped precedent.",
        "The scaffold/series holdout algorithm is not declared anywhere in the release or the repository.",
        "Whether the near-duplicate groups are genuine distinctions or representation artifacts is unknown.",
        "Eligibility of crude and semi-pure HT-chem material for a pEC50 target is undecided.",
        "The modelling consequence of the primary table containing no inactives is not assessed here.",
    ]:
        print(f"  - {line}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
