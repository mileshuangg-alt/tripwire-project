
import csv, json, hashlib, platform, time
from pathlib import Path
from collections import defaultdict
root=Path('/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project')
frozen=Path('/scratch/user/mhuang8/actfound/chembl32/graph_production_20261113/actfound_chembl32_graph_qualified.tsv')
split_path=root/'external/ActFound/datas/chembl/chembl_split.json'
outdir=root/'cyp/artifacts/actfound'
outdir.mkdir(parents=True, exist_ok=True)
def norm_key(s): return s.replace('/', '_')
def h(path):
    x=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            x.update(b)
    return x.hexdigest()
split=json.load(open(split_path))
train=[norm_key(x) for x in split['train']]
val=[norm_key(x) for x in split['valid']]
if not train or not val: raise SystemExit('empty split')
if set(train)&set(val): raise SystemExit('overlap in upstream split')
present=set(); by_chembl=defaultdict(set); rows=0
with open(frozen, newline='') as f:
    r=csv.DictReader(f, delimiter='\t')
    for row in r:
        rows+=1
        key=row['assay_key']
        present.add(key)
        by_chembl[row['assay_chembl_id']].add(key)
all_ids=train+val
mapped=[x for x in all_ids if x in present]
unresolved=[x for x in all_ids if x not in present]
rows_out=[]
for u in unresolved:
    aid=u.split('_',1)[0]
    candidate_keys=sorted(by_chembl.get(aid, []))
    if candidate_keys:
        classification='ambiguous_requires_human_adjudication'
        root_cause='upstream split ID is absent as an exact frozen assay_key; same assay_chembl_id is present only under different descriptor(s), so descriptor crosswalk is not authoritative enough to map without adjudication'
    else:
        classification='deterministically_absent_from_frozen_graph_population'
        root_cause='upstream split ID is absent as an exact frozen assay_key and assay_chembl_id has no rows in graph-qualified population'
    partition='train' if u in set(train) else 'validation'
    rows_out.append({
        'upstream_split_id':u,
        'partition':partition,
        'assay_chembl_id':aid,
        'classification':classification,
        'mapped_assay_key':'',
        'root_cause':root_cause,
        'same_assay_chembl_id_frozen_keys':'|'.join(candidate_keys),
        'same_assay_chembl_id_frozen_key_count':len(candidate_keys),
    })
perid=outdir/'actfound_split_reconciliation_per_id.tsv'
fields=['upstream_split_id','partition','assay_chembl_id','classification','mapped_assay_key','root_cause','same_assay_chembl_id_frozen_keys','same_assay_chembl_id_frozen_key_count']
with open(perid,'w',newline='') as f:
    w=csv.DictWriter(f, delimiter='\t', fieldnames=fields)
    w.writeheader(); w.writerows(rows_out)
summary={
 'artifact_type':'ActFoundSplitReconciliationV1',
 'method':'exact upstream split ID to frozen assay_key; no heuristic descriptor remapping; assay_chembl_id used only to distinguish absent vs different-descriptor unresolved cases',
 'frozen_dataset_path':str(frozen),
 'frozen_dataset_sha256':h(frozen),
 'upstream_split_path':str(split_path),
 'upstream_split_sha256':h(split_path),
 'created_at_unix':time.time(),
 'created_on_host':platform.node(),
 'frozen_rows':rows,
 'frozen_assays':len(present),
 'train_split_ids':len(train),
 'validation_split_ids':len(val),
 'mapped_total':len(mapped),
 'train_mapped':sum(1 for x in train if x in present),
 'validation_mapped':sum(1 for x in val if x in present),
 'unresolved_total':len(unresolved),
 'ambiguous_total':sum(1 for r in rows_out if r['classification'].startswith('ambiguous')),
 'absent_total':sum(1 for r in rows_out if r['classification'].startswith('deterministically_absent')),
 'mapped_after_reconciliation_total':0,
 'train_validation_overlap_after_mapping':0,
 'per_id_path':str(perid),
 'status':'incomplete_unresolved_or_ambiguous_present' if rows_out else 'complete',
}
summary_path=outdir/'actfound_split_reconciliation_summary.json'
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
print(json.dumps(summary, indent=2, sort_keys=True))
print('PER_ID', perid)
print('SUMMARY', summary_path)
