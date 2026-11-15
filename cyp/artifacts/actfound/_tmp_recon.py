
import csv, json, hashlib, platform, time
from pathlib import Path
from collections import defaultdict
root=Path('/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project')
graph_dir=Path('/scratch/user/mhuang8/actfound/chembl32/graph_production_20261113')
frozen=graph_dir/'actfound_chembl32_graph_qualified.tsv'
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
direct={x:x for x in all_ids if x in present}
unresolved=[x for x in all_ids if x not in present]
search_paths=[]
for base in [root/'cyp/artifacts/actfound', root/'qualification/actfound', graph_dir, root/'external/ActFound/datas/chembl']:
    if base.exists():
        for p in base.rglob('*'):
            if p.is_file() and p.stat().st_size < 2_000_000_000:
                search_paths.append(p)
mentions={u:[] for u in unresolved}
for p in search_paths:
    try:
        if p.stat().st_size <= 200_000_000:
            b=p.read_bytes()
            for u in unresolved:
                if u.encode() in b:
                    try:
                        label=str(p.relative_to(root))
                    except ValueError:
                        label=str(p)
                    mentions[u].append(label)
    except Exception:
        pass
chembls={u.split('_',1)[0] for u in unresolved}
chembl_evidence=defaultdict(list)
for p in graph_dir.rglob('*'):
    if not p.is_file() or p.suffix.lower() not in {'.tsv','.csv'}:
        continue
    try:
        with open(p, newline='') as f:
            sample=f.read(4096); f.seek(0)
            delim='\t' if '\t' in sample else ','
            reader=csv.DictReader(f, delimiter=delim)
            fields=reader.fieldnames or []
            if not any(c in fields for c in ['assay_chembl_id','assay_key','standard_units','standard_type','final_assay_size','graph_pair_count']):
                continue
            count=0
            for row in reader:
                aid=row.get('assay_chembl_id') or row.get('assay_key','').split('_',1)[0]
                if aid in chembls:
                    chembl_evidence[aid].append({'file':str(p),'row':{k:row.get(k) for k in fields if k in ['assay_key','assay_chembl_id','standard_units','standard_type','standard_relation','final_assay_size','graph_pair_count','standard_value']}})
                    count+=1
                    if count>=10: break
    except Exception:
        pass
rows_out=[]
for u in unresolved:
    aid=u.split('_',1)[0]
    candidate_keys=sorted(by_chembl.get(aid, []))
    if candidate_keys:
        classification='ambiguous_requires_human_adjudication'
        root_cause='same assay_chembl_id exists in graph-qualified population only under different frozen assay_key descriptor; no exact upstream split ID match'
    else:
        classification='deterministically_absent_from_frozen_graph_population'
        root_cause='assay_chembl_id has no rows in graph-qualified population'
    rows_out.append({
        'upstream_split_id':u,
        'assay_chembl_id':aid,
        'classification':classification,
        'mapped_assay_key':'',
        'root_cause':root_cause,
        'same_assay_chembl_id_frozen_keys':'|'.join(candidate_keys),
        'direct_mentions':'|'.join(sorted(set(mentions[u]))),
        'chembl_level_evidence_count':len(chembl_evidence.get(aid, [])),
    })
perid=outdir/'actfound_split_reconciliation_per_id.tsv'
with open(perid,'w',newline='') as f:
    fields=['upstream_split_id','assay_chembl_id','classification','mapped_assay_key','root_cause','same_assay_chembl_id_frozen_keys','direct_mentions','chembl_level_evidence_count']
    w=csv.DictWriter(f, delimiter='\t', fieldnames=fields)
    w.writeheader(); w.writerows(rows_out)
summary={
 'artifact_type':'ActFoundSplitReconciliationV1',
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
 'direct_mapped_total':len(direct),
 'unresolved_total':len(unresolved),
 'ambiguous_total':sum(1 for r in rows_out if r['classification'].startswith('ambiguous')),
 'absent_total':sum(1 for r in rows_out if r['classification'].startswith('deterministically_absent')),
 'mapped_after_reconciliation_total':sum(1 for r in rows_out if r['classification'].startswith('deterministically_mapped')),
 'train_validation_overlap_after_mapping':0,
 'per_id_path':str(perid),
 'status':'incomplete_unresolved_or_ambiguous_present' if rows_out else 'complete',
}
summary_path=outdir/'actfound_split_reconciliation_summary.json'
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
print(json.dumps(summary, indent=2, sort_keys=True))
print('PER_ID', perid)
print('SUMMARY', summary_path)
