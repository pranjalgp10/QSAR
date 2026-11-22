"""Resolve canonical duplicate structures using the cited Syk-paper rule.

Input is preserved. Outputs are written to data/processed/ with source-row
mapping and per-structure rationale. The paper rule averages IC50 when the
full range is within 10% of the median; otherwise it retains minimum IC50.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT=Path(__file__).resolve().parents[2]

def canonical(smiles):
    mol=Chem.MolFromSmiles(str(smiles))
    if mol is None: raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True), mol

def curate(source=ROOT/'processed_df.csv', output_dir=ROOT/'data'/'processed'):
    df=pd.read_csv(source)
    required={'Smiles','IC50','pIC50'}
    if not required.issubset(df): raise ValueError(f"Expected {required}")
    if df.isna().any().any() or (df.IC50<=0).any(): raise ValueError('Missing fields or nonpositive IC50')
    conv=9-np.log10(df.IC50.to_numpy(float))
    if not np.allclose(conv,df.pIC50.to_numpy(float),atol=2e-5): raise ValueError('IC50 units inconsistent with nM conversion')
    cans=[];mols=[]
    for s in df.Smiles: c,m=canonical(s);cans.append(c);mols.append(m)
    df=df.copy();df['source_row']=np.arange(len(df));df['canonical_smiles']=cans
    records=[];duplicate_rows=[]
    for can,g in df.groupby('canonical_smiles',sort=True):
        vals=g.IC50.to_numpy(float); med=float(np.median(vals)); lo=float(vals.min());hi=float(vals.max())
        if len(g)==1:
            resolved=float(vals[0]);rule='single canonical structure record'
        elif np.allclose(vals,vals[0],rtol=0,atol=1e-12):
            resolved=float(vals[0]);rule='exact repeated IC50; retained one value'
        elif hi-lo <= .10*med:
            resolved=float(np.mean(vals));rule='paper rule: mean IC50 because full range <= 10% of median'
        else:
            resolved=lo;rule='paper rule: minimum IC50 because full range > 10% of median; potency-favoring choice'
        pic=float(9-np.log10(resolved))
        first=g.iloc[0]
        mol=Chem.MolFromSmiles(can)
        scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False) or f'ACYCLIC::{can}'
        records.append({'Smiles':can,'canonical_smiles':can,'IC50':resolved,'pIC50':pic,'scaffold_group':scaffold,'source_rows':';'.join(map(str,g.source_row.tolist())),'n_source_records':len(g)})
        if len(g)>1:
            duplicate_rows.append({'canonical_smiles':can,'n_records':len(g),'source_rows':';'.join(map(str,g.source_row.tolist())),'IC50_min':lo,'IC50_median':med,'IC50_max':hi,'IC50_range':hi-lo,'resolved_IC50':resolved,'resolved_pIC50':pic,'rule':rule})
    curated=pd.DataFrame(records);dups=pd.DataFrame(duplicate_rows)
    output_dir.mkdir(parents=True,exist_ok=True)
    curated.to_csv(output_dir/'syk_qsar_curated.csv',index=False)
    dups.to_csv(output_dir/'syk_duplicate_resolution.csv',index=False)
    report={'source':str(source),'input_records':len(df),'valid_structures':len(df),'unique_literal_smiles':int(df.Smiles.nunique()),'unique_canonical_structures':len(curated),'duplicate_groups':len(dups),'extra_records_removed':len(df)-len(curated),'duplicate_rule':'within 10% of median IC50: average IC50; otherwise retain minimum IC50, adapted from the Syk paper; exact repeated values collapse to one','activity_pIC50_conversion':'9-log10(IC50[nM]); internal arithmetic verified, source units remain unverified','warning':'The minimum-IC50 branch favors potency and can bias labels upward. Source assay metadata are unavailable. Input appears to be post-IQR-filtered; raw records are required to revisit that filtering.','pIC50_summary':curated.pIC50.describe(percentiles=[.1,.25,.5,.75,.9]).to_dict()}
    (output_dir/'syk_curation_report.json').write_text(json.dumps(report,indent=2,default=float),encoding='utf8')
    print(json.dumps(report,indent=2,default=float))
    return curated

if __name__=='__main__': curate()
