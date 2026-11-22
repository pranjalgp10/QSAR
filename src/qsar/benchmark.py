"""Auditable scaffold-aware Syk QSAR benchmark.

Runs with the repository venv:
    .venv/Scripts/python.exe src/qsar/benchmark.py
All output is written below results/qsar_benchmark/. No binary potency threshold
or pseudolabel acceptance cutoff is selected by this script.
"""
from __future__ import annotations
import argparse, json, logging, platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn, rdkit, xgboost
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[2]
SEED=140
LOG=logging.getLogger("qsar_benchmark")

def canonicalize(smiles: str) -> tuple[str,str]:
    mol=Chem.MolFromSmiles(str(smiles))
    if mol is None: raise ValueError(f"Invalid SMILES: {smiles}")
    can=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
    scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False)
    # Keep each acyclic structure as its own group: empty Murcko scaffolds must
    # not put every acyclic molecule into one artificial mega-scaffold.
    return can, scaffold or f"ACYCLIC::{can}"

def audit_data(path: Path) -> tuple[pd.DataFrame,dict[str,Any],pd.DataFrame,pd.DataFrame]:
    df=pd.read_csv(path)
    required={"Smiles","IC50","pIC50"}
    if not required.issubset(df.columns): raise ValueError(f"Expected {required}; got {set(df.columns)}")
    if df[list(required)].isna().any().any(): raise ValueError("Required fields contain missing values")
    if (df.IC50<=0).any(): raise ValueError("IC50 must be positive; inspect censored/invalid values")
    pairs=[canonicalize(s) for s in df.Smiles]
    df=df.copy(); df["canonical_smiles"]=[x[0] for x in pairs];df["scaffold_group"]=[x[1] for x in pairs]
    # Verify arithmetic only. Unit provenance is not recoverable from this CSV.
    conv=9-np.log10(df.IC50.to_numpy(float)); err=np.abs(conv-df.pIC50.to_numpy(float))
    if np.max(err)>2e-5: raise ValueError("pIC50 does not agree with nM conversion; verify units/source")
    counts=df.canonical_smiles.value_counts(); dup=set(counts[counts>1].index)
    duplicates=df[df.canonical_smiles.isin(dup)].sort_values("canonical_smiles")
    dup_summary=duplicates.groupby("canonical_smiles").agg(n_records=("pIC50","size"),pIC50_min=("pIC50","min"),pIC50_max=("pIC50","max"),pIC50_median=("pIC50","median"),pIC50_range=("pIC50",lambda z:float(z.max()-z.min()))).reset_index()
    audit={"records":int(len(df)),"valid_smiles":int(len(df)),"invalid_smiles":0,"literal_smiles_unique":int(df.Smiles.nunique()),"canonical_structures":int(df.canonical_smiles.nunique()),"duplicate_structure_groups":int(len(dup)),"excess_duplicate_rows":int(len(df)-df.canonical_smiles.nunique()),"disconnected_smiles":int(df.Smiles.astype(str).str.contains(r"\.").sum()),"pIC50_conversion_max_abs_error":float(err.max()),"conversion_assumption":"IC50 in nM; internal consistency verified, source-unit provenance not present in CSV","metadata_available":list(df.columns)}
    return df,audit,duplicates,dup_summary

def make_fingerprints(smiles: list[str], radius=2,bits=2048):
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=bits)
    X=np.zeros((len(smiles),bits),dtype=np.uint8)
    mols=[]
    for i,s in enumerate(smiles):
        mol=Chem.MolFromSmiles(s);mols.append(mol)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol),X[i])
    return X,mols

def split_scaffolds(groups: np.ndarray, y: np.ndarray,seed=SEED):
    ix=np.arange(len(y))
    dev,test=next(GroupShuffleSplit(n_splits=1,test_size=.20,random_state=seed).split(ix,y,groups))
    trrel,varel=next(GroupShuffleSplit(n_splits=1,test_size=.20,random_state=seed+1).split(dev,y[dev],groups[dev]))
    train,val=dev[trrel],dev[varel]
    sets=[set(groups[z]) for z in (train,val,test)]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)): raise AssertionError("Scaffold leakage")
    return train,val,test

def metrics(y,p):
    return {"n":int(len(y)),"MAE":float(mean_absolute_error(y,p)),"MSE":float(mean_squared_error(y,p)),"RMSE":float(mean_squared_error(y,p)**.5),"R2":float(r2_score(y,p))}

def model_spaces(seed=SEED,quick=False):
    return {
      "RandomForest":(RandomForestRegressor(random_state=seed,n_jobs=-1),{"n_estimators":[300,600],"max_features":["sqrt",.5],"min_samples_leaf":[1,2,4],"max_depth":[None,28]}),
      "ExtraTrees":(ExtraTreesRegressor(random_state=seed,n_jobs=-1),{"n_estimators":[300,600],"max_features":["sqrt",.5],"min_samples_leaf":[1,2,4],"max_depth":[None,28]}),
      "SVR":(make_pipeline(StandardScaler(),SVR()),{"svr__C":[1,5,20,50],"svr__epsilon":[.1,.2,.4],"svr__gamma":["scale",.001,.01]}),
      "XGBoost":(XGBRegressor(objective="reg:squarederror",random_state=seed,n_jobs=1,verbosity=0),{"n_estimators":[250,500],"max_depth":[4,7],"learning_rate":[.03,.07],"subsample":[.8,1.]}),
      "HistGradientBoosting":(HistGradientBoostingRegressor(random_state=seed),{"max_iter":[150,300],"max_leaf_nodes":[15,31],"learning_rate":[.04,.08],"l2_regularization":[0.,2.]})}

def tune_models(X,y,groups,seed=SEED,quick=False):
    nfold=min(4,len(np.unique(groups)))
    cv=list(GroupKFold(n_splits=nfold).split(X,y,groups))
    fitted={}; params={}; search_rows=[]
    for name,(est,space) in model_spaces(seed).items():
        search=RandomizedSearchCV(est,space,n_iter=4,scoring="neg_root_mean_squared_error",cv=cv,refit=True,random_state=seed,n_jobs=1,return_train_score=True)
        search.fit(X,y); fitted[name]=search.best_estimator_;params[name]=search.best_params_
        search_rows.append({"model":name,"best_cv_rmse":float(-search.best_score_),"best_params":search.best_params_})
        LOG.info("%s best grouped-CV RMSE %.4f",name,-search.best_score_)
    return fitted,params,search_rows,cv

def nested_group_cv(X,y,groups,seed=SEED,n_outer=5):
    """Nested scaffold CV: inner model tuning never sees outer validation scaffolds."""
    oof={name:np.full(len(y),np.nan) for name in [*model_spaces(seed).keys(),"Stacking"]}
    outer=list(GroupKFold(n_splits=min(n_outer,len(np.unique(groups)))).split(X,y,groups))
    for fold,(itr,ite) in enumerate(outer):
        LOG.info("Nested scaffold CV outer fold %d/%d",fold+1,len(outer))
        base,_,_,_=tune_models(X[itr],y[itr],groups[itr],seed+fold)
        fitted={**base,"Stacking":fit_stack(base,X[itr],y[itr],groups[itr])}
        for name,model in fitted.items():
            model.fit(X[itr],y[itr]);oof[name][ite]=model.predict(X[ite])
    return {name:pred for name,pred in oof.items()},outer

def fit_stack(base_models,X,y,groups):
    cv=list(GroupKFold(n_splits=min(4,len(np.unique(groups)))).split(X,y,groups))
    return StackingRegressor(estimators=[(k,clone(v)) for k,v in base_models.items()],final_estimator=Ridge(),cv=cv,n_jobs=1).fit(X,y)

def scaffold_bootstrap_ci(y,p,groups,n_boot=1000,seed=SEED):
    rng=np.random.default_rng(seed); uniq=np.unique(groups); vals=[]
    for _ in range(n_boot):
        drawn=rng.choice(uniq,size=len(uniq),replace=True); ix=np.concatenate([np.flatnonzero(groups==g) for g in drawn])
        vals.append([mean_absolute_error(y[ix],p[ix]),mean_squared_error(y[ix],p[ix])**.5,r2_score(y[ix],p[ix])])
    arr=np.asarray(vals)
    return {k:[float(v) for v in np.quantile(arr[:,i],[.025,.975])] for i,k in enumerate(["MAE","RMSE","R2"])}

def agreement_ad(base_models,X_train,X_eval,train_mols,eval_mols):
    pred=np.column_stack([m.predict(X_eval) for m in base_models.values()])
    spread=pred.std(axis=1,ddof=1)
    maxsim=[]
    trainfps=[Chem.RDKFingerprint(m) for m in train_mols]
    for mol in eval_mols:
        fp=Chem.RDKFingerprint(mol)
        maxsim.append(max(DataStructs.BulkTanimotoSimilarity(fp,trainfps)) if trainfps else float("nan"))
    return pred,spread,np.asarray(maxsim)

def reliability_table(score,error,n=5):
    """Empirical error/coverage by score quantile; descriptive, not a cutoff selector."""
    bins=pd.qcut(pd.Series(score),q=min(n,pd.Series(score).nunique()),duplicates="drop")
    return pd.DataFrame({"score":score,"abs_error":error,"bin":bins.astype(str)}).groupby("bin",observed=True).agg(
        count=("abs_error","size"),mean_score=("score","mean"),MAE=("abs_error","mean"),
        p90_absolute_error=("abs_error",lambda z:float(np.quantile(z,.9,method="higher")))).reset_index()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=Path,default=ROOT/"processed_df.csv");ap.add_argument("--output",type=Path,default=ROOT/"results"/"qsar_benchmark");ap.add_argument("--seed",type=int,default=SEED);ap.add_argument("--curation",type=Path,help="Reviewed duplicate resolution CSV: canonical_smiles,resolved_pIC50,rationale");args=ap.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    df,audit,dups,dup_summary=audit_data(args.data);audit["activity_summary"]=df.pIC50.describe(percentiles=[.1,.25,.5,.75,.9]).to_dict()
    dups.to_csv(out/"duplicate_structure_review.csv",index=False)
    dup_summary.to_csv(out/"duplicate_structure_summary.csv",index=False)
    (out/"data_audit.json").write_text(json.dumps(audit,indent=2,default=float),encoding="utf8")
    # Do not silently overweight structurally duplicate rows or invent replicate aggregation.
    if len(dup_summary):
        if args.curation is None:
            raise RuntimeError(f"Found {len(dup_summary)} repeated canonical structures. Review duplicate_structure_review.csv and supply an explicit, justified resolution table with --curation before model training.")
        resolutions=pd.read_csv(args.curation)
        needed={"canonical_smiles","resolved_pIC50","rationale"}
        if not needed.issubset(resolutions.columns): raise ValueError(f"Curation file must include {needed}")
        if resolutions.canonical_smiles.duplicated().any() or resolutions.rationale.isna().any() or (resolutions.rationale.astype(str).str.strip()=="").any(): raise ValueError("Curation must have one nonempty rationale per canonical structure")
        expected=set(dup_summary.canonical_smiles); supplied=set(resolutions.canonical_smiles)
        if expected!=supplied: raise ValueError("Curation file must resolve every duplicate group and no other structures")
        df=df.drop_duplicates("canonical_smiles",keep="first").copy()
        mapping=dict(zip(resolutions.canonical_smiles,resolutions.resolved_pIC50))
        mask=df.canonical_smiles.isin(mapping);df.loc[mask,"pIC50"]=[mapping[s] for s in df.loc[mask,"canonical_smiles"]]
    X,mols=make_fingerprints(df.canonical_smiles.tolist());y=df.pIC50.to_numpy(float);groups=df.scaffold_group.to_numpy()
    train,val,test=split_scaffolds(groups,y,args.seed)
    split=np.full(len(y),"test",dtype=object);split[train]="train";split[val]="validation"
    manifest=df[["Smiles","canonical_smiles","scaffold_group"]].copy();manifest["partition"]=split;manifest["pIC50"]=y;manifest.to_csv(out/"split_manifest.csv",index=False)
    # Tuning is restricted to training. Validation is split for model selection vs uncertainty calibration.
    vsel,vcal=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=args.seed+2).split(val,y[val],groups[val]))
    val_select,val_cal=val[vsel],val[vcal]
    nested_oof,nested_folds=nested_group_cv(X[train],y[train],groups[train],args.seed)
    nested_metrics={name:metrics(y[train],pred) for name,pred in nested_oof.items()}
    pd.DataFrame({"row_index":train,"Smiles":df.Smiles.iloc[train].values,"scaffold":groups[train],**{f"oof_{k}":v for k,v in nested_oof.items()}}).to_csv(out/"nested_scaffold_oof_predictions.csv",index=False)
    best,params,searchlog,_=tune_models(X[train],y[train],groups[train],args.seed)
    models={**best,"Stacking":fit_stack(best,X[train],y[train],groups[train])}
    predictions={name:{"select":m.predict(X[val_select]),"cal":m.predict(X[val_cal]),"test":m.predict(X[test])} for name,m in models.items()}
    # Selection uses only validation-selection subset; test metrics are reported only after selection freezes.
    selection={n:metrics(y[val_select],v["select"]) for n,v in predictions.items()};winner=min(selection,key=lambda n:selection[n]["RMSE"])
    base_names=list(best);bcal,spread_cal,sim_cal=agreement_ad(best,X[train],X[val_cal],[mols[i] for i in train],[mols[i] for i in val_cal])
    btest,spread_test,sim_test=agreement_ad(best,X[train],X[test],[mols[i] for i in train],[mols[i] for i in test])
    # Distribution-free absolute residual quantile calibrated only on separate validation-calibration data.
    calerr=np.abs(y[val_cal]-predictions[winner]["cal"]);alpha=.10
    q=min(1.,np.ceil((len(calerr)+1)*(1-alpha))/len(calerr));interval=float(np.quantile(calerr,q,method="higher"))
    testpred=predictions[winner]["test"]
    # Separate descriptive reliability tables support later scientific cutoff decisions.
    reliability_table(spread_cal,calerr).to_csv(out/"agreement_reliability_validation.csv",index=False)
    reliability_table(1-sim_cal,calerr).to_csv(out/"applicability_domain_reliability_validation.csv",index=False)
    report={"nested_scaffold_cv_metrics_train_partition":nested_metrics,"winner_by_validation_rmse":winner,"selection_metrics":selection,"test_metrics_all_models":{n:metrics(y[test],p["test"]) for n,p in predictions.items()},"selected_test_scaffold_bootstrap_95ci":scaffold_bootstrap_ci(y[test],testpred,groups[test]),"test_interval_nominal_90":{"half_width_pIC50":interval,"empirical_coverage":float(np.mean(np.abs(y[test]-testpred)<=interval))},"split_sizes":{"train":len(train),"validation":len(val),"test":len(test),"validation_selection":len(val_select),"validation_calibration":len(val_cal)},"split_scaffolds":{"train":len(set(groups[train])),"validation":len(set(groups[val])),"test":len(set(groups[test]))},"activity_distribution":{"train":df.pIC50.iloc[train].describe().to_dict(),"validation":df.pIC50.iloc[val].describe().to_dict(),"test":df.pIC50.iloc[test].describe().to_dict()},"test_status":"internal scaffold holdout only; prior experimental evaluation on this CSV means it is not virgin external validation","confidence":"four-regressor prediction SD and nearest training Morgan RDK fingerprint Tanimoto are separate diagnostics; agreement is not itself calibrated probability","confidence_calibration":{"validation_calibration_rows":len(val_cal),"prediction_sd_error_spearman":float(pd.Series(spread_cal).corr(pd.Series(calerr),method="spearman")),"similarity_error_spearman":float(pd.Series(sim_cal).corr(pd.Series(calerr),method="spearman"))},"hyperparameters":params,"tuning":searchlog,"versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__,"rdkit":rdkit.__version__,"xgboost":xgboost.__version__},"data_audit":audit}
    # Persist model-by-model predictions for audit; no acceptance cutoff is silently applied.
    predframe=pd.DataFrame({"Smiles":df.Smiles.iloc[test].values,"canonical_smiles":df.canonical_smiles.iloc[test].values,"observed_pIC50":y[test],"selected_model":winner,"selected_prediction":testpred,"selected_lower_90":testpred-interval,"selected_upper_90":testpred+interval,"ensemble_sd":spread_test,"max_train_similarity":sim_test})
    for name in models: predframe[f"prediction_{name}"]=predictions[name]["test"]
    predframe.to_csv(out/"test_predictions.csv",index=False)
    pd.DataFrame({"model":list(selection),"validation_selection":list(selection.values()),"test":list(report["test_metrics_all_models"].values())}).to_json(out/"model_comparison.json",orient="records",indent=2)
    (out/"report.json").write_text(json.dumps(report,indent=2,default=float),encoding="utf8")
    # Deployment refit on train+validation only, after test reporting.
    dev=np.concatenate([train,val]); selected=fit_stack(best,X[dev],y[dev],groups[dev]) if winner=="Stacking" else clone(best[winner]).fit(X[dev],y[dev])
    deployment_bases={k:clone(m).fit(X[dev],y[dev]) for k,m in best.items()}
    joblib.dump({"model":selected,"winner":winner,"fingerprint":{"family":"Morgan","radius":2,"bits":2048},"version":{"sklearn":sklearn.__version__,"rdkit":rdkit.__version__},"agreement_base_models":deployment_bases},out/"deployment_model.joblib")
    LOG.info("Completed. Winner=%s output=%s",winner,out)

if __name__=="__main__": main()
