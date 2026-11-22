"""Experiment 3: stacking combinations and meta-models with nested OOF bases."""
from __future__ import annotations
import argparse,hashlib,json,logging,platform,time
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np,pandas as pd,sklearn,rdkit,xgboost
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression,Ridge,ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.qsar.experiment_01 import build_features,candidates,tune
LOG=logging.getLogger('experiment03')

def metric_set(y,p):
    e=np.asarray(p)-np.asarray(y)
    return {'n':int(len(y)),'MAE':float(mean_absolute_error(y,p)),'MSE':float(mean_squared_error(y,p)),'RMSE':float(mean_squared_error(y,p)**.5),'R2':float(r2_score(y,p)),'median_absolute_error':float(np.median(np.abs(e))),'residual_mean':float(e.mean()),'residual_sd':float(e.std(ddof=1))}

def paired_scaffold_bootstrap(pred_df, candidate, comparator, n_boot=2000, seed=1701):
    """Paired bootstrap of pooled out-of-fold errors, resampling whole scaffolds."""
    wide=pred_df.pivot(index=['row_index','scaffold','observed_pIC50'],columns='candidate',values='predicted_pIC50')
    if candidate not in wide or comparator not in wide: raise ValueError('Missing candidate predictions')
    scaffolds,group_ix=np.unique(wide.index.get_level_values('scaffold'),return_inverse=True)
    y=wide.index.get_level_values('observed_pIC50').to_numpy(float)
    e1=wide[candidate].to_numpy(float)-y;e0=wide[comparator].to_numpy(float)-y
    rng=np.random.default_rng(seed);diff=[]
    for _ in range(n_boot):
        draw=rng.integers(0,len(scaffolds),size=len(scaffolds));counts=np.bincount(draw,minlength=len(scaffolds));w=counts[group_ix];den=w.sum()
        diff.append([np.sqrt(np.sum(w*e1**2)/den)-np.sqrt(np.sum(w*e0**2)/den),np.sum(w*np.abs(e1))/den-np.sum(w*np.abs(e0))/den])
    a=np.asarray(diff)
    return {'candidate':candidate,'comparator':comparator,'scaffolds':int(len(scaffolds)),'bootstrap_replicates':n_boot,'RMSE_difference':float(np.sqrt(np.mean(e1**2))-np.sqrt(np.mean(e0**2))),'RMSE_difference_95CI':np.quantile(a[:,0],[.025,.975]).tolist(),'MAE_difference':float(np.mean(np.abs(e1))-np.mean(np.abs(e0))),'MAE_difference_95CI':np.quantile(a[:,1],[.025,.975]).tolist(),'cluster_unit':'Bemis-Murcko scaffold'}

def candidates_meta():
    out={'OLS':LinearRegression()}
    for a in [.1,1.,10.,100.]:out[f'Ridge_a{a:g}']=Ridge(alpha=a)
    for l1 in [.1,.5,.9]:out[f'ElasticNet_l1{l1:g}']=ElasticNet(alpha=.01,l1_ratio=l1,max_iter=20000,tol=1e-4,random_state=1701)
    return out

def combos():
    return {
      'all5':['RandomForest','ExtraTrees','SVR','XGBoost','HistGradientBoosting'],
      'boosting_pair':['XGBoost','HistGradientBoosting'],
      'rf_boosting':['RandomForest','XGBoost','HistGradientBoosting'],
      'rf_et_xgb':['RandomForest','ExtraTrees','XGBoost'],
      'complementarity_svR':['RandomForest','XGBoost','HistGradientBoosting','SVR']}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',type=Path);args=parser.parse_args()
    cfgpath=ROOT/'experiments'/'qsar'/'experiment_03_stacking_optimisation'/'config.json';cfg=json.loads(cfgpath.read_text(encoding='utf8'))
    runid=args.resume.name if args.resume else datetime.now(timezone.utc).strftime('exp03_%Y%m%dT%H%M%SZ')
    out=args.resume if args.resume else ROOT/'results'/'qsar'/'experiment_03_stacking_optimisation'/runid
    if args.resume and not out.is_absolute():out=ROOT/out
    for p in ['configs','predictions','metrics','manifests','logs','reports']: (out/p).mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'logs'/'run.log',encoding='utf8'),logging.StreamHandler()],format='%(asctime)s %(levelname)s %(message)s')
    cp=out/'configs'/'config.json'
    if args.resume and cp.exists() and json.loads(cp.read_text(encoding='utf8'))!=cfg:raise ValueError('Resume config mismatch')
    cp.write_text(json.dumps(cfg,indent=2),encoding='utf8')
    df=pd.read_csv(ROOT/'data'/'processed'/'syk_qsar_curated.csv');y=df.pIC50.to_numpy(float);g=df.scaffold_group.astype(str).to_numpy();X,_=build_features(df,cfg['fingerprint']['bits'],cfg['fingerprint']['radius'])
    if not np.isfinite(y).all() or len(np.unique(g)) < cfg['outer_folds']:
        raise ValueError('Dataset has non-finite pIC50 values or too few unique scaffolds')
    folds=list(GroupKFold(n_splits=cfg['outer_folds'],shuffle=True,random_state=cfg['seed']).split(X,y,g))
    # Fail closed if any molecular scaffold crosses an outer boundary.
    for train_ix,test_ix in folds:
        if set(g[train_ix]) & set(g[test_ix]): raise AssertionError('Scaffold leakage across outer fold')
    metrics_path=out/'metrics'/'fold_metrics_checkpoint.csv';pred_path=out/'predictions'/'outer_predictions_checkpoint.csv';oof_path=out/'predictions'/'meta_oof_checkpoint.csv';manifest_path=out/'manifests'/'folds_checkpoint.json';coef_path=out/'metrics'/'meta_coefficients_checkpoint.csv'
    rows=pd.read_csv(metrics_path).to_dict('records') if args.resume and metrics_path.exists() else []
    preds=pd.read_csv(pred_path).to_dict('records') if args.resume and pred_path.exists() else []
    oof_rows=pd.read_csv(oof_path).to_dict('records') if args.resume and oof_path.exists() else []
    coef_rows=pd.read_csv(coef_path).to_dict('records') if args.resume and coef_path.exists() else []
    fold_records=json.loads(manifest_path.read_text(encoding='utf8')) if args.resume and manifest_path.exists() else []
    done={int(x['fold']) for x in fold_records}; base_names=list(candidates(cfg['seed']))
    for fi,(otr,ote) in enumerate(folds):
        fno=fi+1
        if fno in done:LOG.info('Skipping checkpoint outer fold %d',fno);continue
        LOG.info('Outer fold %d/%d: train %d, test %d',fno,len(folds),len(otr),len(ote));tg=g[otr]
        metacv=list(GroupKFold(n_splits=cfg['meta_oof_folds']).split(X[otr],y[otr],tg))
        oof=np.full((len(otr),len(base_names)),np.nan);base_test=np.zeros((len(ote),len(base_names)))
        # Hyperparameters for each OOF prediction are selected without its scaffold fold.
        for mi,(im,iv) in enumerate(metacv):
            for bi,bname in enumerate(base_names):
                fitted,params,inner,secs=tune(bname,X[otr][im],y[otr][im],tg[im],cfg['seed']+1000*fno+100*mi+bi,space_repeats=cfg['base_search_trials'])
                oof[iv,bi]=fitted.predict(X[otr][iv])
                for loc in iv:
                    ix=int(otr[loc]);oof_rows.append({'outer_fold':fno,'meta_fold':mi+1,'row_index':ix,'canonical_smiles':df.canonical_smiles.iloc[ix],'scaffold':g[ix],'observed_pIC50':float(y[ix]),'base_model':bname,'oof_prediction':float(oof[loc,bi]),'inner_cv_rmse':inner,'selected_params':json.dumps(params)})
        if not np.isfinite(oof).all():raise AssertionError('Incomplete OOF base prediction matrix')
        # Refit independently tuned base learners on the complete outer-train fold.
        base_params={};
        for bi,bname in enumerate(base_names):
            fitted,params,inner,secs=tune(bname,X[otr],y[otr],tg,cfg['seed']+2000*fno+bi,space_repeats=cfg['base_search_trials'])
            base_test[:,bi]=fitted.predict(X[ote]);base_params[bname]=params
        candidate_predictions={};candidate_models=[]
        for combo_name,names in combos().items():
            cols=[base_names.index(n) for n in names]
            for meta_name,meta_proto in candidates_meta().items():
                meta=meta_proto.fit(oof[:,cols],y[otr]);pp=meta.predict(base_test[:,cols]);key=f'{combo_name}__{meta_name}'
                candidate_predictions[key]=pp;candidate_models.append((key,meta,cols))
                co=np.asarray(meta.coef_).reshape(-1)
                coef_rows.append({'outer_fold':fno,'candidate':key,'intercept':float(np.asarray(meta.intercept_).reshape(-1)[0]),'coefficients':json.dumps(dict(zip(names,co.tolist())))})
        # Metrics and molecule-level paired predictions for every candidate.
        for key,pp in candidate_predictions.items():
            rows.append({'outer_fold':fno,'candidate':key,**metric_set(y[ote],pp)})
            for ix,pred in zip(ote,pp):preds.append({'outer_fold':fno,'row_index':int(ix),'canonical_smiles':df.canonical_smiles.iloc[ix],'scaffold':g[ix],'observed_pIC50':float(y[ix]),'candidate':key,'predicted_pIC50':float(pred)})
        fold_records.append({'fold':fno,'seed':cfg['seed'],'train_indices':otr.tolist(),'test_indices':ote.tolist(),'train_scaffolds':int(len(set(g[otr]))),'test_scaffolds':int(len(set(g[ote]))),'scaffold_overlap':0,'base_params':base_params})
        pd.DataFrame(rows).to_csv(metrics_path,index=False);pd.DataFrame(preds).to_csv(pred_path,index=False);pd.DataFrame(oof_rows).to_csv(oof_path,index=False);pd.DataFrame(coef_rows).to_csv(coef_path,index=False);manifest_path.write_text(json.dumps(fold_records),encoding='utf8')
    mdf=pd.DataFrame(rows);pdf=pd.DataFrame(preds);odf=pd.DataFrame(oof_rows);cdf=pd.DataFrame(coef_rows)
    mdf.to_csv(out/'metrics'/'fold_metrics.csv',index=False);pdf.to_csv(out/'predictions'/'outer_predictions.csv',index=False);odf.to_csv(out/'predictions'/'meta_oof_predictions.csv',index=False);cdf.to_csv(out/'metrics'/'meta_coefficients.csv',index=False);(out/'manifests'/'folds.json').write_text(json.dumps(fold_records,indent=2),encoding='utf8')
    expected=cfg['outer_folds']*len(combos())*len(candidates_meta())
    if len(mdf) != expected: raise AssertionError(f'Expected {expected} candidate-fold scores, found {len(mdf)}')
    summary=mdf.groupby('candidate').agg(folds=('outer_fold','count'),RMSE_mean=('RMSE','mean'),RMSE_sd=('RMSE','std'),RMSE_median=('RMSE','median'),MAE_mean=('MAE','mean'),MAE_sd=('MAE','std'),MSE_mean=('MSE','mean'),R2_mean=('R2','mean')).sort_values('RMSE_mean').reset_index();summary.to_csv(out/'metrics'/'summary.csv',index=False)
    pooled=[]
    for key,part in pdf.groupby('candidate'):
        pooled.append({'candidate':key,**metric_set(part.observed_pIC50.to_numpy(float),part.predicted_pIC50.to_numpy(float))})
    pooled_df=pd.DataFrame(pooled).sort_values('RMSE');pooled_df.to_csv(out/'metrics'/'pooled_oof_metrics.csv',index=False)
    screened_best=str(summary.iloc[0].candidate)
    comparisons=[]
    for ref in ['boosting_pair__OLS','all5__Ridge_a100','rf_boosting__OLS']:
        if ref != screened_best: comparisons.append(paired_scaffold_bootstrap(pdf,screened_best,ref,seed=cfg['seed']))
    pd.DataFrame(comparisons).to_csv(out/'metrics'/'screened_best_paired_scaffold_bootstrap.csv',index=False)
    # Residual diversity uses leakage-free inner OOF predictions only.
    wide=odf.pivot_table(index=['outer_fold','row_index'],columns='base_model',values='oof_prediction');truth=odf.drop_duplicates(['outer_fold','row_index']).set_index(['outer_fold','row_index']).observed_pIC50
    resid=wide.sub(truth,axis=0);resid.corr().to_csv(out/'metrics'/'oof_residual_correlation.csv')
    coeff=[]
    for model_name,part in cdf.groupby('candidate'):
        vals=[json.loads(v) for v in part.coefficients]
        names=sorted({n for row in vals for n in row})
        for name in names:
            arr=np.asarray([row.get(name,np.nan) for row in vals],float);arr=arr[np.isfinite(arr)]
            coeff.append({'candidate':model_name,'base_model':name,'folds':len(arr),'coefficient_mean':float(arr.mean()),'coefficient_sd':float(arr.std(ddof=1)) if len(arr)>1 else 0.0,'positive_fraction':float(np.mean(arr>0)),'negative_fraction':float(np.mean(arr<0))})
    pd.DataFrame(coeff).to_csv(out/'metrics'/'coefficient_stability.csv',index=False)
    data_hash=hashlib.sha256((ROOT/'data'/'processed'/'syk_qsar_curated.csv').read_bytes()).hexdigest()
    report={'experiment':'03 stacking architecture and meta-model optimization','run_id':runid,'dataset_rows':len(df),'dataset_sha256':data_hash,'outer_folds':cfg['outer_folds'],'candidate_with_lowest_mean_outer_rmse_screening_only':screened_best,'summary':summary.to_dict(orient='records'),'pooled_oof_metrics':pooled_df.to_dict(orient='records'),'screened_best_paired_scaffold_bootstrap':comparisons,'base_error_correlation_from_nested_oof':resid.corr().to_dict(),'limitations':'One five-fold scaffold allocation and two hyperparameter search trials per fit were used to bound compute. Candidate rankings are exploratory model-selection results; no independent external test exists. Because the same outer folds screen multiple candidates, the lowest-ranked candidate metric is selection-optimistic and is not an unbiased final estimate. Pooled OOF metrics combine outer-fold held-out predictions once per structure. Scaffold-cluster bootstrap intervals compare the screened winner with a few predeclared alternatives but do not correct winner-selection or multiple-comparison optimism. Strictly nested OOF base tuning excludes each meta fold; meta-model settings were fixed before outer-fold evaluation.','software':{'python':platform.python_version(),'sklearn':sklearn.__version__,'rdkit':rdkit.__version__,'xgboost':xgboost.__version__}}
    (out/'reports'/'report.json').write_text(json.dumps(report,indent=2,default=float),encoding='utf8')
    table=['| Candidate | Folds | Mean RMSE | SD RMSE | Mean MAE | Mean R² |','|---|---:|---:|---:|---:|---:|']
    for _,r in summary.iterrows(): table.append(f"| {r.candidate} | {int(r.folds)} | {r.RMSE_mean:.4f} | {r.RMSE_sd:.4f} | {r.MAE_mean:.4f} | {r.R2_mean:.4f} |")
    md=['# Experiment 3 results','',f"Run: `{runid}`",'',f"Data: {len(df):,} curated structures; SHA-256 `{data_hash}`.",'',f"Outer evaluation: {cfg['outer_folds']}-fold scaffold-grouped CV. No independent external cohort is available.",'','## Candidate summary','',*table,'','## Pooled out-of-fold scores','',f"The pooled out-of-fold table is saved at `../metrics/pooled_oof_metrics.csv`; these metrics aggregate every held-out molecule once.",'','## Interpretation','',f"Lowest mean outer-fold RMSE in this screening: **{screened_best}**. This is a candidate selected from comparisons on the same outer folds; its displayed performance is subject to selection optimism and must not be reported as an unbiased final estimate.",'','Paired scaffold-bootstrap comparisons with selected reference candidates are in `../metrics/screened_best_paired_scaffold_bootstrap.csv`. They are descriptive and do not correct for choosing the winner on these folds. Base-error correlations were calculated from leakage-controlled nested OOF predictions. See `../metrics/oof_residual_correlation.csv` and coefficient stability output.','',report['limitations'],'']
    (out/'reports'/'report.md').write_text('\n'.join(md),encoding='utf8')
    (out/'status.json').write_text(json.dumps({'status':'complete','run_id':runid,'completed_utc':datetime.now(timezone.utc).isoformat(),'outer_folds_completed':len(fold_records),'candidate_fold_scores':len(mdf)},indent=2),encoding='utf8')
    LOG.info('Experiment 3 complete: %s',out);print(out)
if __name__=='__main__':main()
