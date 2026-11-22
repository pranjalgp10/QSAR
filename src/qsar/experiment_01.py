"""Repeated paired scaffold-CV model comparison (Experiment 1).

The immutable reference benchmark is not read for tuning. Run:
  .venv/Scripts/python.exe src/qsar/experiment_01.py
This intentionally uses a reduced, predeclared inner search (3 candidates) and
two repeats; increase repeats/trials only with sufficient compute and a new run ID.
"""
from __future__ import annotations
import json, logging, platform, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, sklearn, xgboost, rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data'/'processed'/'syk_qsar_curated.csv'
CFG=ROOT/'experiments'/'qsar'/'experiment_01_model_comparison'/'config.json'
LOG=logging.getLogger('experiment01')

def metrics(y,p):
    e=np.asarray(p)-np.asarray(y)
    return {'n':int(len(y)),'MAE':float(mean_absolute_error(y,p)),'MSE':float(mean_squared_error(y,p)),'RMSE':float(mean_squared_error(y,p)**.5),'R2':float(r2_score(y,p)),'median_absolute_error':float(np.median(np.abs(e))),'residual_mean':float(np.mean(e)),'residual_sd':float(np.std(e,ddof=1))}

def build_features(df,bits=2048,radius=2):
    smiles=df.canonical_smiles.astype(str).tolist();X=np.zeros((len(smiles),bits),dtype=np.uint8);mols=[]
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=bits)
    for i,s in enumerate(smiles):
        mol=Chem.MolFromSmiles(s)
        if mol is None: raise ValueError(f'Invalid canonical SMILES at row {i}')
        mols.append(mol);DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol),X[i])
    return X,mols

def candidates(seed):
    return {
      'RandomForest':(RandomForestRegressor(random_state=seed,n_jobs=-1),{'n_estimators':[200,300,400],'max_features':['sqrt',.5],'min_samples_leaf':[1,2,4],'max_depth':[None,28]}),
      'ExtraTrees':(ExtraTreesRegressor(random_state=seed,n_jobs=-1),{'n_estimators':[200,300,400],'max_features':['sqrt',.5],'min_samples_leaf':[1,2,4],'max_depth':[None,28]}),
      'SVR':(make_pipeline(StandardScaler(),SVR()),{'svr__C':[1,5,20,50],'svr__epsilon':[.1,.2,.4],'svr__gamma':['scale',.001,.01]}),
      'XGBoost':(XGBRegressor(objective='reg:squarederror',random_state=seed,n_jobs=1,verbosity=0),{'n_estimators':[200,300,400],'max_depth':[4,6,8],'learning_rate':[.03,.06],'subsample':[.8,1.]}),
      'HistGradientBoosting':(HistGradientBoostingRegressor(random_state=seed),{'max_iter':[150,250,350],'max_leaf_nodes':[15,31],'learning_rate':[.04,.07],'l2_regularization':[0.,2.]})}

def tune(name,X,y,g,seed,space_repeats=3):
    model,space=candidates(seed)[name]
    cv=list(GroupKFold(n_splits=min(3,len(np.unique(g)))).split(X,y,g))
    s=RandomizedSearchCV(model,space,n_iter=space_repeats,scoring='neg_root_mean_squared_error',cv=cv,n_jobs=1,refit=True,random_state=seed,return_train_score=True)
    LOG.info('Tuning %s on %d structures (%d grouped folds x %d candidates)',name,len(y),len(cv),space_repeats)
    t=time.perf_counter();s.fit(X,y)
    LOG.info('Finished %s: CV RMSE %.4f in %.1f s',name,-s.best_score_,time.perf_counter()-t)
    return s.best_estimator_,s.best_params_,float(-s.best_score_),time.perf_counter()-t

def fit_leakage_controlled_stack(X,y,g,outer_test_X,outer_test_y,seed):
    """Baseline stack with fixed, predeclared reference params and scaffold OOF.

    Keeping this baseline configuration fixed prevents selection information
    from leaking across its OOF meta-folds. Experiment 3 will optimize it.
    """
    base_names=['RandomForest','ExtraTrees','SVR','XGBoost','HistGradientBoosting']
    def fixed_models(s):
        return {
          'RandomForest':RandomForestRegressor(n_estimators=300,max_depth=28,min_samples_leaf=1,max_features=.5,random_state=s,n_jobs=-1),
          'ExtraTrees':ExtraTreesRegressor(n_estimators=300,max_depth=28,min_samples_leaf=1,max_features=.5,random_state=s,n_jobs=-1),
          'SVR':make_pipeline(StandardScaler(),SVR(C=5,epsilon=.1,gamma='scale')),
          'XGBoost':XGBRegressor(objective='reg:squarederror',n_estimators=250,max_depth=7,learning_rate=.03,subsample=.8,random_state=s,n_jobs=1,verbosity=0),
          'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=300,max_leaf_nodes=15,learning_rate=.04,l2_regularization=2.,random_state=s)}
    meta_cv=list(GroupKFold(n_splits=min(3,len(np.unique(g)))).split(X,y,g))
    oof=np.zeros((len(y),len(base_names)))
    for j,(itr,iva) in enumerate(meta_cv):
        for k,(name,model) in enumerate(fixed_models(seed+17*j).items()):
            model.fit(X[itr],y[itr]);oof[iva,k]=model.predict(X[iva])
    meta=Ridge(alpha=1.0).fit(oof,y)
    base_test=[];params={name:'fixed reference hyperparameters' for name in base_names}
    for name,model in fixed_models(seed+101).items():
        model.fit(X,y);base_test.append(model.predict(outer_test_X))
    pred=meta.predict(np.column_stack(base_test))
    return pred,meta.coef_.tolist(),params

def paired_scaffold_bootstrap(pred_df, competitor, n_boot=1000, seed=140):
    """Cluster bootstrap molecules by scaffold, pairing repeated predictions."""
    wide=pred_df.pivot(index=['repeat','row_index','scaffold','observed_pIC50'],columns='model',values='predicted_pIC50').reset_index()
    if wide[['Stacking',competitor]].isna().any().any(): raise ValueError('Missing paired OOF predictions')
    scaffolds,group_index=np.unique(wide.scaffold.to_numpy(),return_inverse=True);rng=np.random.default_rng(seed); vals=[]
    y=wide.observed_pIC50.to_numpy(); p_stack=wide.Stacking.to_numpy(); p_comp=wide[competitor].to_numpy()
    while len(vals)<n_boot:
        draws=rng.integers(0,len(scaffolds),size=len(scaffolds));counts=np.bincount(draws,minlength=len(scaffolds));w=counts[group_index]
        e1=p_stack-y;e2=p_comp-y;den=w.sum()
        mae1=np.sum(w*np.abs(e1))/den;mae2=np.sum(w*np.abs(e2))/den
        rmse1=np.sqrt(np.sum(w*e1*e1)/den);rmse2=np.sqrt(np.sum(w*e2*e2)/den)
        ym=np.sum(w*y)/den;sst=np.sum(w*(y-ym)**2)
        if sst<=0: continue
        r21=1-np.sum(w*e1*e1)/sst;r22=1-np.sum(w*e2*e2)/sst
        vals.append([mae1-mae2,rmse1-rmse2,r21-r22])
    a=np.asarray(vals)
    return {'MAE_difference_95CI':np.quantile(a[:,0],[.025,.975]).tolist(),'RMSE_difference_95CI':np.quantile(a[:,1],[.025,.975]).tolist(),'R2_difference_95CI':np.quantile(a[:,2],[.025,.975]).tolist(),'bootstrap_replicates':n_boot,'cluster_unit':'Bemis-Murcko scaffold'}

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--resume',type=Path,help='Resume an interrupted run directory from its last complete-fold checkpoint');args=parser.parse_args()
    cfg=json.loads(CFG.read_text(encoding='utf8'))
    run_root=ROOT/'results'/'qsar'/'experiment_01_model_comparison'
    run_id=args.resume.name if args.resume else datetime.now(timezone.utc).strftime('exp01_%Y%m%dT%H%M%SZ')
    out=(args.resume if args.resume else run_root/run_id)
    if args.resume and not out.is_absolute(): out=ROOT/out
    for sub in ['configs','predictions','metrics','manifests','logs','reports']: (out/sub).mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'logs'/'run.log',encoding='utf8'),logging.StreamHandler()],format='%(asctime)s %(levelname)s %(message)s')
    confpath=out/'configs'/'config.json'
    if args.resume and confpath.exists() and json.loads(confpath.read_text(encoding='utf8'))!=cfg:
        raise ValueError('Cannot resume: configuration differs from the saved run')
    confpath.write_text(json.dumps(cfg,indent=2),encoding='utf8')
    df=pd.read_csv(DATA);y=df.pIC50.to_numpy(float);g=df.scaffold_group.astype(str).to_numpy();X,mols=build_features(df,cfg['fingerprint']['bits'],cfg['fingerprint']['radius'])
    metrics_cp=out/'metrics'/'fold_metrics_checkpoint.csv';pred_cp=out/'predictions'/'outer_oof_predictions_checkpoint.csv';fold_cp=out/'manifests'/'folds_checkpoint.json'
    all_rows=pd.read_csv(metrics_cp).to_dict('records') if args.resume and metrics_cp.exists() else []
    all_predictions=pd.read_csv(pred_cp).to_dict('records') if args.resume and pred_cp.exists() else []
    all_folds=json.loads(fold_cp.read_text(encoding='utf8')) if args.resume and fold_cp.exists() else []
    completed={f'r{x["repeat"]}_f{x["fold"]}' for x in all_folds}
    for rep,seed in enumerate(cfg['split']['seeds']):
        outer=list(GroupKFold(n_splits=cfg['split']['outer_folds'],shuffle=True,random_state=seed).split(X,y,g))
        for fold,(itr,ite) in enumerate(outer):
            train_scaf=set(g[itr]);test_scaf=set(g[ite]);assert not train_scaf&test_scaf
            foldid=f'r{rep+1}_f{fold+1}'
            if foldid in completed: LOG.info('Skipping completed checkpoint %s',foldid);continue
            LOG.info('%s train=%d test=%d scaffolds=%d/%d',foldid,len(itr),len(ite),len(train_scaf),len(test_scaf))
            fold_rows=[];predictions={}
            for name in candidates(seed):
                model,par,inner_rmse,seconds=tune(name,X[itr],y[itr],g[itr],seed+fold,space_repeats=cfg['search']['n_iter'])
                # RandomizedSearchCV(refit=True) has already refit the best model on all outer-training rows.
                predictions[name]=model.predict(X[ite])
                m=metrics(y[ite],predictions[name]);fold_rows.append({'repeat':rep+1,'fold':fold+1,'fold_id':foldid,'model':name,**m,'inner_cv_rmse':inner_rmse,'fit_search_seconds':seconds,'params':json.dumps(par)})
            st,coef,sp=fit_leakage_controlled_stack(X[itr],y[itr],g[itr],X[ite],y[ite],seed+fold)
            predictions['Stacking']=st;fold_rows.append({'repeat':rep+1,'fold':fold+1,'fold_id':foldid,'model':'Stacking',**metrics(y[ite],st),'inner_cv_rmse':None,'fit_search_seconds':None,'params':json.dumps({'base':sp,'ridge_coef':coef})})
            all_rows.extend(fold_rows);all_folds.append({'repeat':rep+1,'fold':fold+1,'seed':seed,'train_indices':itr.tolist(),'test_indices':ite.tolist(),'train_scaffolds':len(train_scaf),'test_scaffolds':len(test_scaf)})
            for name,pred in predictions.items():
                for ix,p in zip(ite,pred): all_predictions.append({'repeat':rep+1,'fold':fold+1,'fold_id':foldid,'row_index':int(ix),'canonical_smiles':df.canonical_smiles.iloc[ix],'scaffold':g[ix],'observed_pIC50':float(y[ix]),'model':name,'predicted_pIC50':float(p)})
            pd.DataFrame(all_rows).to_csv(metrics_cp,index=False)
            pd.DataFrame(all_predictions).to_csv(pred_cp,index=False)
            fold_cp.write_text(json.dumps(all_folds),encoding='utf8')
    fold_df=pd.DataFrame(all_rows);pred_df=pd.DataFrame(all_predictions)
    fold_df.to_csv(out/'metrics'/'fold_metrics.csv',index=False);pred_df.to_csv(out/'predictions'/'outer_oof_predictions.csv',index=False)
    (out/'manifests'/'folds.json').write_text(json.dumps(all_folds,indent=2),encoding='utf8')
    summary=fold_df.groupby('model').agg(folds=('fold_id','count'),RMSE_mean=('RMSE','mean'),RMSE_sd=('RMSE','std'),RMSE_median=('RMSE','median'),RMSE_q25=('RMSE',lambda x:x.quantile(.25)),RMSE_q75=('RMSE',lambda x:x.quantile(.75)),MAE_mean=('MAE','mean'),R2_mean=('R2','mean')).reset_index()
    summary.to_csv(out/'metrics'/'summary.csv',index=False)
    pivot=fold_df.pivot_table(index='fold_id',columns='model',values=['RMSE','MAE','R2'])
    paired=[]
    for competitor in ['HistGradientBoosting','XGBoost','RandomForest','ExtraTrees','SVR']:
        boot=paired_scaffold_bootstrap(pred_df,competitor,seed=cfg['split']['seeds'][0])
        for metric in ['RMSE','MAE','R2']:
            d=pivot[metric]['Stacking']-pivot[metric][competitor]
            better=(d<0) if metric!='R2' else (d>0)
            paired.append({'metric':metric,'comparison':f'Stacking minus {competitor}','paired_fold_n':len(d),'mean_difference':float(d.mean()),'median_difference':float(d.median()),'folds_stacking_better':int(better.sum()),'fold_difference_sd':float(d.std(ddof=1)),'scaffold_cluster_bootstrap':boot})
    pd.DataFrame(paired).to_csv(out/'metrics'/'paired_model_differences.csv',index=False)
    report={'experiment':'01 repeated scaffold-aware model comparison','run_id':run_id,'scaffold_cv_repeats':len(cfg['split']['seeds']),'outer_folds':cfg['split']['outer_folds'],'search_trials_per_model_fold':cfg['search']['n_iter'],'summary':summary.to_dict(orient='records'),'paired_rmse':paired,'interpretation_limit':'Fold results are paired and scaffold-held-out. With few repeated folds, paired differences are descriptive; do not claim superiority from small differences. This is internal validation, not external validation.','software':{'python':platform.python_version(),'sklearn':sklearn.__version__,'rdkit':rdkit.__version__,'xgboost':xgboost.__version__}}
    (out/'reports'/'report.json').write_text(json.dumps(report,indent=2,default=float),encoding='utf8')
    LOG.info('Experiment complete: %s',out)
    print(out)

if __name__=='__main__':main()
