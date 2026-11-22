import numpy as np
import pytest
from rdkit import Chem
from src.qsar.benchmark import canonicalize, make_fingerprints, split_scaffolds
from src.qsar.experiment_01 import paired_scaffold_bootstrap

def test_canonical_smiles_groups_equivalent_encodings():
    a,_=canonicalize("CCO");b,_=canonicalize("OCC")
    assert a==b

def test_invalid_smiles_fails_loudly():
    with pytest.raises(ValueError): canonicalize("not a smiles")

def test_fingerprint_size_and_repeatability():
    x,_=make_fingerprints(["CCO","CCN"])
    assert x.shape==(2,2048)
    x2,_=make_fingerprints(["CCO","CCN"])
    assert np.array_equal(x,x2)

def test_scaffold_split_disjoint_and_deterministic():
    # Multiple distinct scaffolds and duplicate members within groups.
    groups=np.array(["a","a","b","b","c","c","d","d","e","e","f","f"])
    y=np.arange(len(groups),dtype=float)
    one=split_scaffolds(groups,y,7);two=split_scaffolds(groups,y,7)
    assert all(np.array_equal(a,b) for a,b in zip(one,two))
    g=[set(groups[i]) for i in one]
    assert all(not (g[i]&g[j]) for i in range(3) for j in range(i))

def test_paired_bootstrap_uses_scaffold_clusters():
    import pandas as pd
    frame=pd.DataFrame([
      {'repeat':1,'row_index':0,'scaffold':'a','observed_pIC50':7.,'model':'Stacking','predicted_pIC50':7.1},
      {'repeat':1,'row_index':0,'scaffold':'a','observed_pIC50':7.,'model':'RF','predicted_pIC50':7.3},
      {'repeat':1,'row_index':1,'scaffold':'b','observed_pIC50':8.,'model':'Stacking','predicted_pIC50':8.1},
      {'repeat':1,'row_index':1,'scaffold':'b','observed_pIC50':8.,'model':'RF','predicted_pIC50':8.3},
    ])
    result=paired_scaffold_bootstrap(frame,'RF',n_boot=20,seed=3)
    assert result['cluster_unit']=='Bemis-Murcko scaffold'
    assert result['RMSE_difference_95CI'][1] < 0
