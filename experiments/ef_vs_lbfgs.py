"""EF vs L-BFGS on molecules not used to choose the initial Hessian.

h0 = 16 eV/A^2 was picked by sweeping on ethanol / benzaldehyde /
chlorobenzene / thioanisole / menthol. These six are held out, so the ratio
below is not fitted. Run with the project on PYTHONPATH.
"""
import numpy as np
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
import mlxmolkit.nddo.anal_grad as AG
from mlxmolkit.nddo.gradient import nddo_optimize
_o = AG.analytical_gradient; c = {'n': 0}
def counted(*a, **k):
    c['n'] += 1; return _o(*a, **k)
AG.analytical_gradient = counted

# HELD OUT: none of these were used to pick h0
HOLDOUT = [('anisole','COc1ccccc1'), ('cyclohexanol','OC1CCCCC1'),
           ('butyl acetate','CCCCOC(C)=O'), ('indole','c1ccc2[nH]ccc2c1'),
           ('geraniol','CC(C)=CCC/C(C)=C/CO'), ('camphor','CC1(C)C2CCC1(C)C(=O)C2')]
print(f"{'molecule':15s} {'LBFGS':>6s} {'EF':>5s} {'ratio':>6s} {'conv':>5s} {'|dHf|':>8s}")
tl = te = 0
for name, smi in HOLDOUT:
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    if AllChem.EmbedMolecule(m, randomSeed=42) != 0: continue
    Z = [a.GetAtomicNum() for a in m.GetAtoms()]; R = m.GetConformer().GetPositions()
    c['n'] = 0; l = nddo_optimize(Z, R.copy(), method='PM6'); nl = c['n']
    c['n'] = 0; e = nddo_optimize(Z, R.copy(), method='PM6', optimizer='ef'); ne = c['n']
    tl += nl; te += ne
    d = abs(l['heat_of_formation_kcal'] - e['heat_of_formation_kcal'])
    print(f"{name:15s} {nl:6d} {ne:5d} {ne/nl:6.2f}x {str(e['converged'])[:5]:>5s} {d:8.4f}")
print(f"{'TOTAL (held out)':15s} {tl:6d} {te:5d} {te/tl:6.2f}x")
