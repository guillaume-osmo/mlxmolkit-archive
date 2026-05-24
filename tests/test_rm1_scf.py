"""
Unit tests for RM1/AM1/PM6 SCF engine.

Validates against PYSEQM reference values (exact to < 0.001 eV).
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import pytest

from mlxmolkit.rm1.scf import rm1_energy, rm1_energy_batch


# Test geometries (Angstrom)
MOLS = {
    'H2':  ([1,1], np.array([[0,0,0],[0.74,0,0.0]])),
    'H2O': ([8,1,1], np.array([[0,0,0],[0.9584,0,0],[-0.2396,0.9275,0.0]])),
    'CH4': ([6,1,1,1,1], np.array([[0,0,0],[.6276,.6276,.6276],[.6276,-.6276,-.6276],[-.6276,.6276,-.6276],[-.6276,-.6276,.6276]])),
    'NH3': ([7,1,1,1], np.array([[0,0,0],[.9377,-.3816,0],[-.4689,.8119,0],[-.4689,-.4303,.8299]])),
}

# PYSEQM reference values (exact)
PYSEQM_REF = {
    'RM1': {
        'H2':  {'E_elec': -42.279154, 'E_nuc': 13.780913, 'E_tot': -28.498242},
        'H2O': {'E_elec': -488.951381, 'E_nuc': 143.380852, 'E_tot': -345.570528},
        'CH4': {'E_elec': -391.804076, 'E_nuc': 209.044867, 'E_tot': -182.759209},
        'NH3': {'E_elec': -439.119959, 'E_nuc': 186.812105, 'E_tot': -252.307854},
    },
    'AM1': {
        'H2':  {'E_elec': -40.847729, 'E_nuc': 13.376394, 'E_tot': -27.471336},
        'H2O': {'E_elec': -493.546003, 'E_nuc': 144.984542, 'E_tot': -348.561461},
        'CH4': {'E_elec': -386.849762, 'E_nuc': 203.668120, 'E_tot': -183.181642},
        'NH3': {'E_elec': -433.776574, 'E_nuc': 185.943634, 'E_tot': -247.832940},
    },
    'PM6': {
        'H2':  {'E_elec': -43.504671, 'E_nuc': 15.391764, 'E_tot': -28.112907},
        'H2O': {'E_elec': -457.613320, 'E_nuc': 138.540585, 'E_tot': -319.072734},
        'CH4': {'E_elec': -393.765398, 'E_nuc': 216.596686, 'E_tot': -177.168712},
        'NH3': {'E_elec': -409.149911, 'E_nuc': 189.328518, 'E_tot': -219.821393},
    },
}


@pytest.mark.parametrize("method", ["RM1", "AM1", "PM6"])
@pytest.mark.parametrize("mol_name", ["H2", "H2O", "CH4", "NH3"])
def test_energy_vs_pyseqm(method, mol_name):
    """Test that energies match PYSEQM to < 0.001 eV."""
    atoms, coords = MOLS[mol_name]
    ref = PYSEQM_REF[method][mol_name]

    result = rm1_energy(list(atoms), coords, method=method, max_iter=200, conv_tol=1e-8)

    assert result['converged'], f"{method} {mol_name} did not converge"
    assert abs(result['electronic_eV'] - ref['E_elec']) < 0.001, \
        f"{method} {mol_name} E_elec: {result['electronic_eV']:.6f} vs {ref['E_elec']:.6f}"
    assert abs(result['nuclear_eV'] - ref['E_nuc']) < 0.001, \
        f"{method} {mol_name} E_nuc: {result['nuclear_eV']:.6f} vs {ref['E_nuc']:.6f}"
    assert abs(result['energy_eV'] - ref['E_tot']) < 0.001, \
        f"{method} {mol_name} E_tot: {result['energy_eV']:.6f} vs {ref['E_tot']:.6f}"


def test_batch_matches_single():
    """Test that batch SCF gives same results as single molecule."""
    mol_list = [(list(a), c) for a, c in MOLS.values()]
    batch_results = rm1_energy_batch(mol_list, method='RM1', use_metal=False)

    for i, (name, (atoms, coords)) in enumerate(MOLS.items()):
        single = rm1_energy(list(atoms), coords, method='RM1')
        assert abs(batch_results[i]['energy_eV'] - single['energy_eV']) < 1e-10, \
            f"Batch vs single mismatch for {name}"


def test_all_methods_converge():
    """Test that all 7 methods converge for water."""
    atoms, coords = MOLS['H2O']
    for method in ['RM1', 'AM1', 'PM3', 'PM6', 'AM1_STAR', 'RM1_STAR']:
        result = rm1_energy(list(atoms), coords, method=method, max_iter=200)
        assert result['converged'], f"{method} did not converge for H2O"


def test_density_trace():
    """Test that Tr(P) = n_electrons for all methods."""
    atoms, coords = MOLS['H2O']
    for method in ['RM1', 'AM1', 'PM6']:
        result = rm1_energy(list(atoms), coords, method=method)
        P = result['density']
        n_elec = 8  # O(6) + H(1) + H(1)
        assert abs(np.trace(P) - n_elec) < 0.01, \
            f"{method} Tr(P) = {np.trace(P):.4f} != {n_elec}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
