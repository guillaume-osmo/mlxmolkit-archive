"""
MMFF94 optimization using Shivam's in-kernel BFGS Metal kernel.

Entire BFGS loop runs on GPU — zero CPU round-trips.
Two variants:
  - Single-thread per conformer (for large molecules)
  - Threadgroup-parallel TG=32 (for typical molecules, faster Hessian)

MMFF params are small (O(atoms), not O(atoms²)), so replicating them
per conformer has negligible memory overhead.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import mlx.core as mx

from .mmff_params import MMFFParams, extract_mmff_params

_KERNEL_DIR = Path(__file__).parent

# Load Metal shader source files
_MSL_HEADER = (_KERNEL_DIR / "mmff_bfgs_header.metal").read_text()
_MSL_SOURCE = (_KERNEL_DIR / "mmff_bfgs_source.metal").read_text()
_MSL_SOURCE_TG = (_KERNEL_DIR / "mmff_bfgs_source_tg.metal").read_text()
_MSL_SOURCE_LBFGS_TG = (_KERNEL_DIR / "mmff_lbfgs_source_tg.metal").read_text()

TG_SIZE = 32
LBFGS_M = 8
MAX_ATOMS_METAL = 64

# Kernel caches
_mmff_kernel = None
_mmff_kernel_tg = None
_mmff_kernel_lbfgs_tg = None


def _pack_mmff_for_nk(
    mmff_params_list: List[MMFFParams],
    conf_counts: List[int],
) -> dict:
    """Pack N molecules × k conformers into flat Metal buffers.

    Each conformer is treated as a separate 'molecule' in the batch.
    Params are replicated k times per molecule (negligible overhead).
    """
    n_mols_logical = len(mmff_params_list)
    C = sum(conf_counts)

    # Build atom_starts for C 'molecules' (each conformer = 1 molecule)
    atom_starts = np.zeros(C + 1, dtype=np.int32)
    hessian_starts = np.zeros(C + 1, dtype=np.int32)
    c = 0
    for mol_idx, k in enumerate(conf_counts):
        n_a = mmff_params_list[mol_idx].n_atoms
        n_terms = n_a * 3
        for _ in range(k):
            atom_starts[c + 1] = atom_starts[c] + n_a
            hessian_starts[c + 1] = hessian_starts[c] + n_terms * n_terms
            c += 1

    total_pos_size = int(atom_starts[-1]) * 3
    total_hessian_size = int(hessian_starts[-1])

    def _i32(a): return np.array(a, dtype=np.int32) if len(a) > 0 else np.zeros(0, dtype=np.int32)
    def _f32(a): return np.array(a, dtype=np.float32) if len(a) > 0 else np.zeros(0, dtype=np.float32)
    def _pack_pairs(i1, i2, off):
        if len(i1) > 0: return np.stack([_i32(i1)+off, _i32(i2)+off], axis=1).flatten()
        return np.zeros(2, dtype=np.int32)
    def _pack_trips(i1, i2, i3, off):
        if len(i1) > 0: return np.stack([_i32(i1)+off, _i32(i2)+off, _i32(i3)+off], axis=1).flatten()
        return np.zeros(3, dtype=np.int32)
    def _pack_quads(i1, i2, i3, i4, off):
        if len(i1) > 0: return np.stack([_i32(i1)+off, _i32(i2)+off, _i32(i3)+off, _i32(i4)+off], axis=1).flatten()
        return np.zeros(4, dtype=np.int32)
    def _pack_params(*arrs, fb=1):
        if len(arrs[0]) > 0: return np.stack([_f32(a) for a in arrs], axis=1).flatten()
        return np.zeros(fb, dtype=np.float32)

    # Build per-term-type starts arrays (C+1 each) and flat term data
    all_bond_pairs, all_bond_params = [], []
    all_angle_trips, all_angle_params = [], []
    all_sb_trips, all_sb_params = [], []
    all_oop_quads, all_oop_params = [], []
    all_tor_quads, all_tor_params = [], []
    all_vdw_pairs, all_vdw_params = [], []
    all_ele_pairs, all_ele_params = [], []

    bond_ts = np.zeros(C + 1, dtype=np.int32)
    angle_ts = np.zeros(C + 1, dtype=np.int32)
    sb_ts = np.zeros(C + 1, dtype=np.int32)
    oop_ts = np.zeros(C + 1, dtype=np.int32)
    tor_ts = np.zeros(C + 1, dtype=np.int32)
    vdw_ts = np.zeros(C + 1, dtype=np.int32)
    ele_ts = np.zeros(C + 1, dtype=np.int32)

    deg_to_rad = np.pi / 180.0
    c = 0
    for mol_idx, k in enumerate(conf_counts):
        p = mmff_params_list[mol_idx]
        for _ in range(k):
            off = int(atom_starts[c])
            nb, na, nsb = len(p.bond_idx1), len(p.angle_idx1), len(p.strbend_idx1)
            noop, ntor = len(p.oop_idx1), len(p.torsion_idx1)
            nvdw, nele = len(p.vdw_idx1), len(p.ele_idx1)

            bond_ts[c+1] = bond_ts[c] + nb
            angle_ts[c+1] = angle_ts[c] + na
            sb_ts[c+1] = sb_ts[c] + nsb
            oop_ts[c+1] = oop_ts[c] + noop
            tor_ts[c+1] = tor_ts[c] + ntor
            vdw_ts[c+1] = vdw_ts[c] + nvdw
            ele_ts[c+1] = ele_ts[c] + nele

            if nb > 0:
                all_bond_pairs.append(_pack_pairs(p.bond_idx1, p.bond_idx2, off))
                all_bond_params.append(_pack_params(p.bond_kb, p.bond_r0, fb=2))
            if na > 0:
                all_angle_trips.append(_pack_trips(p.angle_idx1, p.angle_idx2, p.angle_idx3, off))
                all_angle_params.append(_pack_params(p.angle_ka, p.angle_theta0 * deg_to_rad, p.angle_is_linear.astype(np.float32), fb=3))
            if nsb > 0:
                all_sb_trips.append(_pack_trips(p.strbend_idx1, p.strbend_idx2, p.strbend_idx3, off))
                all_sb_params.append(_pack_params(p.strbend_r0_ij, p.strbend_r0_kj, p.strbend_theta0 * deg_to_rad, p.strbend_kba_ijk, p.strbend_kba_kji, fb=5))
            if noop > 0:
                all_oop_quads.append(_pack_quads(p.oop_idx1, p.oop_idx2, p.oop_idx3, p.oop_idx4, off))
                all_oop_params.append(_f32(p.oop_koop))
            if ntor > 0:
                all_tor_quads.append(_pack_quads(p.torsion_idx1, p.torsion_idx2, p.torsion_idx3, p.torsion_idx4, off))
                all_tor_params.append(_pack_params(p.torsion_V1, p.torsion_V2, p.torsion_V3, fb=3))
            if nvdw > 0:
                all_vdw_pairs.append(_pack_pairs(p.vdw_idx1, p.vdw_idx2, off))
                all_vdw_params.append(_pack_params(p.vdw_R_star, p.vdw_eps, fb=2))
            if nele > 0:
                all_ele_pairs.append(_pack_pairs(p.ele_idx1, p.ele_idx2, off))
                ele_scale = np.where(p.ele_is_1_4 > 0.5, 0.75, 1.0).astype(np.float32)
                all_ele_params.append(_pack_params(p.ele_charge_term, np.ones(nele, dtype=np.float32), ele_scale, fb=3))
            c += 1

    def _cat_or_fb(parts, fb_size):
        return np.concatenate(parts) if parts else np.zeros(fb_size, dtype=parts[0].dtype if parts else np.float32)

    all_ts = np.concatenate([bond_ts, angle_ts, sb_ts, oop_ts, tor_ts, vdw_ts, ele_ts])

    return {
        'atom_starts': mx.array(atom_starts),
        'hessian_starts': mx.array(hessian_starts),
        'config': mx.array(np.array([C, 200, 1e-4], dtype=np.float32)),
        'all_term_starts': mx.array(all_ts),
        'bond_pairs': mx.array(_cat_or_fb(all_bond_pairs, 2) if all_bond_pairs else np.zeros(2, dtype=np.int32)),
        'bond_params': mx.array(_cat_or_fb(all_bond_params, 2) if all_bond_params else np.zeros(2, dtype=np.float32)),
        'angle_trips': mx.array(_cat_or_fb(all_angle_trips, 3) if all_angle_trips else np.zeros(3, dtype=np.int32)),
        'angle_params': mx.array(_cat_or_fb(all_angle_params, 3) if all_angle_params else np.zeros(3, dtype=np.float32)),
        'sb_trips': mx.array(_cat_or_fb(all_sb_trips, 3) if all_sb_trips else np.zeros(3, dtype=np.int32)),
        'sb_params': mx.array(_cat_or_fb(all_sb_params, 5) if all_sb_params else np.zeros(5, dtype=np.float32)),
        'oop_quads': mx.array(_cat_or_fb(all_oop_quads, 4) if all_oop_quads else np.zeros(4, dtype=np.int32)),
        'oop_params': mx.array(_cat_or_fb(all_oop_params, 1) if all_oop_params else np.zeros(1, dtype=np.float32)),
        'tor_quads': mx.array(_cat_or_fb(all_tor_quads, 4) if all_tor_quads else np.zeros(4, dtype=np.int32)),
        'tor_params': mx.array(_cat_or_fb(all_tor_params, 3) if all_tor_params else np.zeros(3, dtype=np.float32)),
        'vdw_pairs': mx.array(_cat_or_fb(all_vdw_pairs, 2) if all_vdw_pairs else np.zeros(2, dtype=np.int32)),
        'vdw_params': mx.array(_cat_or_fb(all_vdw_params, 2) if all_vdw_params else np.zeros(2, dtype=np.float32)),
        'ele_pairs': mx.array(_cat_or_fb(all_ele_pairs, 2) if all_ele_pairs else np.zeros(2, dtype=np.int32)),
        'ele_params': mx.array(_cat_or_fb(all_ele_params, 3) if all_ele_params else np.zeros(3, dtype=np.float32)),
        'total_pos_size': total_pos_size,
        'total_hessian_size': total_hessian_size,
        'C': C,
    }


def _get_mmff_kernel_tg():
    global _mmff_kernel_tg
    if _mmff_kernel_tg is None:
        tg_header = _MSL_HEADER + f"\nconstant int TG_SIZE_VAL = {TG_SIZE};\n"
        _mmff_kernel_tg = mx.fast.metal_kernel(
            name="mmff_bfgs_tg",
            input_names=[
                "pos", "atom_starts", "hessian_starts", "config",
                "all_term_starts",
                "bond_pairs", "bond_params",
                "angle_trips", "angle_params",
                "sb_trips", "sb_params",
                "oop_quads", "oop_params",
                "tor_quads", "tor_params",
                "vdw_pairs", "vdw_params",
                "ele_pairs", "ele_params",
            ],
            output_names=[
                "out_pos", "out_energies", "out_statuses",
                "work_grad", "work_dir", "work_scratch", "work_hessian",
            ],
            header=tg_header,
            source=_MSL_SOURCE_TG,
        )
    return _mmff_kernel_tg


def _get_mmff_kernel_lbfgs_tg():
    global _mmff_kernel_lbfgs_tg
    if _mmff_kernel_lbfgs_tg is None:
        tg_header = _MSL_HEADER + f"\nconstant int TG_SIZE_VAL = {TG_SIZE};\nconstant int LBFGS_M_VAL = {LBFGS_M};\n"
        _mmff_kernel_lbfgs_tg = mx.fast.metal_kernel(
            name="mmff_lbfgs_tg",
            input_names=[
                "pos", "atom_starts", "lbfgs_starts", "config",
                "all_term_starts",
                "bond_pairs", "bond_params",
                "angle_trips", "angle_params",
                "sb_trips", "sb_params",
                "oop_quads", "oop_params",
                "tor_quads", "tor_params",
                "vdw_pairs", "vdw_params",
                "ele_pairs", "ele_params",
            ],
            output_names=[
                "out_pos", "out_energies", "out_statuses",
                "work_grad", "work_dir", "work_scratch",
                "work_lbfgs", "work_rho", "work_alpha",
            ],
            header=tg_header,
            source=_MSL_SOURCE_LBFGS_TG,
        )
    return _mmff_kernel_lbfgs_tg


def mmff_minimize_nk(
    mmff_params_list: List[MMFFParams],
    conf_counts: List[int],
    positions_3d: np.ndarray,
    *,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    use_lbfgs: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MMFF94 optimization entirely on GPU — zero CPU round-trips.

    Parameters
    ----------
    mmff_params_list : list of MMFFParams
        One per molecule (replicated per conformer internally).
    conf_counts : list of int
        k_i conformers per molecule.
    positions_3d : np.ndarray
        Flat (total_coords,) float32.
    use_lbfgs : bool
        If True, use L-BFGS (O(mn) memory, no dense Hessian).
        If False (default), use full BFGS (O(n²) Hessian, faster for
        small molecules but more memory).

    Returns
    -------
    out_pos, energies, converged
    """
    packed = _pack_mmff_for_nk(mmff_params_list, conf_counts)
    packed['config'] = mx.array(np.array([packed['C'], max_iters, grad_tol], dtype=np.float32))

    C = packed['C']
    total_pos_size = packed['total_pos_size']
    total_hessian_size = packed['total_hessian_size']

    if use_lbfgs:
        # L-BFGS: replace dense Hessian with history vectors
        lbfgs_starts = np.zeros(C + 1, dtype=np.int32)
        for c in range(C):
            n_a = int(packed['atom_starts'][c + 1].item()) - int(packed['atom_starts'][c].item())
            n_terms = n_a * 3
            lbfgs_starts[c + 1] = lbfgs_starts[c] + 2 * LBFGS_M * n_terms
        total_lbfgs = int(lbfgs_starts[-1])

        kernel = _get_mmff_kernel_lbfgs_tg()
        outputs = kernel(
            inputs=[
                mx.array(positions_3d),
                packed['atom_starts'], mx.array(lbfgs_starts), packed['config'],
                packed['all_term_starts'],
                packed['bond_pairs'], packed['bond_params'],
                packed['angle_trips'], packed['angle_params'],
                packed['sb_trips'], packed['sb_params'],
                packed['oop_quads'], packed['oop_params'],
                packed['tor_quads'], packed['tor_params'],
                packed['vdw_pairs'], packed['vdw_params'],
                packed['ele_pairs'], packed['ele_params'],
            ],
            output_shapes=[
                (total_pos_size,), (C,), (C,),
                (total_pos_size,), (total_pos_size,),
                (total_pos_size * 3,),
                (max(1, total_lbfgs),),
                (max(1, C * LBFGS_M),),
                (max(1, C * LBFGS_M),),
            ],
            output_dtypes=[
                mx.float32, mx.float32, mx.int32,
                mx.float32, mx.float32, mx.float32,
                mx.float32, mx.float32, mx.float32,
            ],
            grid=(C * TG_SIZE, 1, 1),
            threadgroup=(TG_SIZE, 1, 1),
            template=[("total_pos_size", total_pos_size)],
        )
    else:
        # Full BFGS with dense Hessian
        kernel = _get_mmff_kernel_tg()
        outputs = kernel(
            inputs=[
                mx.array(positions_3d),
                packed['atom_starts'], packed['hessian_starts'], packed['config'],
                packed['all_term_starts'],
                packed['bond_pairs'], packed['bond_params'],
                packed['angle_trips'], packed['angle_params'],
                packed['sb_trips'], packed['sb_params'],
                packed['oop_quads'], packed['oop_params'],
                packed['tor_quads'], packed['tor_params'],
                packed['vdw_pairs'], packed['vdw_params'],
                packed['ele_pairs'], packed['ele_params'],
            ],
            output_shapes=[
                (total_pos_size,), (C,), (C,),
                (total_pos_size,), (total_pos_size,),
                (total_pos_size * 3,),
                (max(total_hessian_size, 1),),
            ],
            output_dtypes=[
                mx.float32, mx.float32, mx.int32,
                mx.float32, mx.float32, mx.float32, mx.float32,
            ],
            grid=(C * TG_SIZE, 1, 1),
            threadgroup=(TG_SIZE, 1, 1),
            template=[("total_pos_size", total_pos_size)],
        )

    mx.eval(outputs[0], outputs[1], outputs[2])
    return np.array(outputs[0]), np.array(outputs[1]), np.array(outputs[2]) == 0
