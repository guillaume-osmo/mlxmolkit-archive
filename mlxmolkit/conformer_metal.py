"""
N×k parallel DG conformer generation with shared constraints.

One Metal threadgroup per conformer (C threadgroups total, where C = Σ k_i).
Each threadgroup has TPM=32 threads that parallelize energy computation,
line search, and L-BFGS two-loop recursion.

Constraints are stored ONCE per molecule and shared across k conformers
via ``conf_to_mol`` indirection.  Only positions differ between conformers
of the same molecule.

Adapted from shivampatel10/mlxmolkit's dg_lbfgs.py threadgroup model.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import mlx.core as mx

from .shared_batch import SharedConstraintBatch

DEFAULT_TPM = 32
DEFAULT_LBFGS_M = 8

# ---------------------------------------------------------------------------
# Metal kernel source (MSL)
# ---------------------------------------------------------------------------

_MSL_HEADER = """
// ---- Constants ----
constant float TOLX = 1.2e-6f;
constant float FUNCTOL = 1e-4f;
constant float MOVETOL = 1e-6f;
constant float MAX_STEP_FACTOR = 100.0f;
constant int MAX_LS_ITERS = 1000;

// ---- Distance violation energy (LOCAL indices + atom_off) ----
inline float dist_violation_e(
    const device float* pos, int i1, int i2,
    float lb2, float ub2, float wt, int dim, int atom_off
) {
    float d2 = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = pos[(atom_off + i1) * dim + d] - pos[(atom_off + i2) * dim + d];
        d2 += diff * diff;
    }
    float e = 0.0f;
    if (d2 > ub2) {
        float val = d2 / ub2 - 1.0f;
        e = wt * val * val;
    } else if (d2 < lb2) {
        float val = 2.0f * lb2 / (lb2 + d2) - 1.0f;
        e = wt * val * val;
    }
    return e;
}

// ---- Distance violation gradient (LOCAL indices + atom_off) ----
inline void dist_violation_g(
    const device float* pos, device float* grad,
    int i1, int i2, float lb2, float ub2, float wt, int dim, int atom_off
) {
    float d2 = 0.0f;
    float diff[4];
    for (int d = 0; d < dim; d++) {
        diff[d] = pos[(atom_off + i1) * dim + d] - pos[(atom_off + i2) * dim + d];
        d2 += diff[d] * diff[d];
    }
    float pf = 0.0f;
    if (d2 > ub2) {
        pf = wt * 4.0f * (d2 / ub2 - 1.0f) / ub2;
    } else if (d2 < lb2) {
        float l2d2 = d2 + lb2;
        pf = wt * 8.0f * lb2 * (1.0f - 2.0f * lb2 / l2d2) / (l2d2 * l2d2);
    }
    if (pf != 0.0f) {
        for (int d = 0; d < dim; d++) {
            float g = pf * diff[d];
            grad[(atom_off + i1) * dim + d] += g;
            grad[(atom_off + i2) * dim + d] -= g;
        }
    }
}

// ---- Chiral violation energy (LOCAL indices + atom_off) ----
inline float chiral_violation_e(
    const device float* pos,
    int i1, int i2, int i3, int i4,
    float vol_lower, float vol_upper, float wt, int dim, int atom_off
) {
    float v1[3], v2[3], v3[3];
    for (int d = 0; d < 3; d++) {
        v1[d] = pos[(atom_off+i1)*dim+d] - pos[(atom_off+i4)*dim+d];
        v2[d] = pos[(atom_off+i2)*dim+d] - pos[(atom_off+i4)*dim+d];
        v3[d] = pos[(atom_off+i3)*dim+d] - pos[(atom_off+i4)*dim+d];
    }
    float cx = v2[1]*v3[2] - v2[2]*v3[1];
    float cy = v2[2]*v3[0] - v2[0]*v3[2];
    float cz = v2[0]*v3[1] - v2[1]*v3[0];
    float vol = v1[0]*cx + v1[1]*cy + v1[2]*cz;
    float e = 0.0f;
    if (vol < vol_lower) { float d = vol - vol_lower; e = wt * d * d; }
    else if (vol > vol_upper) { float d = vol - vol_upper; e = wt * d * d; }
    return e;
}

// ---- Chiral violation gradient (LOCAL indices + atom_off) ----
inline void chiral_violation_g(
    const device float* pos, device float* grad,
    int i1, int i2, int i3, int i4,
    float vol_lower, float vol_upper, float wt, int dim, int atom_off
) {
    float v1[3], v2[3], v3[3];
    for (int d = 0; d < 3; d++) {
        v1[d] = pos[(atom_off+i1)*dim+d] - pos[(atom_off+i4)*dim+d];
        v2[d] = pos[(atom_off+i2)*dim+d] - pos[(atom_off+i4)*dim+d];
        v3[d] = pos[(atom_off+i3)*dim+d] - pos[(atom_off+i4)*dim+d];
    }
    float cx = v2[1]*v3[2] - v2[2]*v3[1];
    float cy = v2[2]*v3[0] - v2[0]*v3[2];
    float cz = v2[0]*v3[1] - v2[1]*v3[0];
    float vol = v1[0]*cx + v1[1]*cy + v1[2]*cz;
    float pf = 0.0f;
    if (vol < vol_lower) pf = 2.0f * wt * (vol - vol_lower);
    else if (vol > vol_upper) pf = 2.0f * wt * (vol - vol_upper);
    if (pf == 0.0f) return;
    float g1x = pf * cx, g1y = pf * cy, g1z = pf * cz;
    float g2x = pf * (v3[1]*v1[2] - v3[2]*v1[1]);
    float g2y = pf * (v3[2]*v1[0] - v3[0]*v1[2]);
    float g2z = pf * (v3[0]*v1[1] - v3[1]*v1[0]);
    float g3x = pf * (v2[2]*v1[1] - v2[1]*v1[2]);
    float g3y = pf * (v2[0]*v1[2] - v2[2]*v1[0]);
    float g3z = pf * (v2[1]*v1[0] - v2[0]*v1[1]);
    int o = atom_off;
    grad[(o+i1)*dim+0]+=g1x; grad[(o+i1)*dim+1]+=g1y; grad[(o+i1)*dim+2]+=g1z;
    grad[(o+i2)*dim+0]+=g2x; grad[(o+i2)*dim+1]+=g2y; grad[(o+i2)*dim+2]+=g2z;
    grad[(o+i3)*dim+0]+=g3x; grad[(o+i3)*dim+1]+=g3y; grad[(o+i3)*dim+2]+=g3z;
    grad[(o+i4)*dim+0]-=(g1x+g2x+g3x);
    grad[(o+i4)*dim+1]-=(g1y+g2y+g3y);
    grad[(o+i4)*dim+2]-=(g1z+g2z+g3z);
}

// ---- Fourth dimension energy/gradient ----
inline float fourth_dim_e(const device float* pos, int idx, float wt, int dim, int atom_off) {
    if (dim != 4) return 0.0f;
    float w = pos[(atom_off + idx) * dim + 3];
    return wt * w * w;
}
inline void fourth_dim_g(const device float* pos, device float* grad, int idx, float wt, int dim, int atom_off) {
    if (dim != 4) return;
    float w = pos[(atom_off + idx) * dim + 3];
    grad[(atom_off + idx) * dim + 3] += 2.0f * wt * w;
}

// ---- Threadgroup parallel primitives ----
inline float tg_reduce_sum(threadgroup float* s, uint tid, uint n) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float result = s[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}
inline float tg_reduce_max(threadgroup float* s, uint tid, uint n) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] = max(s[tid], s[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float result = s[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}
inline float parallel_dot(const device float* a, const device float* b,
    int n, uint tid, uint tpm, threadgroup float* s) {
    float sum = 0.0f;
    for (int i = (int)tid; i < n; i += (int)tpm) sum += a[i] * b[i];
    s[tid] = sum;
    return tg_reduce_sum(s, tid, tpm);
}
inline void parallel_saxpy(device float* a, float alpha, const device float* b,
    int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) a[i] += alpha * b[i];
    threadgroup_barrier(mem_flags::mem_device);
}
inline void parallel_scale(device float* a, float alpha, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) a[i] *= alpha;
    threadgroup_barrier(mem_flags::mem_device);
}
inline void parallel_copy(device float* dst, const device float* src, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) dst[i] = src[i];
    threadgroup_barrier(mem_flags::mem_device);
}
inline void parallel_set(device float* a, float val, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) a[i] = val;
    threadgroup_barrier(mem_flags::mem_device);
}
inline void parallel_neg_copy(device float* dst, const device float* src, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) dst[i] = -src[i];
    threadgroup_barrier(mem_flags::mem_device);
}
"""

# Main kernel body — one threadgroup per CONFORMER, TPM threads per threadgroup
# conf_to_mol indirection for shared constraints
_MSL_DG_BODY = """
    uint tid = thread_position_in_threadgroup.x;   // 0..TPM-1
    uint conf_idx = threadgroup_position_in_grid.x; // which conformer
    const uint tpm = TPM;
    const int lbfgs_m = LBFGS_M;

    threadgroup float shared[TPM];

    // Config
    int n_confs_cfg = (int)config[0];
    int max_iters = (int)config[1];
    float grad_tol = config[2];
    float chiral_weight = config[3];
    float fourth_dim_weight = config[4];
    int dim = (int)config[5];
    int total_pos_size = (int)config[6];

    if ((int)conf_idx >= n_confs_cfg) return;

    // ---- Shared constraint indirection ----
    int mol_idx = conf_to_mol[conf_idx];
    int atom_off = conf_atom_starts[conf_idx]; // this conformer's atom offset
    int n_atoms = mol_n_atoms[mol_idx];
    int n_vars = n_atoms * dim;

    // Constraint boundaries (per molecule — SHARED across conformers)
    int dist_start = dist_term_starts[mol_idx];
    int dist_end = dist_term_starts[mol_idx + 1];
    int chiral_start_t = chiral_term_starts[mol_idx];
    int chiral_end_t = chiral_term_starts[mol_idx + 1];
    int fourth_start_t = fourth_term_starts_arr[mol_idx];
    int fourth_end_t = fourth_term_starts_arr[mol_idx + 1];

    // L-BFGS history offset for this conformer
    int lbfgs_start = lbfgs_history_starts[conf_idx];

    // Copy initial positions to output (parallel)
    parallel_copy(&out_pos[atom_off * dim], &pos[atom_off * dim], n_vars, tid, tpm);

    // Working pointers (each conformer has its own slice)
    device float* my_pos = &out_pos[atom_off * dim];
    device float* my_grad = &work_grad[atom_off * dim];
    device float* my_dir = &work_dir[atom_off * dim];
    device float* my_old_pos = &work_scratch[atom_off * dim];
    device float* my_old_grad = &work_scratch[total_pos_size + atom_off * dim];
    device float* my_q = &work_scratch[2 * total_pos_size + atom_off * dim];

    device float* my_S = &work_lbfgs[lbfgs_start];
    device float* my_Y = &work_lbfgs[lbfgs_start + lbfgs_m * n_vars];
    device float* my_rho = &work_rho[conf_idx * lbfgs_m];

    // ---- Initial energy (parallel) + gradient ----
    parallel_set(my_grad, 0.0f, n_vars, tid, tpm);

    float local_energy = 0.0f;
    for (int t = dist_start + (int)tid; t < dist_end; t += (int)tpm)
        local_energy += dist_violation_e(out_pos, dist_pairs[t*2], dist_pairs[t*2+1],
            dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim, atom_off);
    for (int t = chiral_start_t + (int)tid; t < chiral_end_t; t += (int)tpm)
        local_energy += chiral_violation_e(out_pos,
            chiral_quads[t*4], chiral_quads[t*4+1], chiral_quads[t*4+2], chiral_quads[t*4+3],
            chiral_bounds[t*2], chiral_bounds[t*2+1], chiral_weight, dim, atom_off);
    for (int t = fourth_start_t + (int)tid; t < fourth_end_t; t += (int)tpm)
        local_energy += fourth_dim_e(out_pos, fourth_idx_arr[t], fourth_dim_weight, dim, atom_off);
    shared[tid] = local_energy;
    float energy = tg_reduce_sum(shared, tid, tpm);

#if PARALLEL_GRAD
    // All threads compute gradient via gather — each thread owns a stripe of vars
    for (int v = (int)tid; v < n_vars; v += (int)tpm) {
        int my_atom = v / dim; int my_coord = v % dim; float g = 0.0f;
        for (int t = dist_start; t < dist_end; t++) {
            int a = dist_pairs[t*2], b = dist_pairs[t*2+1];
            if (my_atom != a && my_atom != b) continue;
            float d2 = 0.0f; float dif[4];
            for (int d = 0; d < dim; d++) { dif[d] = out_pos[(atom_off+a)*dim+d] - out_pos[(atom_off+b)*dim+d]; d2 += dif[d]*dif[d]; }
            float lb2 = dist_bounds[t*3], ub2 = dist_bounds[t*3+1], wt = dist_bounds[t*3+2]; float pf = 0.0f;
            if (d2 > ub2) pf = wt*4.0f*(d2/ub2-1.0f)/ub2;
            else if (d2 < lb2) { float l = d2+lb2; pf = wt*8.0f*lb2*(1.0f-2.0f*lb2/l)/(l*l); }
            if (pf != 0.0f) g += ((my_atom==a)?1.0f:-1.0f) * pf * dif[my_coord];
        }
        if (my_coord < 3) {
            for (int t = chiral_start_t; t < chiral_end_t; t++) {
                int i1=chiral_quads[t*4],i2=chiral_quads[t*4+1],i3=chiral_quads[t*4+2],i4=chiral_quads[t*4+3];
                if (my_atom!=i1&&my_atom!=i2&&my_atom!=i3&&my_atom!=i4) continue;
                float v1[3],v2[3],v3[3];
                for (int d=0;d<3;d++){v1[d]=out_pos[(atom_off+i1)*dim+d]-out_pos[(atom_off+i4)*dim+d];v2[d]=out_pos[(atom_off+i2)*dim+d]-out_pos[(atom_off+i4)*dim+d];v3[d]=out_pos[(atom_off+i3)*dim+d]-out_pos[(atom_off+i4)*dim+d];}
                float cx=v2[1]*v3[2]-v2[2]*v3[1],cy=v2[2]*v3[0]-v2[0]*v3[2],cz=v2[0]*v3[1]-v2[1]*v3[0];
                float vol=v1[0]*cx+v1[1]*cy+v1[2]*cz;
                float vl=chiral_bounds[t*2],vu=chiral_bounds[t*2+1];float pf=0.0f;
                if(vol<vl)pf=2.0f*chiral_weight*(vol-vl);else if(vol>vu)pf=2.0f*chiral_weight*(vol-vu);
                if(pf!=0.0f){
                    float gc[4][3];gc[0][0]=pf*cx;gc[0][1]=pf*cy;gc[0][2]=pf*cz;
                    gc[1][0]=pf*(v3[1]*v1[2]-v3[2]*v1[1]);gc[1][1]=pf*(v3[2]*v1[0]-v3[0]*v1[2]);gc[1][2]=pf*(v3[0]*v1[1]-v3[1]*v1[0]);
                    gc[2][0]=pf*(v2[2]*v1[1]-v2[1]*v1[2]);gc[2][1]=pf*(v2[0]*v1[2]-v2[2]*v1[0]);gc[2][2]=pf*(v2[1]*v1[0]-v2[0]*v1[1]);
                    gc[3][0]=-(gc[0][0]+gc[1][0]+gc[2][0]);gc[3][1]=-(gc[0][1]+gc[1][1]+gc[2][1]);gc[3][2]=-(gc[0][2]+gc[1][2]+gc[2][2]);
                    int at4[4]={i1,i2,i3,i4};for(int k=0;k<4;k++){if(my_atom==at4[k])g+=gc[k][my_coord];}
                }
            }
        }
        if (dim==4&&my_coord==3) { for (int t=fourth_start_t;t<fourth_end_t;t++) { if(my_atom==fourth_idx_arr[t]) g+=2.0f*fourth_dim_weight*out_pos[(atom_off+my_atom)*dim+3]; } }
        my_grad[v] = g;
    }
    threadgroup_barrier(mem_flags::mem_device);
#else
    // Thread 0 computes gradient serially (no atomics)
    if (tid == 0) {
        for (int t = dist_start; t < dist_end; t++)
            dist_violation_g(out_pos, work_grad, dist_pairs[t*2], dist_pairs[t*2+1],
                dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim, atom_off);
        for (int t = chiral_start_t; t < chiral_end_t; t++)
            chiral_violation_g(out_pos, work_grad,
                chiral_quads[t*4], chiral_quads[t*4+1], chiral_quads[t*4+2], chiral_quads[t*4+3],
                chiral_bounds[t*2], chiral_bounds[t*2+1], chiral_weight, dim, atom_off);
        for (int t = fourth_start_t; t < fourth_end_t; t++)
            fourth_dim_g(out_pos, work_grad, fourth_idx_arr[t], fourth_dim_weight, dim, atom_off);
    }
    threadgroup_barrier(mem_flags::mem_device);
#endif

    parallel_neg_copy(my_dir, my_grad, n_vars, tid, tpm);

    float local_sum_sq = 0.0f;
    for (int i = (int)tid; i < n_vars; i += (int)tpm) local_sum_sq += my_pos[i] * my_pos[i];
    shared[tid] = local_sum_sq;
    float sum_sq = tg_reduce_sum(shared, tid, tpm);
    float max_step = MAX_STEP_FACTOR * max(sqrt(sum_sq), (float)n_vars);

    int status = 1;
    int hist_count = 0;
    int hist_idx = 0;

    for (int iter = 0; iter < max_iters && status == 1; iter++) {
        parallel_copy(my_old_pos, my_pos, n_vars, tid, tpm);
        float old_energy = energy;

        float local_dir_sq = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) local_dir_sq += my_dir[i] * my_dir[i];
        shared[tid] = local_dir_sq;
        float dir_norm = sqrt(tg_reduce_sum(shared, tid, tpm));
        if (dir_norm > max_step) parallel_scale(my_dir, max_step / dir_norm, n_vars, tid, tpm);

        float slope = parallel_dot(my_dir, my_grad, n_vars, tid, tpm, shared);

        float local_test_max = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            float ad = abs(my_dir[i]);
            float ap = max(abs(my_pos[i]), 1.0f);
            float t = ad / ap;
            if (t > local_test_max) local_test_max = t;
        }
        shared[tid] = local_test_max;
        float lambda_min = MOVETOL / max(tg_reduce_max(shared, tid, tpm), 1e-30f);

        float lam = 1.0f, prev_lam = 1.0f, prev_e = old_energy;
        bool ls_done = false;

        for (int ls_iter = 0; ls_iter < MAX_LS_ITERS && !ls_done; ls_iter++) {
            if (lam < lambda_min) { parallel_copy(my_pos, my_old_pos, n_vars, tid, tpm); ls_done = true; break; }

            for (int i = (int)tid; i < n_vars; i += (int)tpm)
                my_pos[i] = my_old_pos[i] + lam * my_dir[i];
            threadgroup_barrier(mem_flags::mem_device);

            float local_trial_e = 0.0f;
            for (int t = dist_start + (int)tid; t < dist_end; t += (int)tpm)
                local_trial_e += dist_violation_e(out_pos, dist_pairs[t*2], dist_pairs[t*2+1],
                    dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim, atom_off);
            for (int t = chiral_start_t + (int)tid; t < chiral_end_t; t += (int)tpm)
                local_trial_e += chiral_violation_e(out_pos,
                    chiral_quads[t*4], chiral_quads[t*4+1], chiral_quads[t*4+2], chiral_quads[t*4+3],
                    chiral_bounds[t*2], chiral_bounds[t*2+1], chiral_weight, dim, atom_off);
            for (int t = fourth_start_t + (int)tid; t < fourth_end_t; t += (int)tpm)
                local_trial_e += fourth_dim_e(out_pos, fourth_idx_arr[t], fourth_dim_weight, dim, atom_off);
            shared[tid] = local_trial_e;
            float trial_e = tg_reduce_sum(shared, tid, tpm);

            if (trial_e - old_energy <= FUNCTOL * lam * slope) {
                energy = trial_e; ls_done = true;
            } else {
                float tmp_lam;
                if (ls_iter == 0) {
                    tmp_lam = -slope / (2.0f * (trial_e - old_energy - slope));
                } else {
                    float rhs1 = trial_e - old_energy - lam * slope;
                    float rhs2 = prev_e - old_energy - prev_lam * slope;
                    float lam_sq = lam * lam, lam2_sq = prev_lam * prev_lam;
                    float denom_v = lam - prev_lam;
                    if (abs(denom_v) < 1e-30f) { tmp_lam = 0.5f * lam; }
                    else {
                        float a = (rhs1/lam_sq - rhs2/lam2_sq) / denom_v;
                        float b = (-prev_lam*rhs1/lam_sq + lam*rhs2/lam2_sq) / denom_v;
                        if (abs(a) < 1e-30f) tmp_lam = (abs(b)>1e-30f) ? -slope/(2.0f*b) : 0.5f*lam;
                        else {
                            float disc = b*b - 3.0f*a*slope;
                            if (disc < 0.0f) tmp_lam = 0.5f*lam;
                            else if (b <= 0.0f) tmp_lam = (-b+sqrt(disc))/(3.0f*a);
                            else tmp_lam = -slope/(b+sqrt(disc));
                        }
                    }
                }
                tmp_lam = clamp(tmp_lam, 0.1f * lam, 0.5f * lam);
                prev_lam = lam; prev_e = trial_e; lam = tmp_lam;
            }
        }

        if (!ls_done) parallel_copy(my_pos, my_old_pos, n_vars, tid, tpm);

        for (int i = (int)tid; i < n_vars; i += (int)tpm)
            my_old_pos[i] = my_pos[i] - my_old_pos[i]; // s_k
        threadgroup_barrier(mem_flags::mem_device);

        float local_tolx = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            float t = abs(my_old_pos[i]) / max(abs(my_pos[i]), 1.0f);
            if (t > local_tolx) local_tolx = t;
        }
        shared[tid] = local_tolx;
        if (tg_reduce_max(shared, tid, tpm) < TOLX) { status = 0; break; }

        parallel_copy(my_old_grad, my_grad, n_vars, tid, tpm);
        parallel_set(my_grad, 0.0f, n_vars, tid, tpm);

        float local_new_e = 0.0f;
        for (int t = dist_start + (int)tid; t < dist_end; t += (int)tpm)
            local_new_e += dist_violation_e(out_pos, dist_pairs[t*2], dist_pairs[t*2+1],
                dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim, atom_off);
        for (int t = chiral_start_t + (int)tid; t < chiral_end_t; t += (int)tpm)
            local_new_e += chiral_violation_e(out_pos,
                chiral_quads[t*4], chiral_quads[t*4+1], chiral_quads[t*4+2], chiral_quads[t*4+3],
                chiral_bounds[t*2], chiral_bounds[t*2+1], chiral_weight, dim, atom_off);
        for (int t = fourth_start_t + (int)tid; t < fourth_end_t; t += (int)tpm)
            local_new_e += fourth_dim_e(out_pos, fourth_idx_arr[t], fourth_dim_weight, dim, atom_off);
        shared[tid] = local_new_e;
        energy = tg_reduce_sum(shared, tid, tpm);

#if PARALLEL_GRAD
        for (int v = (int)tid; v < n_vars; v += (int)tpm) {
            int my_a2 = v / dim; int my_c2 = v % dim; float g2 = 0.0f;
            for (int t = dist_start; t < dist_end; t++) {
                int a = dist_pairs[t*2], b = dist_pairs[t*2+1];
                if (my_a2 != a && my_a2 != b) continue;
                float d2 = 0.0f; float df[4];
                for (int d = 0; d < dim; d++) { df[d] = out_pos[(atom_off+a)*dim+d] - out_pos[(atom_off+b)*dim+d]; d2 += df[d]*df[d]; }
                float lb2 = dist_bounds[t*3], ub2 = dist_bounds[t*3+1], wt = dist_bounds[t*3+2]; float pf = 0.0f;
                if (d2 > ub2) pf = wt*4.0f*(d2/ub2-1.0f)/ub2;
                else if (d2 < lb2) { float l = d2+lb2; pf = wt*8.0f*lb2*(1.0f-2.0f*lb2/l)/(l*l); }
                if (pf != 0.0f) g2 += ((my_a2==a)?1.0f:-1.0f) * pf * df[my_c2];
            }
            if (my_c2 < 3) {
                for (int t = chiral_start_t; t < chiral_end_t; t++) {
                    int i1=chiral_quads[t*4],i2=chiral_quads[t*4+1],i3=chiral_quads[t*4+2],i4=chiral_quads[t*4+3];
                    if (my_a2!=i1&&my_a2!=i2&&my_a2!=i3&&my_a2!=i4) continue;
                    float u1[3],u2[3],u3[3];
                    for (int d=0;d<3;d++){u1[d]=out_pos[(atom_off+i1)*dim+d]-out_pos[(atom_off+i4)*dim+d];u2[d]=out_pos[(atom_off+i2)*dim+d]-out_pos[(atom_off+i4)*dim+d];u3[d]=out_pos[(atom_off+i3)*dim+d]-out_pos[(atom_off+i4)*dim+d];}
                    float cx=u2[1]*u3[2]-u2[2]*u3[1],cy=u2[2]*u3[0]-u2[0]*u3[2],cz=u2[0]*u3[1]-u2[1]*u3[0];
                    float vol=u1[0]*cx+u1[1]*cy+u1[2]*cz;
                    float vl=chiral_bounds[t*2],vu=chiral_bounds[t*2+1];float pf=0.0f;
                    if(vol<vl)pf=2.0f*chiral_weight*(vol-vl);else if(vol>vu)pf=2.0f*chiral_weight*(vol-vu);
                    if(pf!=0.0f){float gc[4][3];gc[0][0]=pf*cx;gc[0][1]=pf*cy;gc[0][2]=pf*cz;gc[1][0]=pf*(u3[1]*u1[2]-u3[2]*u1[1]);gc[1][1]=pf*(u3[2]*u1[0]-u3[0]*u1[2]);gc[1][2]=pf*(u3[0]*u1[1]-u3[1]*u1[0]);gc[2][0]=pf*(u2[2]*u1[1]-u2[1]*u1[2]);gc[2][1]=pf*(u2[0]*u1[2]-u2[2]*u1[0]);gc[2][2]=pf*(u2[1]*u1[0]-u2[0]*u1[1]);gc[3][0]=-(gc[0][0]+gc[1][0]+gc[2][0]);gc[3][1]=-(gc[0][1]+gc[1][1]+gc[2][1]);gc[3][2]=-(gc[0][2]+gc[1][2]+gc[2][2]);int at4[4]={i1,i2,i3,i4};for(int k=0;k<4;k++){if(my_a2==at4[k])g2+=gc[k][my_c2];}}
                }
            }
            if (dim==4&&my_c2==3){for(int t=fourth_start_t;t<fourth_end_t;t++){if(my_a2==fourth_idx_arr[t])g2+=2.0f*fourth_dim_weight*out_pos[(atom_off+my_a2)*dim+3];}}
            my_grad[v] = g2;
        }
        threadgroup_barrier(mem_flags::mem_device);
#else
        if (tid == 0) {
            for (int t = dist_start; t < dist_end; t++)
                dist_violation_g(out_pos, work_grad, dist_pairs[t*2], dist_pairs[t*2+1],
                    dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim, atom_off);
            for (int t = chiral_start_t; t < chiral_end_t; t++)
                chiral_violation_g(out_pos, work_grad,
                    chiral_quads[t*4], chiral_quads[t*4+1], chiral_quads[t*4+2], chiral_quads[t*4+3],
                    chiral_bounds[t*2], chiral_bounds[t*2+1], chiral_weight, dim, atom_off);
            for (int t = fourth_start_t; t < fourth_end_t; t++)
                fourth_dim_g(out_pos, work_grad, fourth_idx_arr[t], fourth_dim_weight, dim, atom_off);
        }
        threadgroup_barrier(mem_flags::mem_device);
#endif

        float local_grad_test = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            float t = abs(my_grad[i]) * max(abs(my_pos[i]), 1.0f);
            if (t > local_grad_test) local_grad_test = t;
        }
        shared[tid] = local_grad_test;
        if (tg_reduce_max(shared, tid, tpm) / max(energy, 1.0f) < grad_tol) { status = 0; break; }

        // L-BFGS update
        for (int i = (int)tid; i < n_vars; i += (int)tpm)
            my_q[i] = my_grad[i] - my_old_grad[i]; // y_k
        threadgroup_barrier(mem_flags::mem_device);

        float ys_dot = parallel_dot(my_q, my_old_pos, n_vars, tid, tpm, shared);

        if (ys_dot > 1e-10f) {
            int slot = hist_idx % lbfgs_m;
            parallel_copy(&my_S[slot * n_vars], my_old_pos, n_vars, tid, tpm);
            parallel_copy(&my_Y[slot * n_vars], my_q, n_vars, tid, tpm);
            if (tid == 0) my_rho[slot] = 1.0f / ys_dot;
            threadgroup_barrier(mem_flags::mem_device);
            hist_idx++;
            if (hist_count < lbfgs_m) hist_count++;
        }

        // Two-loop recursion
        parallel_copy(my_q, my_grad, n_vars, tid, tpm);
        device float* my_alpha = &work_alpha[conf_idx * lbfgs_m];

        for (int j = hist_count - 1; j >= 0; j--) {
            int slot = (hist_idx - 1 - (hist_count - 1 - j)) % lbfgs_m;
            if (slot < 0) slot += lbfgs_m;
            float alpha_j = my_rho[slot] * parallel_dot(&my_S[slot*n_vars], my_q, n_vars, tid, tpm, shared);
            if (tid == 0) my_alpha[j] = alpha_j;
            threadgroup_barrier(mem_flags::mem_device);
            parallel_saxpy(my_q, -alpha_j, &my_Y[slot*n_vars], n_vars, tid, tpm);
        }
        if (hist_count > 0) {
            int newest = (hist_idx - 1) % lbfgs_m;
            if (newest < 0) newest += lbfgs_m;
            float sy = parallel_dot(&my_S[newest*n_vars], &my_Y[newest*n_vars], n_vars, tid, tpm, shared);
            float yy = parallel_dot(&my_Y[newest*n_vars], &my_Y[newest*n_vars], n_vars, tid, tpm, shared);
            parallel_scale(my_q, sy / max(yy, 1e-30f), n_vars, tid, tpm);
        }
        for (int j = 0; j < hist_count; j++) {
            int slot = (hist_idx - 1 - (hist_count - 1 - j)) % lbfgs_m;
            if (slot < 0) slot += lbfgs_m;
            float beta_j = my_rho[slot] * parallel_dot(&my_Y[slot*n_vars], my_q, n_vars, tid, tpm, shared);
            float alpha_j = my_alpha[j];
            parallel_saxpy(my_q, alpha_j - beta_j, &my_S[slot*n_vars], n_vars, tid, tpm);
        }
        parallel_neg_copy(my_dir, my_q, n_vars, tid, tpm);
    }

    if (tid == 0) {
        out_energies[conf_idx] = energy;
        out_statuses[conf_idx] = status;
    }
"""

# Cache: (tpm, lbfgs_m, parallel_grad) → compiled kernel
_dg_kernel_cache: dict[tuple, object] = {}


def _build_dg_kernel(
    tpm: int = DEFAULT_TPM,
    lbfgs_m: int = DEFAULT_LBFGS_M,
    parallel_grad: bool = False,
):
    """Compile the DG L-BFGS Metal kernel with shared constraints.

    Parameters
    ----------
    parallel_grad : bool
        If True, all TPM threads compute gradient in parallel using
        atomic scatter-add.  Faster for molecules with many constraints
        (>500 terms) but adds atomic overhead.  Default False (thread 0
        serial gradient, zero atomic cost).
    """
    pg_flag = "1" if parallel_grad else "0"
    header = _MSL_HEADER.replace("TPM", str(tpm)).replace("LBFGS_M", str(lbfgs_m))
    header = f"#define PARALLEL_GRAD {pg_flag}\n" + header
    source = _MSL_DG_BODY.replace("TPM", str(tpm)).replace("LBFGS_M", str(lbfgs_m))

    return mx.fast.metal_kernel(
        name=f"dg_lbfgs_shared_pg{pg_flag}",
        input_names=[
            "pos", "config",
            "conf_to_mol", "conf_atom_starts", "mol_n_atoms",
            "dist_term_starts", "dist_pairs", "dist_bounds",
            "chiral_term_starts", "chiral_quads", "chiral_bounds",
            "fourth_term_starts_arr", "fourth_idx_arr",
            "lbfgs_history_starts",
        ],
        output_names=[
            "out_pos", "out_energies", "out_statuses",
            "work_grad", "work_dir", "work_scratch",
            "work_lbfgs", "work_rho", "work_alpha",
        ],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


def _get_dg_kernel(
    tpm: int = DEFAULT_TPM,
    lbfgs_m: int = DEFAULT_LBFGS_M,
    parallel_grad: bool = False,
):
    key = (tpm, lbfgs_m, parallel_grad)
    if key not in _dg_kernel_cache:
        _dg_kernel_cache[key] = _build_dg_kernel(tpm, lbfgs_m, parallel_grad)
    return _dg_kernel_cache[key]


def dg_minimize_shared(
    batch: SharedConstraintBatch,
    positions: np.ndarray,
    *,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    chiral_weight: float = 1.0,
    fourth_dim_weight: float = 0.1,
    tpm: int = DEFAULT_TPM,
    lbfgs_m: int = DEFAULT_LBFGS_M,
    parallel_grad: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run DG L-BFGS on all C conformers in parallel with shared constraints.

    Parameters
    ----------
    batch : SharedConstraintBatch
        N molecules × k conformers with shared constraints.
    positions : np.ndarray, shape (n_atoms_total * dim,)
        Initial positions (random 4D).
    parallel_grad : bool
        If True, all TPM threads compute gradient via atomic scatter-add.
        Useful for large molecules (>500 distance constraints).
        Default False (thread 0 serial — no atomic overhead).

    Returns
    -------
    out_positions : np.ndarray
        Optimized positions.
    energies : np.ndarray, shape (C,)
        Final energy per conformer.
    statuses : np.ndarray, shape (C,)
        0 = converged, 1 = max_iters reached.
    """
    C = batch.n_confs_total
    dim = batch.dim
    total_pos_size = int(batch.conf_atom_starts[-1]) * dim

    # Pack config (config[6] = total_pos_size for kernel scratch indexing)
    config = np.array([
        C, max_iters, grad_tol, chiral_weight, fourth_dim_weight, dim,
        total_pos_size,
    ], dtype=np.float32)

    # Pack constraint arrays (LOCAL indices, interleaved)
    n_dist = len(batch.dist_idx1)
    if n_dist > 0:
        dist_pairs = np.stack([batch.dist_idx1, batch.dist_idx2], axis=1).flatten().astype(np.int32)
        dist_bounds = np.stack([batch.dist_lb2, batch.dist_ub2, batch.dist_weight], axis=1).flatten().astype(np.float32)
    else:
        dist_pairs = np.zeros(2, dtype=np.int32)
        dist_bounds = np.zeros(3, dtype=np.float32)

    n_chiral = len(batch.chiral_idx1)
    if n_chiral > 0:
        chiral_quads = np.stack([
            batch.chiral_idx1, batch.chiral_idx2,
            batch.chiral_idx3, batch.chiral_idx4,
        ], axis=1).flatten().astype(np.int32)
        chiral_bounds = np.stack([
            batch.chiral_vol_lower, batch.chiral_vol_upper,
        ], axis=1).flatten().astype(np.float32)
    else:
        chiral_quads = np.zeros(4, dtype=np.int32)
        chiral_bounds = np.zeros(2, dtype=np.float32)

    # L-BFGS history starts per conformer
    lbfgs_starts = np.zeros(C + 1, dtype=np.int32)
    for c in range(C):
        n_atoms = batch.mol_n_atoms[batch.conf_to_mol[c]]
        n_vars = n_atoms * dim
        lbfgs_starts[c + 1] = lbfgs_starts[c] + 2 * lbfgs_m * n_vars
    total_lbfgs = int(lbfgs_starts[-1])

    # Convert to MLX
    kernel = _get_dg_kernel(tpm, lbfgs_m, parallel_grad)
    results = kernel(
        inputs=[
            mx.array(positions),
            mx.array(config),
            mx.array(batch.conf_to_mol),
            mx.array(batch.conf_atom_starts),
            mx.array(batch.mol_n_atoms),
            mx.array(batch.dist_term_starts),
            mx.array(dist_pairs),
            mx.array(dist_bounds),
            mx.array(batch.chiral_term_starts),
            mx.array(chiral_quads),
            mx.array(chiral_bounds),
            mx.array(batch.fourth_term_starts),
            mx.array(batch.fourth_idx),
            mx.array(lbfgs_starts[:-1]),  # per-conformer starts
        ],
        grid=(C, 1, 1),
        threadgroup=(tpm, 1, 1),
        output_shapes=[
            (total_pos_size,),    # out_pos
            (C,),                 # out_energies
            (C,),                 # out_statuses
            (total_pos_size,),    # work_grad
            (total_pos_size,),    # work_dir
            (3 * total_pos_size,),  # work_scratch (old_pos, old_grad, q)
            (max(1, total_lbfgs),),  # work_lbfgs (S + Y history)
            (max(1, C * lbfgs_m),),  # work_rho
            (max(1, C * lbfgs_m),),  # work_alpha
        ],
        output_dtypes=[
            mx.float32, mx.float32, mx.int32,
            mx.float32, mx.float32, mx.float32,
            mx.float32, mx.float32, mx.float32,
        ],
    )
    mx.eval(results[0], results[1], results[2])

    out_pos = np.array(results[0])
    energies = np.array(results[1])
    statuses = np.array(results[2])

    return out_pos, energies, statuses
