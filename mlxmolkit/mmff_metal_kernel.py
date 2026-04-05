"""
Fused Metal kernel for MMFF94 energy + analytic gradient.

One thread per conformer computes ALL 7 MMFF terms (energy + gradient)
in a single kernel launch. This eliminates:
  - Auto-diff backward pass overhead
  - Multiple kernel dispatch (one per MLX op)
  - Intermediate memory allocations
  - Python-level loop over energy terms

Architecture:
  - Parameters packed into 3 flat buffers: idx_buf (int32), param_buf (float32), meta (uint32)
  - Grid: (C,) threads, one per conformer
  - Each thread reads shared parameters + its own positions
  - Each thread writes to its own energy + gradient slice
  - No atomic operations needed (no cross-thread writes)

Reference: Halgren, J. Comput. Chem. 1996, 17, 490-519 (MMFF94)
           GROMACS manual (torsion/OOP gradient formulas)
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlxmolkit.mmff_params import MMFFParams

# ---------------------------------------------------------------------------
# Metal kernel source (MSL) — fused energy + analytic gradient
# ---------------------------------------------------------------------------

_MMFF_KERNEL_SOURCE = """
const float MDYNE = 143.9325f;
const float CS = -2.0f;
const float CB = -0.007f;
const float ABC = 0.043844f;
const float SBC = 2.51210f;
const float ELC = 332.0716f;
const float ELD = 0.05f;
const float R2D = 57.295779513082323f;
const float KEPS = 1e-12f;

#define GP(atom) float3(pos[pos_off + uint(atom)*3], pos[pos_off + uint(atom)*3+1], pos[pos_off + uint(atom)*3+2])
#define AG(atom, fv) { uint _b = pos_off + uint(atom)*3; grad_out[_b] += (fv).x; grad_out[_b+1] += (fv).y; grad_out[_b+2] += (fv).z; }

uint cid = thread_position_in_grid.x;

// Unpack meta
uint N    = meta[0];
uint nb   = meta[1];
uint na   = meta[2];
uint nsb  = meta[3];
uint noop = meta[4];
uint ntor = meta[5];
uint nvdw = meta[6];
uint nele = meta[7];

uint dim = N * 3;
uint pos_off = cid * dim;

// ---- Initialize gradient to zero ----
for (uint i = 0; i < dim; i++) {
    grad_out[pos_off + i] = 0.0f;
}

float E = 0.0f;

// ==== Index offsets into idx_buf ====
uint ib_i = 0,             ib_j = nb;
uint ia_i = 2*nb,          ia_j = 2*nb + na,       ia_k = 2*nb + 2*na;
uint is_i = 2*nb + 3*na,   is_j = is_i + nsb,      is_k = is_j + nsb;
uint io_i = is_k + nsb,    io_j = io_i + noop,     io_k = io_j + noop,  io_l = io_k + noop;
uint it_i = io_l + noop,   it_j = it_i + ntor,     it_k = it_j + ntor,  it_l = it_k + ntor;
uint iv_i = it_l + ntor,   iv_j = iv_i + nvdw;
uint ie_i = iv_j + nvdw,   ie_j = ie_i + nele;

// ==== Float offsets into param_buf ====
uint pb_kb = 0,            pb_r0 = nb;
uint pa_ka = 2*nb,         pa_t0 = 2*nb + na;
uint ps_kij = 2*nb+2*na,   ps_kji = ps_kij+nsb, ps_rij = ps_kji+nsb, ps_rkj = ps_rij+nsb, ps_t0 = ps_rkj+nsb;
uint po_k  = ps_t0 + nsb;
uint pt_V1 = po_k + noop,  pt_V2 = pt_V1+ntor, pt_V3 = pt_V2+ntor;
uint pv_R  = pt_V3 + ntor, pv_e = pv_R + nvdw;
uint pe_ct = pv_e + nvdw,  pe_sc = pe_ct + nele;

// ================================================================
// 1. BOND STRETCH
// ================================================================
for (uint b = 0; b < nb; b++) {
    int ai = idx_buf[ib_i + b], aj = idx_buf[ib_j + b];
    float kb = param_buf[pb_kb + b], r0 = param_buf[pb_r0 + b];
    float3 d = GP(ai) - GP(aj);
    float r = length(d) + KEPS;
    float dr = r - r0;
    E += MDYNE * kb * 0.5f * dr * dr * (1.0f + CS*dr + (7.0f/12.0f)*CS*CS*dr*dr);
    float dE = MDYNE * kb * dr * (1.0f + 1.5f*CS*dr + (7.0f/6.0f)*CS*CS*dr*dr);
    float3 f = (dE / r) * d;
    AG(ai, f);
    AG(aj, -f);
}

// ================================================================
// 2. ANGLE BEND
// ================================================================
for (uint a = 0; a < na; a++) {
    int ai = idx_buf[ia_i + a], aj = idx_buf[ia_j + a], ak = idx_buf[ia_k + a];
    float ka = param_buf[pa_ka + a], theta0 = param_buf[pa_t0 + a];
    float3 v1 = GP(ai) - GP(aj);
    float3 v2 = GP(ak) - GP(aj);
    float r1 = length(v1) + KEPS, r2 = length(v2) + KEPS;
    float ct = clamp(dot(v1, v2) / (r1 * r2), -1.0f + 1e-7f, 1.0f - 1e-7f);
    float theta = acos(ct);
    float st = sin(theta);
    st = max(st, KEPS);
    float dt = (theta - theta0) * R2D;
    E += ABC * ka * 0.5f * dt * dt * (1.0f + CB * dt);
    float dE = ABC * ka * dt * (1.0f + 1.5f * CB * dt) * R2D;
    float3 gi = dE * (ct * v1 / r1 - v2 / r2) / (r1 * st);
    float3 gk = dE * (ct * v2 / r2 - v1 / r1) / (r2 * st);
    AG(ai, gi);
    AG(aj, -gi - gk);
    AG(ak, gk);
}

// ================================================================
// 3. STRETCH-BEND
// ================================================================
for (uint s = 0; s < nsb; s++) {
    int ai = idx_buf[is_i + s], aj = idx_buf[is_j + s], ak = idx_buf[is_k + s];
    float kij = param_buf[ps_kij + s], kji = param_buf[ps_kji + s];
    float r0ij = param_buf[ps_rij + s], r0kj = param_buf[ps_rkj + s];
    float theta0 = param_buf[ps_t0 + s];
    float3 vij = GP(ai) - GP(aj);
    float3 vkj = GP(ak) - GP(aj);
    float rij = length(vij) + KEPS, rkj = length(vkj) + KEPS;
    float ct = clamp(dot(vij, vkj) / (rij * rkj), -1.0f + 1e-7f, 1.0f - 1e-7f);
    float theta = acos(ct);
    float st = max(sin(theta), KEPS);
    float drij = rij - r0ij, drkj = rkj - r0kj;
    float dtheta_deg = (theta - theta0) * R2D;
    E += SBC * (kij * drij + kji * drkj) * dtheta_deg;
    float dE_dth = SBC * (kij * drij + kji * drkj) * R2D;
    float dE_drij = SBC * kij * dtheta_deg;
    float dE_drkj = SBC * kji * dtheta_deg;
    float3 dij_hat = vij / rij, dkj_hat = vkj / rkj;
    float3 dt_dvi = (ct * vij / rij - vkj / rkj) / (rij * st);
    float3 dt_dvk = (ct * vkj / rkj - vij / rij) / (rkj * st);
    float3 gi = dE_dth * dt_dvi + dE_drij * dij_hat;
    float3 gk = dE_dth * dt_dvk + dE_drkj * dkj_hat;
    AG(ai, gi);
    AG(aj, -gi - gk);
    AG(ak, gk);
}

// ================================================================
// 4. OUT-OF-PLANE BENDING
// ================================================================
for (uint o = 0; o < noop; o++) {
    int a1 = idx_buf[io_i+o], a2 = idx_buf[io_j+o], a3 = idx_buf[io_k+o], a4 = idx_buf[io_l+o];
    float koop = param_buf[po_k + o];
    float3 u = GP(a1) - GP(a2);
    float3 v = GP(a3) - GP(a2);
    float3 w = GP(a4) - GP(a2);
    float3 nv = cross(u, v);
    float A = length(nv) + KEPS, B = length(w) + KEPS;
    float AB = A * B;
    float3 nh = nv / A, wh = w / B;
    float sinc = clamp(dot(nv, w) / AB, -1.0f, 1.0f);
    float chi = asin(sinc);
    float chi_d = chi * R2D;
    float cosc = cos(chi);
    cosc = max(cosc, KEPS);
    E += ABC * koop * 0.5f * chi_d * chi_d;
    float dE = ABC * koop * chi_d * R2D / cosc;
    float3 vxw = cross(v, w), wxu = cross(w, u);
    float3 vxn = cross(v, nh), nxu = cross(nh, u);
    float3 ds1 = vxw / AB - sinc * vxn / A;
    float3 ds3 = wxu / AB - sinc * nxu / A;
    float3 ds4 = (nh - sinc * wh) / B;
    float3 ds2 = -(ds1 + ds3 + ds4);
    AG(a1, dE * ds1);
    AG(a2, dE * ds2);
    AG(a3, dE * ds3);
    AG(a4, dE * ds4);
}

// ================================================================
// 5. TORSION (energy + GROMACS/CHARMM analytic gradient)
// ================================================================
for (uint t = 0; t < ntor; t++) {
    int a1 = idx_buf[it_i+t], a2 = idx_buf[it_j+t], a3 = idx_buf[it_k+t], a4 = idx_buf[it_l+t];
    float V1 = param_buf[pt_V1+t], V2 = param_buf[pt_V2+t], V3 = param_buf[pt_V3+t];
    float3 p1 = GP(a1), p2 = GP(a2);
    float3 p3 = GP(a3), p4 = GP(a4);
    float3 b1 = p2 - p1, b2 = p3 - p2, b3 = p4 - p3;
    float3 c1 = cross(b1, b2), c2 = cross(b2, b3);
    float b2n = length(b2) + KEPS;
    float3 mv = cross(c1, b2 / b2n);
    float n1n2 = sqrt(dot(c1,c1) * dot(c2,c2) + KEPS);
    float sw = dot(mv, c2) / n1n2, cw = dot(c1, c2) / n1n2;
    float omega = atan2(sw, cw);
    E += 0.5f * (V1*(1.0f+cos(omega)) + V2*(1.0f-cos(2.0f*omega)) + V3*(1.0f+cos(3.0f*omega)));
    float dE = 0.5f * (-V1*sin(omega) + 2.0f*V2*sin(2.0f*omega) - 3.0f*V3*sin(3.0f*omega));
    // GROMACS torsion gradient
    float3 rij = p1 - p2, rkj = p3 - p2, rkl = p3 - p4;
    float3 m = cross(rij, rkj), n = cross(rkj, rkl);
    float msq = dot(m, m) + KEPS, nsq = dot(n, n) + KEPS;
    float rkjn = length(rkj) + KEPS, rkjn2 = rkjn * rkjn;
    float3 fi = (-rkjn / msq) * m * dE;
    float3 fl = ( rkjn / nsq) * n * dE;
    float pij = dot(rij, rkj) / rkjn2, pkl = dot(rkl, rkj) / rkjn2;
    float3 fj = -fi + pij * fi - pkl * fl;
    float3 fk = -fl - pij * fi + pkl * fl;
    AG(a1, fi);
    AG(a2, fj);
    AG(a3, fk);
    AG(a4, fl);
}

// ================================================================
// 6. VAN DER WAALS (Buffered 14-7)
// ================================================================
for (uint v = 0; v < nvdw; v++) {
    int ai = idx_buf[iv_i + v], aj = idx_buf[iv_j + v];
    float Rstar = param_buf[pv_R + v], eps_v = param_buf[pv_e + v];
    float3 d = GP(ai) - GP(aj);
    float r = length(d) + KEPS;
    float3 dh = d / r;
    float rho = r / Rstar;
    float rho7 = rho*rho*rho*rho*rho*rho*rho;
    float t1 = 1.07f / (rho + 0.07f);
    float t2 = 1.12f / (rho7 + 0.12f) - 2.0f;
    float t1_7 = t1*t1*t1*t1*t1*t1*t1;
    E += eps_v * t1_7 * t2;
    float dt1 = -1.07f / ((rho+0.07f)*(rho+0.07f));
    float dt2 = -1.12f * 7.0f * rho*rho*rho*rho*rho*rho / ((rho7+0.12f)*(rho7+0.12f));
    float dE = eps_v * (7.0f * t1_7 / t1 * dt1 * t2 + t1_7 * dt2) / Rstar;
    float3 f = dE * dh;
    AG(ai, f);
    AG(aj, -f);
}

// ================================================================
// 7. ELECTROSTATIC (constant dielectric)
// ================================================================
for (uint e = 0; e < nele; e++) {
    int ai = idx_buf[ie_i + e], aj = idx_buf[ie_j + e];
    float ct = param_buf[pe_ct + e], sc = param_buf[pe_sc + e];
    float3 d = GP(ai) - GP(aj);
    float r = length(d) + KEPS;
    float3 dh = d / r;
    float denom = r + ELD;
    E += sc * ELC * ct / denom;
    float dE = -sc * ELC * ct / (denom * denom);
    float3 f = dE * dh;
    AG(ai, f);
    AG(aj, -f);
}

energy_out[cid] = E;
"""

# ---------------------------------------------------------------------------
# Mega-kernel: many molecules in a single kernel launch
# ---------------------------------------------------------------------------
# Each thread looks up its molecule via a dispatch table, so all conformers
# of all molecules run in one GPU dispatch.
#
# Extra inputs vs the single-molecule kernel:
#   all_meta     (M*8,)  uint32  – packed per-molecule meta
#   dispatch     (T*3,)  uint32  – [meta_off, idx_base, param_base] per conf
#   pos_offsets  (T,)    uint32  – flat position offset per conformer

_MMFF_MEGA_KERNEL_SOURCE = """
const float MDYNE = 143.9325f;
const float CS = -2.0f;
const float CB = -0.007f;
const float ABC = 0.043844f;
const float SBC = 2.51210f;
const float ELC = 332.0716f;
const float ELD = 0.05f;
const float R2D = 57.295779513082323f;
const float KEPS = 1e-12f;

uint cid = thread_position_in_grid.x;

// ---- Per-conformer dispatch: look up molecule params ----
uint mol_meta_off = dispatch[cid * 3u];
uint idx_base     = dispatch[cid * 3u + 1u];
uint param_base   = dispatch[cid * 3u + 2u];

uint N    = all_meta[mol_meta_off];
uint nb   = all_meta[mol_meta_off + 1u];
uint na   = all_meta[mol_meta_off + 2u];
uint nsb  = all_meta[mol_meta_off + 3u];
uint noop = all_meta[mol_meta_off + 4u];
uint ntor = all_meta[mol_meta_off + 5u];
uint nvdw = all_meta[mol_meta_off + 6u];
uint nele = all_meta[mol_meta_off + 7u];

uint dim = N * 3u;
uint pos_off = pos_offsets[cid];

#define GP(atom) float3(pos[pos_off + uint(atom)*3u], pos[pos_off + uint(atom)*3u+1u], pos[pos_off + uint(atom)*3u+2u])
#define AG(atom, fv) { uint _b = pos_off + uint(atom)*3u; grad_out[_b] += (fv).x; grad_out[_b+1u] += (fv).y; grad_out[_b+2u] += (fv).z; }

// Initialize gradient to zero
for (uint i = 0u; i < dim; i++) {
    grad_out[pos_off + i] = 0.0f;
}
float E = 0.0f;

// ==== Index offsets (with base) ====
uint ib_i = idx_base,                           ib_j = idx_base + nb;
uint ia_i = idx_base + 2u*nb,                   ia_j = idx_base + 2u*nb + na,       ia_k = idx_base + 2u*nb + 2u*na;
uint is_i = idx_base + 2u*nb + 3u*na,           is_j = is_i + nsb,                  is_k = is_j + nsb;
uint io_i = is_k + nsb,                          io_j = io_i + noop,                 io_k = io_j + noop,  io_l = io_k + noop;
uint it_i = io_l + noop,                         it_j = it_i + ntor,                 it_k = it_j + ntor,  it_l = it_k + ntor;
uint iv_i = it_l + ntor,                         iv_j = iv_i + nvdw;
uint ie_i = iv_j + nvdw,                         ie_j = ie_i + nele;

// ==== Float offsets (with base) ====
uint pb_kb = param_base,                         pb_r0 = param_base + nb;
uint pa_ka = param_base + 2u*nb,                 pa_t0 = param_base + 2u*nb + na;
uint ps_kij = param_base + 2u*nb + 2u*na,       ps_kji = ps_kij+nsb, ps_rij = ps_kji+nsb, ps_rkj = ps_rij+nsb, ps_t0 = ps_rkj+nsb;
uint po_k  = ps_t0 + nsb;
uint pt_V1 = po_k + noop,                       pt_V2 = pt_V1+ntor, pt_V3 = pt_V2+ntor;
uint pv_R  = pt_V3 + ntor,                      pv_e = pv_R + nvdw;
uint pe_ct = pv_e + nvdw,                       pe_sc = pe_ct + nele;

// ================================================================
// 1. BOND STRETCH
// ================================================================
for (uint b = 0u; b < nb; b++) {
    int ai = idx_buf[ib_i + b], aj = idx_buf[ib_j + b];
    float kb = param_buf[pb_kb + b], r0 = param_buf[pb_r0 + b];
    float3 d = GP(ai) - GP(aj);
    float r = length(d) + KEPS;
    float dr = r - r0;
    E += MDYNE * kb * 0.5f * dr * dr * (1.0f + CS*dr + (7.0f/12.0f)*CS*CS*dr*dr);
    float dE = MDYNE * kb * dr * (1.0f + 1.5f*CS*dr + (7.0f/6.0f)*CS*CS*dr*dr);
    float3 f = (dE / r) * d;
    AG(ai, f);
    AG(aj, -f);
}

// ================================================================
// 2. ANGLE BEND
// ================================================================
for (uint a = 0u; a < na; a++) {
    int ai = idx_buf[ia_i + a], aj = idx_buf[ia_j + a], ak = idx_buf[ia_k + a];
    float ka = param_buf[pa_ka + a], theta0 = param_buf[pa_t0 + a];
    float3 v1 = GP(ai) - GP(aj);
    float3 v2 = GP(ak) - GP(aj);
    float r1 = length(v1) + KEPS, r2 = length(v2) + KEPS;
    float ct = clamp(dot(v1, v2) / (r1 * r2), -1.0f + 1e-7f, 1.0f - 1e-7f);
    float theta = acos(ct);
    float st = sin(theta);
    st = max(st, KEPS);
    float dt = (theta - theta0) * R2D;
    E += ABC * ka * 0.5f * dt * dt * (1.0f + CB * dt);
    float dE = ABC * ka * dt * (1.0f + 1.5f * CB * dt) * R2D;
    float3 gi = dE * (ct * v1 / r1 - v2 / r2) / (r1 * st);
    float3 gk = dE * (ct * v2 / r2 - v1 / r1) / (r2 * st);
    AG(ai, gi);
    AG(aj, -gi - gk);
    AG(ak, gk);
}

// ================================================================
// 3. STRETCH-BEND
// ================================================================
for (uint s = 0u; s < nsb; s++) {
    int ai = idx_buf[is_i + s], aj = idx_buf[is_j + s], ak = idx_buf[is_k + s];
    float kij = param_buf[ps_kij + s], kji = param_buf[ps_kji + s];
    float r0ij = param_buf[ps_rij + s], r0kj = param_buf[ps_rkj + s];
    float theta0 = param_buf[ps_t0 + s];
    float3 vij = GP(ai) - GP(aj);
    float3 vkj = GP(ak) - GP(aj);
    float rij = length(vij) + KEPS, rkj = length(vkj) + KEPS;
    float ct = clamp(dot(vij, vkj) / (rij * rkj), -1.0f + 1e-7f, 1.0f - 1e-7f);
    float theta = acos(ct);
    float st = max(sin(theta), KEPS);
    float drij = rij - r0ij, drkj = rkj - r0kj;
    float dtheta_deg = (theta - theta0) * R2D;
    E += SBC * (kij * drij + kji * drkj) * dtheta_deg;
    float dE_dth = SBC * (kij * drij + kji * drkj) * R2D;
    float dE_drij = SBC * kij * dtheta_deg;
    float dE_drkj = SBC * kji * dtheta_deg;
    float3 dij_hat = vij / rij, dkj_hat = vkj / rkj;
    float3 dt_dvi = (ct * vij / rij - vkj / rkj) / (rij * st);
    float3 dt_dvk = (ct * vkj / rkj - vij / rij) / (rkj * st);
    float3 gi = dE_dth * dt_dvi + dE_drij * dij_hat;
    float3 gk = dE_dth * dt_dvk + dE_drkj * dkj_hat;
    AG(ai, gi);
    AG(aj, -gi - gk);
    AG(ak, gk);
}

// ================================================================
// 4. OUT-OF-PLANE BENDING
// ================================================================
for (uint o = 0u; o < noop; o++) {
    int a1 = idx_buf[io_i+o], a2 = idx_buf[io_j+o], a3 = idx_buf[io_k+o], a4 = idx_buf[io_l+o];
    float koop = param_buf[po_k + o];
    float3 u = GP(a1) - GP(a2);
    float3 v = GP(a3) - GP(a2);
    float3 w = GP(a4) - GP(a2);
    float3 nv = cross(u, v);
    float A = length(nv) + KEPS, B = length(w) + KEPS;
    float AB = A * B;
    float3 nh = nv / A, wh = w / B;
    float sinc = clamp(dot(nv, w) / AB, -1.0f, 1.0f);
    float chi = asin(sinc);
    float chi_d = chi * R2D;
    float cosc = cos(chi);
    cosc = max(cosc, KEPS);
    E += ABC * koop * 0.5f * chi_d * chi_d;
    float dE = ABC * koop * chi_d * R2D / cosc;
    float3 vxw = cross(v, w), wxu = cross(w, u);
    float3 vxn = cross(v, nh), nxu = cross(nh, u);
    float3 ds1 = vxw / AB - sinc * vxn / A;
    float3 ds3 = wxu / AB - sinc * nxu / A;
    float3 ds4 = (nh - sinc * wh) / B;
    float3 ds2 = -(ds1 + ds3 + ds4);
    AG(a1, dE * ds1);
    AG(a2, dE * ds2);
    AG(a3, dE * ds3);
    AG(a4, dE * ds4);
}

// ================================================================
// 5. TORSION
// ================================================================
for (uint t = 0u; t < ntor; t++) {
    int a1 = idx_buf[it_i+t], a2 = idx_buf[it_j+t], a3 = idx_buf[it_k+t], a4 = idx_buf[it_l+t];
    float V1 = param_buf[pt_V1+t], V2 = param_buf[pt_V2+t], V3 = param_buf[pt_V3+t];
    float3 p1 = GP(a1), p2 = GP(a2);
    float3 p3 = GP(a3), p4 = GP(a4);
    float3 b1 = p2 - p1, b2 = p3 - p2, b3 = p4 - p3;
    float3 c1 = cross(b1, b2), c2 = cross(b2, b3);
    float b2n = length(b2) + KEPS;
    float3 mv = cross(c1, b2 / b2n);
    float n1n2 = sqrt(dot(c1,c1) * dot(c2,c2) + KEPS);
    float sw = dot(mv, c2) / n1n2, cw = dot(c1, c2) / n1n2;
    float omega = atan2(sw, cw);
    E += 0.5f * (V1*(1.0f+cos(omega)) + V2*(1.0f-cos(2.0f*omega)) + V3*(1.0f+cos(3.0f*omega)));
    float dE = 0.5f * (-V1*sin(omega) + 2.0f*V2*sin(2.0f*omega) - 3.0f*V3*sin(3.0f*omega));
    float3 rij = p1 - p2, rkj = p3 - p2, rkl = p3 - p4;
    float3 m = cross(rij, rkj), n = cross(rkj, rkl);
    float msq = dot(m, m) + KEPS, nsq = dot(n, n) + KEPS;
    float rkjn = length(rkj) + KEPS, rkjn2 = rkjn * rkjn;
    float3 fi = (-rkjn / msq) * m * dE;
    float3 fl = ( rkjn / nsq) * n * dE;
    float pij = dot(rij, rkj) / rkjn2, pkl = dot(rkl, rkj) / rkjn2;
    float3 fj = -fi + pij * fi - pkl * fl;
    float3 fk = -fl - pij * fi + pkl * fl;
    AG(a1, fi);
    AG(a2, fj);
    AG(a3, fk);
    AG(a4, fl);
}

// ================================================================
// 6. VAN DER WAALS (Buffered 14-7)
// ================================================================
for (uint v = 0u; v < nvdw; v++) {
    int ai = idx_buf[iv_i + v], aj = idx_buf[iv_j + v];
    float Rstar = param_buf[pv_R + v], eps_v = param_buf[pv_e + v];
    float3 d = GP(ai) - GP(aj);
    float r = length(d) + KEPS;
    float3 dh = d / r;
    float rho = r / Rstar;
    float rho7 = rho*rho*rho*rho*rho*rho*rho;
    float t1 = 1.07f / (rho + 0.07f);
    float t2 = 1.12f / (rho7 + 0.12f) - 2.0f;
    float t1_7 = t1*t1*t1*t1*t1*t1*t1;
    E += eps_v * t1_7 * t2;
    float dt1 = -1.07f / ((rho+0.07f)*(rho+0.07f));
    float dt2 = -1.12f * 7.0f * rho*rho*rho*rho*rho*rho / ((rho7+0.12f)*(rho7+0.12f));
    float dE = eps_v * (7.0f * t1_7 / t1 * dt1 * t2 + t1_7 * dt2) / Rstar;
    float3 f = dE * dh;
    AG(ai, f);
    AG(aj, -f);
}

// ================================================================
// 7. ELECTROSTATIC
// ================================================================
for (uint e = 0u; e < nele; e++) {
    int ai = idx_buf[ie_i + e], aj = idx_buf[ie_j + e];
    float ct = param_buf[pe_ct + e], sc = param_buf[pe_sc + e];
    float3 d = GP(ai) - GP(aj);
    float r = length(d) + KEPS;
    float3 dh = d / r;
    float denom = r + ELD;
    E += sc * ELC * ct / denom;
    float dE = -sc * ELC * ct / (denom * denom);
    float3 f = dE * dh;
    AG(ai, f);
    AG(aj, -f);
}

energy_out[cid] = E;
"""

# ---------------------------------------------------------------------------
# Parameter packing
# ---------------------------------------------------------------------------


def pack_params_for_metal(
    p: MMFFParams,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Pack MMFF parameters into 3 flat Metal buffers.

    Returns:
        (idx_buf, param_buf, meta) — all as MLX arrays.
    """
    # Index buffer: concatenate all integer index arrays
    idx_arrays = [
        p.bond_idx1, p.bond_idx2,
        p.angle_idx1, p.angle_idx2, p.angle_idx3,
        p.strbend_idx1, p.strbend_idx2, p.strbend_idx3,
        p.oop_idx1, p.oop_idx2, p.oop_idx3, p.oop_idx4,
        p.torsion_idx1, p.torsion_idx2, p.torsion_idx3, p.torsion_idx4,
        p.vdw_idx1, p.vdw_idx2,
        p.ele_idx1, p.ele_idx2,
    ]
    idx_np = np.concatenate([a.astype(np.int32) for a in idx_arrays]) if any(
        len(a) > 0 for a in idx_arrays
    ) else np.zeros(1, dtype=np.int32)

    # Float buffer: concatenate all parameter arrays
    ele_is14 = p.ele_is_1_4.astype(np.float32)
    ele_scale = np.where(ele_is14 > 0.5, 0.75, 1.0).astype(np.float32)
    deg_to_rad = np.pi / 180.0

    param_arrays = [
        p.bond_kb, p.bond_r0,
        p.angle_ka, p.angle_theta0 * deg_to_rad,
        p.strbend_kba_ijk, p.strbend_kba_kji,
        p.strbend_r0_ij, p.strbend_r0_kj,
        p.strbend_theta0 * deg_to_rad,
        p.oop_koop,
        p.torsion_V1, p.torsion_V2, p.torsion_V3,
        p.vdw_R_star, p.vdw_eps,
        p.ele_charge_term, ele_scale,
    ]
    param_np = np.concatenate([a.astype(np.float32) for a in param_arrays]) if any(
        len(a) > 0 for a in param_arrays
    ) else np.zeros(1, dtype=np.float32)

    # Meta: sizes
    meta_np = np.array([
        p.n_atoms,
        len(p.bond_idx1),
        len(p.angle_idx1),
        len(p.strbend_idx1),
        len(p.oop_idx1),
        len(p.torsion_idx1),
        len(p.vdw_idx1),
        len(p.ele_idx1),
    ], dtype=np.uint32)

    return (
        mx.array(idx_np),
        mx.array(param_np),
        mx.array(meta_np),
    )


# ---------------------------------------------------------------------------
# Kernel cache and wrapper
# ---------------------------------------------------------------------------

_mmff_kernel = None


def _get_mmff_kernel():
    global _mmff_kernel
    if _mmff_kernel is None:
        _mmff_kernel = mx.fast.metal_kernel(
            name="mmff_energy_grad_fused",
            input_names=["pos", "idx_buf", "param_buf", "meta"],
            output_names=["energy_out", "grad_out"],
            source=_MMFF_KERNEL_SOURCE,
            ensure_row_contiguous=True,
        )
    return _mmff_kernel


def pack_multi_mol_for_metal(
    params_list: list[MMFFParams],
    conf_counts: list[int],
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, list[int]]:
    """
    Pack parameters for multiple molecules into a single mega-kernel call.

    Args:
        params_list: MMFFParams for each molecule.
        conf_counts: number of conformers per molecule.

    Returns:
        (idx_buf, param_buf, all_meta, dispatch, pos_offsets, dims_per_mol)
        where dims_per_mol[m] = n_atoms_m * 3.
    """
    deg_to_rad = np.pi / 180.0
    all_idx: list[np.ndarray] = []
    all_param: list[np.ndarray] = []
    meta_rows: list[np.ndarray] = []
    dispatch_rows: list[np.ndarray] = []
    pos_offset_list: list[int] = []
    dims_per_mol: list[int] = []

    idx_cursor = 0
    param_cursor = 0
    pos_cursor = 0

    for mol_idx, (p, C) in enumerate(zip(params_list, conf_counts)):
        meta_off = mol_idx * 8
        nb = len(p.bond_idx1)
        na = len(p.angle_idx1)
        nsb = len(p.strbend_idx1)
        noop = len(p.oop_idx1)
        ntor = len(p.torsion_idx1)
        nvdw = len(p.vdw_idx1)
        nele = len(p.ele_idx1)
        dim = p.n_atoms * 3
        dims_per_mol.append(dim)

        meta_rows.append(np.array(
            [p.n_atoms, nb, na, nsb, noop, ntor, nvdw, nele],
            dtype=np.uint32,
        ))

        idx_arrays = [
            p.bond_idx1, p.bond_idx2,
            p.angle_idx1, p.angle_idx2, p.angle_idx3,
            p.strbend_idx1, p.strbend_idx2, p.strbend_idx3,
            p.oop_idx1, p.oop_idx2, p.oop_idx3, p.oop_idx4,
            p.torsion_idx1, p.torsion_idx2, p.torsion_idx3, p.torsion_idx4,
            p.vdw_idx1, p.vdw_idx2,
            p.ele_idx1, p.ele_idx2,
        ]
        mol_idx_np = np.concatenate(
            [a.astype(np.int32) for a in idx_arrays]
        ) if any(len(a) > 0 for a in idx_arrays) else np.zeros(1, dtype=np.int32)

        ele_is14 = p.ele_is_1_4.astype(np.float32)
        ele_scale = np.where(ele_is14 > 0.5, 0.75, 1.0).astype(np.float32)
        param_arrays = [
            p.bond_kb, p.bond_r0,
            p.angle_ka, p.angle_theta0 * deg_to_rad,
            p.strbend_kba_ijk, p.strbend_kba_kji,
            p.strbend_r0_ij, p.strbend_r0_kj,
            p.strbend_theta0 * deg_to_rad,
            p.oop_koop,
            p.torsion_V1, p.torsion_V2, p.torsion_V3,
            p.vdw_R_star, p.vdw_eps,
            p.ele_charge_term, ele_scale,
        ]
        mol_param_np = np.concatenate(
            [a.astype(np.float32) for a in param_arrays]
        ) if any(len(a) > 0 for a in param_arrays) else np.zeros(1, dtype=np.float32)

        all_idx.append(mol_idx_np)
        all_param.append(mol_param_np)

        for _ in range(C):
            dispatch_rows.append(
                np.array([meta_off, idx_cursor, param_cursor], dtype=np.uint32)
            )
            pos_offset_list.append(pos_cursor)
            pos_cursor += dim

        idx_cursor += len(mol_idx_np)
        param_cursor += len(mol_param_np)

    idx_buf = mx.array(np.concatenate(all_idx))
    param_buf = mx.array(np.concatenate(all_param))
    all_meta = mx.array(np.concatenate(meta_rows))
    dispatch = mx.array(np.concatenate(dispatch_rows))
    pos_offsets = mx.array(np.array(pos_offset_list, dtype=np.uint32))

    return idx_buf, param_buf, all_meta, dispatch, pos_offsets, dims_per_mol


# ---------------------------------------------------------------------------
# Mega-kernel cache and wrapper
# ---------------------------------------------------------------------------

_mmff_mega_kernel = None


def _get_mmff_mega_kernel():
    global _mmff_mega_kernel
    if _mmff_mega_kernel is None:
        _mmff_mega_kernel = mx.fast.metal_kernel(
            name="mmff_mega_energy_grad",
            input_names=[
                "pos", "idx_buf", "param_buf", "all_meta",
                "dispatch", "pos_offsets",
            ],
            output_names=["energy_out", "grad_out"],
            source=_MMFF_MEGA_KERNEL_SOURCE,
            ensure_row_contiguous=True,
        )
    return _mmff_mega_kernel


def mmff_energy_grad_metal_mega(
    idx_buf: mx.array,
    param_buf: mx.array,
    all_meta: mx.array,
    dispatch: mx.array,
    pos_offsets: mx.array,
    pos_flat: mx.array,
    total_confs: int,
    total_coords: int,
) -> tuple[mx.array, mx.array]:
    """
    Compute MMFF94 energy + gradient for many molecules in one kernel launch.

    Args:
        idx_buf, param_buf, all_meta, dispatch, pos_offsets:
            from pack_multi_mol_for_metal().
        pos_flat: (total_coords,) float32 – all conformers' positions flat.
        total_confs: total number of conformers across all molecules.
        total_coords: total number of coordinate values (sum of n_atoms*3*C).

    Returns:
        (energies, gradients) — shapes (total_confs,) and (total_coords,).
    """
    kernel = _get_mmff_mega_kernel()
    energy_out, grad_out = kernel(
        inputs=[pos_flat, idx_buf, param_buf, all_meta, dispatch, pos_offsets],
        grid=(total_confs, 1, 1),
        threadgroup=(min(256, total_confs), 1, 1),
        output_shapes=[(total_confs,), (total_coords,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return energy_out, grad_out


def mmff_energy_grad_metal(
    idx_buf: mx.array,
    param_buf: mx.array,
    meta: mx.array,
    positions: mx.array,
) -> tuple[mx.array, mx.array]:
    """
    Compute MMFF94 energy + gradient via fused Metal kernel.

    Args:
        idx_buf, param_buf, meta: packed parameters from pack_params_for_metal().
        positions: (C, N, 3) float32 positions (will be flattened).

    Returns:
        (energies, gradients) — shapes (C,) and (C, N*3).
    """
    C, N, _ = positions.shape
    dim = N * 3
    pos_flat = mx.reshape(positions, (C * dim,))

    kernel = _get_mmff_kernel()
    energy_out, grad_out = kernel(
        inputs=[pos_flat, idx_buf, param_buf, meta],
        grid=(C, 1, 1),
        threadgroup=(min(256, C), 1, 1),
        output_shapes=[(C,), (C * dim,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return energy_out, mx.reshape(grad_out, (C, dim))


# ---------------------------------------------------------------------------
# Fused OPTIMIZER kernel: full optimisation in a single kernel launch
# ---------------------------------------------------------------------------
# Each thread copies positions to an output buffer, then runs max_iters
# iterations of steepest descent (energy → gradient → capped step) entirely
# on the GPU.  Zero Python overhead per iteration.
#
# Uses the output buffers as working storage:
#   pos_out     — working positions (read+write each iteration)
#   grad_scratch — gradient accumulator (zeroed + written each iteration)
#   energy_out  — final energy per conformer
#
# Works with the mega-kernel dispatch table for multi-molecule support.

_MMFF_OPT_KERNEL_SOURCE = """
const float MDYNE = 143.9325f;
const float CS = -2.0f;
const float CB = -0.007f;
const float ABC = 0.043844f;
const float SBC = 2.51210f;
const float ELC = 332.0716f;
const float ELD = 0.05f;
const float R2D = 57.295779513082323f;
const float KEPS = 1e-12f;

uint cid = thread_position_in_grid.x;

// Dispatch: molecule params for this conformer
uint mol_meta_off = dispatch[cid * 3u];
uint idx_base     = dispatch[cid * 3u + 1u];
uint param_base   = dispatch[cid * 3u + 2u];

uint N    = all_meta[mol_meta_off];
uint nb   = all_meta[mol_meta_off + 1u];
uint na   = all_meta[mol_meta_off + 2u];
uint nsb  = all_meta[mol_meta_off + 3u];
uint noop = all_meta[mol_meta_off + 4u];
uint ntor = all_meta[mol_meta_off + 5u];
uint nvdw = all_meta[mol_meta_off + 6u];
uint nele = all_meta[mol_meta_off + 7u];

uint dim = N * 3u;
uint pos_off = pos_offsets[cid];

uint max_it = opt_meta[0];
float max_step  = as_type<float>(opt_meta[1]);
float grad_lr   = as_type<float>(opt_meta[2]);

// Macros: read from pos_out (working positions), write gradient to grad_scratch
#define GP(atom) float3(pos_out[pos_off + uint(atom)*3u], pos_out[pos_off + uint(atom)*3u+1u], pos_out[pos_off + uint(atom)*3u+2u])
#define AG(atom, fv) { uint _b = pos_off + uint(atom)*3u; grad_scratch[_b] += (fv).x; grad_scratch[_b+1u] += (fv).y; grad_scratch[_b+2u] += (fv).z; }

// Copy input positions to working buffers
for (uint i = 0u; i < dim; i++) {
    pos_out[pos_off + i] = pos[pos_off + i];
    prev_pos[pos_off + i] = pos[pos_off + i];
    prev_grad[pos_off + i] = 0.0f;
}

float alpha = grad_lr;
float E_prev = 1e30f;

// ==== Index offsets (with base) ====
uint ib_i = idx_base,                           ib_j = idx_base + nb;
uint ia_i = idx_base + 2u*nb,                   ia_j = idx_base + 2u*nb + na,       ia_k = idx_base + 2u*nb + 2u*na;
uint is_i = idx_base + 2u*nb + 3u*na,           is_j = is_i + nsb,                  is_k = is_j + nsb;
uint io_i = is_k + nsb,                          io_j = io_i + noop,                 io_k = io_j + noop,  io_l = io_k + noop;
uint it_i = io_l + noop,                         it_j = it_i + ntor,                 it_k = it_j + ntor,  it_l = it_k + ntor;
uint iv_i = it_l + ntor,                         iv_j = iv_i + nvdw;
uint ie_i = iv_j + nvdw,                         ie_j = ie_i + nele;

// ==== Float offsets (with base) ====
uint pb_kb = param_base,                         pb_r0 = param_base + nb;
uint pa_ka = param_base + 2u*nb,                 pa_t0 = param_base + 2u*nb + na;
uint ps_kij = param_base + 2u*nb + 2u*na,       ps_kji = ps_kij+nsb, ps_rij = ps_kji+nsb, ps_rkj = ps_rij+nsb, ps_t0 = ps_rkj+nsb;
uint po_k  = ps_t0 + nsb;
uint pt_V1 = po_k + noop,                       pt_V2 = pt_V1+ntor, pt_V3 = pt_V2+ntor;
uint pv_R  = pt_V3 + ntor,                      pv_e = pv_R + nvdw;
uint pe_ct = pv_e + nvdw,                       pe_sc = pe_ct + nele;

// ================================================================
// OPTIMIZATION LOOP
// ================================================================
float E = 0.0f;

for (uint iter = 0u; iter < max_it; iter++) {

    // Zero gradient scratch
    for (uint i = 0u; i < dim; i++) {
        grad_scratch[pos_off + i] = 0.0f;
    }
    E = 0.0f;

    // ---- 1. BOND STRETCH ----
    for (uint b = 0u; b < nb; b++) {
        int ai = idx_buf[ib_i + b], aj = idx_buf[ib_j + b];
        float kb = param_buf[pb_kb + b], r0 = param_buf[pb_r0 + b];
        float3 d = GP(ai) - GP(aj);
        float r = length(d) + KEPS;
        float dr = r - r0;
        E += MDYNE * kb * 0.5f * dr * dr * (1.0f + CS*dr + (7.0f/12.0f)*CS*CS*dr*dr);
        float dE = MDYNE * kb * dr * (1.0f + 1.5f*CS*dr + (7.0f/6.0f)*CS*CS*dr*dr);
        float3 f = (dE / r) * d;
        AG(ai, f);
        AG(aj, -f);
    }

    // ---- 2. ANGLE BEND ----
    for (uint a = 0u; a < na; a++) {
        int ai = idx_buf[ia_i + a], aj = idx_buf[ia_j + a], ak = idx_buf[ia_k + a];
        float ka = param_buf[pa_ka + a], theta0 = param_buf[pa_t0 + a];
        float3 v1 = GP(ai) - GP(aj);
        float3 v2 = GP(ak) - GP(aj);
        float r1 = length(v1) + KEPS, r2 = length(v2) + KEPS;
        float ct = clamp(dot(v1, v2) / (r1 * r2), -1.0f + 1e-7f, 1.0f - 1e-7f);
        float theta = acos(ct);
        float st = sin(theta);
        st = max(st, KEPS);
        float dt = (theta - theta0) * R2D;
        E += ABC * ka * 0.5f * dt * dt * (1.0f + CB * dt);
        float dE = ABC * ka * dt * (1.0f + 1.5f * CB * dt) * R2D;
        float3 gi = dE * (ct * v1 / r1 - v2 / r2) / (r1 * st);
        float3 gk = dE * (ct * v2 / r2 - v1 / r1) / (r2 * st);
        AG(ai, gi);
        AG(aj, -gi - gk);
        AG(ak, gk);
    }

    // ---- 3. STRETCH-BEND ----
    for (uint s = 0u; s < nsb; s++) {
        int ai = idx_buf[is_i + s], aj = idx_buf[is_j + s], ak = idx_buf[is_k + s];
        float kij = param_buf[ps_kij + s], kji = param_buf[ps_kji + s];
        float r0ij = param_buf[ps_rij + s], r0kj = param_buf[ps_rkj + s];
        float theta0 = param_buf[ps_t0 + s];
        float3 vij = GP(ai) - GP(aj);
        float3 vkj = GP(ak) - GP(aj);
        float rij = length(vij) + KEPS, rkj = length(vkj) + KEPS;
        float ct = clamp(dot(vij, vkj) / (rij * rkj), -1.0f + 1e-7f, 1.0f - 1e-7f);
        float theta = acos(ct);
        float st = max(sin(theta), KEPS);
        float drij = rij - r0ij, drkj = rkj - r0kj;
        float dtheta_deg = (theta - theta0) * R2D;
        E += SBC * (kij * drij + kji * drkj) * dtheta_deg;
        float dE_dth = SBC * (kij * drij + kji * drkj) * R2D;
        float dE_drij = SBC * kij * dtheta_deg;
        float dE_drkj = SBC * kji * dtheta_deg;
        float3 dij_hat = vij / rij, dkj_hat = vkj / rkj;
        float3 dt_dvi = (ct * vij / rij - vkj / rkj) / (rij * st);
        float3 dt_dvk = (ct * vkj / rkj - vij / rij) / (rkj * st);
        float3 gi = dE_dth * dt_dvi + dE_drij * dij_hat;
        float3 gk = dE_dth * dt_dvk + dE_drkj * dkj_hat;
        AG(ai, gi);
        AG(aj, -gi - gk);
        AG(ak, gk);
    }

    // ---- 4. OUT-OF-PLANE BENDING ----
    for (uint o = 0u; o < noop; o++) {
        int a1 = idx_buf[io_i+o], a2 = idx_buf[io_j+o], a3 = idx_buf[io_k+o], a4 = idx_buf[io_l+o];
        float koop = param_buf[po_k + o];
        float3 u = GP(a1) - GP(a2);
        float3 v = GP(a3) - GP(a2);
        float3 w = GP(a4) - GP(a2);
        float3 nv = cross(u, v);
        float A = length(nv) + KEPS, B = length(w) + KEPS;
        float AB = A * B;
        float3 nh = nv / A, wh = w / B;
        float sinc = clamp(dot(nv, w) / AB, -1.0f, 1.0f);
        float chi = asin(sinc);
        float chi_d = chi * R2D;
        float cosc = cos(chi);
        cosc = max(cosc, KEPS);
        E += ABC * koop * 0.5f * chi_d * chi_d;
        float dE = ABC * koop * chi_d * R2D / cosc;
        float3 vxw = cross(v, w), wxu = cross(w, u);
        float3 vxn = cross(v, nh), nxu = cross(nh, u);
        float3 ds1 = vxw / AB - sinc * vxn / A;
        float3 ds3 = wxu / AB - sinc * nxu / A;
        float3 ds4 = (nh - sinc * wh) / B;
        float3 ds2 = -(ds1 + ds3 + ds4);
        AG(a1, dE * ds1);
        AG(a2, dE * ds2);
        AG(a3, dE * ds3);
        AG(a4, dE * ds4);
    }

    // ---- 5. TORSION ----
    for (uint t = 0u; t < ntor; t++) {
        int a1 = idx_buf[it_i+t], a2 = idx_buf[it_j+t], a3 = idx_buf[it_k+t], a4 = idx_buf[it_l+t];
        float V1 = param_buf[pt_V1+t], V2 = param_buf[pt_V2+t], V3 = param_buf[pt_V3+t];
        float3 p1 = GP(a1), p2 = GP(a2);
        float3 p3 = GP(a3), p4 = GP(a4);
        float3 b1 = p2 - p1, b2 = p3 - p2, b3 = p4 - p3;
        float3 c1 = cross(b1, b2), c2 = cross(b2, b3);
        float b2n = length(b2) + KEPS;
        float3 mv = cross(c1, b2 / b2n);
        float n1n2 = sqrt(dot(c1,c1) * dot(c2,c2) + KEPS);
        float sw = dot(mv, c2) / n1n2, cw = dot(c1, c2) / n1n2;
        float omega = atan2(sw, cw);
        E += 0.5f * (V1*(1.0f+cos(omega)) + V2*(1.0f-cos(2.0f*omega)) + V3*(1.0f+cos(3.0f*omega)));
        float dE = 0.5f * (-V1*sin(omega) + 2.0f*V2*sin(2.0f*omega) - 3.0f*V3*sin(3.0f*omega));
        float3 rij = p1 - p2, rkj = p3 - p2, rkl = p3 - p4;
        float3 m = cross(rij, rkj), n = cross(rkj, rkl);
        float msq = dot(m, m) + KEPS, nsq = dot(n, n) + KEPS;
        float rkjn = length(rkj) + KEPS, rkjn2 = rkjn * rkjn;
        float3 fi = (-rkjn / msq) * m * dE;
        float3 fl = ( rkjn / nsq) * n * dE;
        float pij = dot(rij, rkj) / rkjn2, pkl = dot(rkl, rkj) / rkjn2;
        float3 fj = -fi + pij * fi - pkl * fl;
        float3 fk = -fl - pij * fi + pkl * fl;
        AG(a1, fi);
        AG(a2, fj);
        AG(a3, fk);
        AG(a4, fl);
    }

    // ---- 6. VAN DER WAALS ----
    for (uint v = 0u; v < nvdw; v++) {
        int ai = idx_buf[iv_i + v], aj = idx_buf[iv_j + v];
        float Rstar = param_buf[pv_R + v], eps_v = param_buf[pv_e + v];
        float3 d = GP(ai) - GP(aj);
        float r = length(d) + KEPS;
        float3 dh = d / r;
        float rho = r / Rstar;
        float rho7 = rho*rho*rho*rho*rho*rho*rho;
        float t1 = 1.07f / (rho + 0.07f);
        float t2 = 1.12f / (rho7 + 0.12f) - 2.0f;
        float t1_7 = t1*t1*t1*t1*t1*t1*t1;
        E += eps_v * t1_7 * t2;
        float dt1 = -1.07f / ((rho+0.07f)*(rho+0.07f));
        float dt2 = -1.12f * 7.0f * rho*rho*rho*rho*rho*rho / ((rho7+0.12f)*(rho7+0.12f));
        float dE = eps_v * (7.0f * t1_7 / t1 * dt1 * t2 + t1_7 * dt2) / Rstar;
        float3 f = dE * dh;
        AG(ai, f);
        AG(aj, -f);
    }

    // ---- 7. ELECTROSTATIC ----
    for (uint e = 0u; e < nele; e++) {
        int ai = idx_buf[ie_i + e], aj = idx_buf[ie_j + e];
        float ct = param_buf[pe_ct + e], sc = param_buf[pe_sc + e];
        float3 d = GP(ai) - GP(aj);
        float r = length(d) + KEPS;
        float3 dh = d / r;
        float denom = r + ELD;
        E += sc * ELC * ct / denom;
        float dE = -sc * ELC * ct / (denom * denom);
        float3 f = dE * dh;
        AG(ai, f);
        AG(aj, -f);
    }

    // ---- ENERGY GUARD: revert on energy increase ----
    if (iter > 0u && E > E_prev + 1.0f) {
        for (uint i = 0u; i < dim; i++)
            pos_out[pos_off + i] = prev_pos[pos_off + i];
        alpha *= 0.5f;
        alpha = max(alpha, 1e-7f);
        E = E_prev;
        continue;
    }

    // ---- BARZILAI-BORWEIN step size (iter > 0) ----
    if (iter > 0u) {
        float ss = 0.0f, sy = 0.0f;
        for (uint i = 0u; i < dim; i++) {
            float si = pos_out[pos_off + i] - prev_pos[pos_off + i];
            float yi = grad_scratch[pos_off + i] - prev_grad[pos_off + i];
            ss += si * si;
            sy += si * yi;
        }
        if (sy > 1e-20f) {
            alpha = clamp(ss / sy, 1e-6f, 0.3f);
        }
    }

    E_prev = E;

    // Save state for next BB computation
    for (uint i = 0u; i < dim; i++) {
        prev_pos[pos_off + i] = pos_out[pos_off + i];
        prev_grad[pos_off + i] = grad_scratch[pos_off + i];
    }

    // ---- GRADIENT STEP with per-atom capping ----
    for (uint a = 0u; a < N; a++) {
        uint b = pos_off + a * 3u;
        float gx = grad_scratch[b], gy = grad_scratch[b+1u], gz = grad_scratch[b+2u];
        float gn = sqrt(gx*gx + gy*gy + gz*gz + 1e-30f);
        float s = min(max_step / gn, alpha);
        pos_out[b]     -= s * gx;
        pos_out[b+1u]  -= s * gy;
        pos_out[b+2u]  -= s * gz;
    }

}  // end optimization loop

energy_out[cid] = E;
"""

# ---------------------------------------------------------------------------
# Fused optimizer kernel cache and wrapper
# ---------------------------------------------------------------------------

_mmff_opt_kernel = None


def _get_mmff_opt_kernel():
    global _mmff_opt_kernel
    if _mmff_opt_kernel is None:
        _mmff_opt_kernel = mx.fast.metal_kernel(
            name="mmff_fused_optimizer",
            input_names=[
                "pos", "idx_buf", "param_buf", "all_meta",
                "dispatch", "pos_offsets", "opt_meta",
            ],
            output_names=[
                "pos_out", "energy_out", "grad_scratch",
                "prev_pos", "prev_grad",
            ],
            source=_MMFF_OPT_KERNEL_SOURCE,
            ensure_row_contiguous=True,
        )
    return _mmff_opt_kernel


def mmff_optimize_metal_fused(
    idx_buf: mx.array,
    param_buf: mx.array,
    all_meta: mx.array,
    dispatch: mx.array,
    pos_offsets: mx.array,
    pos_flat: mx.array,
    total_confs: int,
    total_coords: int,
    max_iters: int = 200,
    max_step: float = 0.3,
    grad_lr: float = 0.001,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Full MMFF optimisation in a single Metal kernel launch.

    Uses Barzilai-Borwein spectral step size for near-L-BFGS convergence
    with zero Python loop overhead.

    Returns:
        (final_positions, final_energies, final_gradients) — flat arrays.
    """
    import struct

    opt_meta_np = np.array([
        max_iters,
        struct.unpack("I", struct.pack("f", max_step))[0],
        struct.unpack("I", struct.pack("f", grad_lr))[0],
    ], dtype=np.uint32)
    opt_meta = mx.array(opt_meta_np)

    kernel = _get_mmff_opt_kernel()
    pos_out, energy_out, grad_scratch, _, _ = kernel(
        inputs=[
            pos_flat, idx_buf, param_buf, all_meta,
            dispatch, pos_offsets, opt_meta,
        ],
        grid=(total_confs, 1, 1),
        threadgroup=(min(256, total_confs), 1, 1),
        output_shapes=[
            (total_coords,), (total_confs,), (total_coords,),
            (total_coords,), (total_coords,),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32,
                       mx.float32, mx.float32],
    )
    return pos_out, energy_out, grad_scratch
