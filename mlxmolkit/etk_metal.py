"""
N×k parallel ETK (3D torsion) minimization with shared constraints.

Same TPM=32 threadgroup architecture as conformer_metal.py (DG stage),
but with ETK energy terms: CSD torsion preferences (6-term Fourier),
improper torsion (planarity), and 1-4 distance constraints.

Constraint indices are LOCAL [0, n_atoms_mol). The kernel pre-adds
``atom_off = conf_atom_starts[conf_idx]`` before calling helpers.
Helpers operate on GLOBAL indices and stay unchanged from shivampatel10.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import mlx.core as mx

from .shared_batch import SharedConstraintBatch

DEFAULT_TPM = 32
DEFAULT_LBFGS_M = 8

# ---------------------------------------------------------------------------
# ETK MSL helpers (from shivampatel10, unchanged — operate on global indices)
# ---------------------------------------------------------------------------

_ETK_HEADER = """
constant float TOLX = 1.2e-6f;
constant float FUNCTOL = 1e-4f;
constant float MOVETOL = 1e-6f;
constant float MAX_STEP_FACTOR = 100.0f;
constant int MAX_LS_ITERS = 1000;

// ---- Torsion cos(phi) ----
inline float calc_cos_phi(const device float* pos, int i1, int i2, int i3, int i4, int dim) {
    float r1[3], r2[3], r3[3], r4[3];
    for (int d=0;d<3;d++) {
        r1[d] = pos[i1*dim+d] - pos[i2*dim+d];
        r2[d] = pos[i3*dim+d] - pos[i2*dim+d];
        r3[d] = -r2[d];
        r4[d] = pos[i4*dim+d] - pos[i3*dim+d];
    }
    float t1x=r1[1]*r2[2]-r1[2]*r2[1], t1y=r1[2]*r2[0]-r1[0]*r2[2], t1z=r1[0]*r2[1]-r1[1]*r2[0];
    float t2x=r3[1]*r4[2]-r3[2]*r4[1], t2y=r3[2]*r4[0]-r3[0]*r4[2], t2z=r3[0]*r4[1]-r3[1]*r4[0];
    float comb = (t1x*t1x+t1y*t1y+t1z*t1z)*(t2x*t2x+t2y*t2y+t2z*t2z);
    if (comb < 1e-16f) return 0.0f;
    return clamp((t1x*t2x+t1y*t2y+t1z*t2z)*rsqrt(comb), -1.0f, 1.0f);
}

// ---- 6-term Fourier torsion energy (Chebyshev recurrence) ----
inline float torsion_e(float c,
    float V0,float V1,float V2,float V3,float V4,float V5,
    float s0,float s1,float s2,float s3,float s4,float s5
) {
    float c2=c*c, c3=c*c2, c4=c*c3, c5=c*c4, c6=c*c5;
    return V0*(1.0f+s0*c) + V1*(1.0f+s1*(2.0f*c2-1.0f))
         + V2*(1.0f+s2*(4.0f*c3-3.0f*c)) + V3*(1.0f+s3*(8.0f*c4-8.0f*c2+1.0f))
         + V4*(1.0f+s4*(16.0f*c5-20.0f*c3+5.0f*c))
         + V5*(1.0f+s5*(32.0f*c6-48.0f*c4+18.0f*c2-1.0f));
}

// ---- Torsion gradient (full 4-atom, from shivampatel10) ----
inline void torsion_g(const device float* pos, device float* grad,
    int i1,int i2,int i3,int i4,
    float V0,float V1,float V2,float V3,float V4,float V5,
    float s0,float s1,float s2,float s3,float s4,float s5, int dim
) {
    float r1[3],r2[3],r3[3],r4[3];
    for (int d=0;d<3;d++) {
        r1[d]=pos[i1*dim+d]-pos[i2*dim+d]; r2[d]=pos[i3*dim+d]-pos[i2*dim+d];
        r3[d]=-r2[d]; r4[d]=pos[i4*dim+d]-pos[i3*dim+d];
    }
    float t0x=r1[1]*r2[2]-r1[2]*r2[1],t0y=r1[2]*r2[0]-r1[0]*r2[2],t0z=r1[0]*r2[1]-r1[1]*r2[0];
    float t1x=r3[1]*r4[2]-r3[2]*r4[1],t1y=r3[2]*r4[0]-r3[0]*r4[2],t1z=r3[0]*r4[1]-r3[1]*r4[0];
    float d02=t0x*t0x+t0y*t0y+t0z*t0z, d12=t1x*t1x+t1y*t1y+t1z*t1z;
    if (d02<1e-16f||d12<1e-16f) return;
    float inv0=rsqrt(max(d02,1e-16f)),inv1=rsqrt(max(d12,1e-16f));
    float tnx0=t0x*inv0,tny0=t0y*inv0,tnz0=t0z*inv0;
    float tnx1=t1x*inv1,tny1=t1y*inv1,tnz1=t1z*inv1;
    float cp=clamp(tnx0*tnx1+tny0*tny1+tnz0*tnz1,-1.0f,1.0f);
    float sp2=1.0f-cp*cp, sp=sqrt(max(sp2,0.0f));
    float c=cp,c2=c*c,c3=c*c2,c4=c*c3;
    float dE=-s0*V0*sp-2.0f*s1*V1*(2.0f*c*sp)-3.0f*s2*V2*(4.0f*c2*sp-sp)
        -4.0f*s3*V3*(8.0f*c3*sp-4.0f*c*sp)-5.0f*s4*V4*(16.0f*c4*sp-12.0f*c2*sp+sp)
        -6.0f*s5*V5*(32.0f*c4*c*sp-32.0f*c3*sp+6.0f*sp);
    float st;
    if (abs(sp)>1e-8f) st=-dE/sp; else st=-dE/max(abs(cp),1e-16f)*sign(cp+1e-30f);
    float dcx0=inv0*(tnx1-cp*tnx0),dcy0=inv0*(tny1-cp*tny0),dcz0=inv0*(tnz1-cp*tnz0);
    float dcx1=inv1*(tnx0-cp*tnx1),dcy1=inv1*(tny0-cp*tny1),dcz1=inv1*(tnz0-cp*tnz1);
    float g1x=st*(dcz0*r2[1]-dcy0*r2[2]),g1y=st*(dcx0*r2[2]-dcz0*r2[0]),g1z=st*(dcy0*r2[0]-dcx0*r2[1]);
    float g4x=st*(dcy1*r3[2]-dcz1*r3[1]),g4y=st*(dcz1*r3[0]-dcx1*r3[2]),g4z=st*(dcx1*r3[1]-dcy1*r3[0]);
    float g2x=st*(dcy0*(r2[2]-r1[2])+dcz0*(r1[1]-r2[1])+dcy1*(-r4[2])+dcz1*r4[1]);
    float g2y=st*(dcx0*(r1[2]-r2[2])+dcz0*(r2[0]-r1[0])+dcx1*r4[2]+dcz1*(-r4[0]));
    float g2z=st*(dcx0*(r2[1]-r1[1])+dcy0*(r1[0]-r2[0])+dcx1*(-r4[1])+dcy1*r4[0]);
    float g3x=st*(dcy0*r1[2]+dcz0*(-r1[1])+dcy1*(r4[2]-r3[2])+dcz1*(r3[1]-r4[1]));
    float g3y=st*(dcx0*(-r1[2])+dcz0*r1[0]+dcx1*(r3[2]-r4[2])+dcz1*(r4[0]-r3[0]));
    float g3z=st*(dcx0*r1[1]+dcy0*(-r1[0])+dcx1*(r4[1]-r3[1])+dcy1*(r3[0]-r4[0]));
    grad[i1*dim+0]+=g1x;grad[i1*dim+1]+=g1y;grad[i1*dim+2]+=g1z;
    grad[i2*dim+0]+=g2x;grad[i2*dim+1]+=g2y;grad[i2*dim+2]+=g2z;
    grad[i3*dim+0]+=g3x;grad[i3*dim+1]+=g3y;grad[i3*dim+2]+=g3z;
    grad[i4*dim+0]+=g4x;grad[i4*dim+1]+=g4y;grad[i4*dim+2]+=g4z;
}

// ---- Improper torsion energy: E = w * (1 - cos(2ω)) ----
inline float improper_e(const device float* pos,
    int ic,int i0,int i1,int i2, float wt, int dim
) {
    float b1[3],b2[3],b3[3];
    for (int d=0;d<3;d++){b1[d]=pos[i0*dim+d]-pos[ic*dim+d];b2[d]=pos[i1*dim+d]-pos[i0*dim+d];b3[d]=pos[i2*dim+d]-pos[i1*dim+d];}
    float n1x=b1[1]*b2[2]-b1[2]*b2[1],n1y=b1[2]*b2[0]-b1[0]*b2[2],n1z=b1[0]*b2[1]-b1[1]*b2[0];
    float n2x=b2[1]*b3[2]-b2[2]*b3[1],n2y=b2[2]*b3[0]-b2[0]*b3[2],n2z=b2[0]*b3[1]-b2[1]*b3[0];
    float n1l=sqrt(n1x*n1x+n1y*n1y+n1z*n1z+1e-12f);
    float n2l=sqrt(n2x*n2x+n2y*n2y+n2z*n2z+1e-12f);
    float b2l=sqrt(b2[0]*b2[0]+b2[1]*b2[1]+b2[2]*b2[2]+1e-12f);
    float cw=(n1x*n2x+n1y*n2y+n1z*n2z)/(n1l*n2l);
    cw=clamp(cw,-1.0f,1.0f);
    float bh0=b2[0]/b2l,bh1=b2[1]/b2l,bh2=b2[2]/b2l;
    float m1x=n1y*bh2-n1z*bh1,m1y=n1z*bh0-n1x*bh2,m1z=n1x*bh1-n1y*bh0;
    float sw=(m1x*n2x+m1y*n2y+m1z*n2z)/(n1l*n2l);
    float omega=atan2(sw,cw);
    return wt*(1.0f-cos(2.0f*omega));
}

// ---- Improper torsion gradient ----
inline void improper_g(const device float* pos, device float* grad,
    int ic,int i0,int i1,int i2, float wt, int dim
) {
    float b1[3],b2[3],b3[3];
    for (int d=0;d<3;d++){b1[d]=pos[i0*dim+d]-pos[ic*dim+d];b2[d]=pos[i1*dim+d]-pos[i0*dim+d];b3[d]=pos[i2*dim+d]-pos[i1*dim+d];}
    float n1x=b1[1]*b2[2]-b1[2]*b2[1],n1y=b1[2]*b2[0]-b1[0]*b2[2],n1z=b1[0]*b2[1]-b1[1]*b2[0];
    float n2x=b2[1]*b3[2]-b2[2]*b3[1],n2y=b2[2]*b3[0]-b2[0]*b3[2],n2z=b2[0]*b3[1]-b2[1]*b3[0];
    float n1l=sqrt(n1x*n1x+n1y*n1y+n1z*n1z+1e-12f);
    float n2l=sqrt(n2x*n2x+n2y*n2y+n2z*n2z+1e-12f);
    float b2l=sqrt(b2[0]*b2[0]+b2[1]*b2[1]+b2[2]*b2[2]+1e-12f);
    float cw=(n1x*n2x+n1y*n2y+n1z*n2z)/(n1l*n2l); cw=clamp(cw,-1.0f,1.0f);
    float bh0=b2[0]/b2l,bh1=b2[1]/b2l,bh2=b2[2]/b2l;
    float m1x=n1y*bh2-n1z*bh1,m1y=n1z*bh0-n1x*bh2,m1z=n1x*bh1-n1y*bh0;
    float sw=(m1x*n2x+m1y*n2y+m1z*n2z)/(n1l*n2l);
    float omega=atan2(sw,cw);
    float dEdw=wt*2.0f*sin(2.0f*omega);
    float n1sq=n1x*n1x+n1y*n1y+n1z*n1z+1e-12f,n2sq=n2x*n2x+n2y*n2y+n2z*n2z+1e-12f;
    float b2sq=b2[0]*b2[0]+b2[1]*b2[1]+b2[2]*b2[2]+1e-12f;
    float f0=-b2l/n1sq,f3=b2l/n2sq;
    float b1b2=b1[0]*b2[0]+b1[1]*b2[1]+b1[2]*b2[2];
    float b3b2=b3[0]*b2[0]+b3[1]*b2[1]+b3[2]*b2[2];
    float f1a=b1b2/b2sq-1.0f,f1b=-b3b2/b2sq,f2a=b3b2/b2sq-1.0f,f2b=-b1b2/b2sq;
    for (int d=0;d<3;d++){
        float dp0=f0*(d==0?n1x:d==1?n1y:n1z);
        float dp3=f3*(d==0?n2x:d==1?n2y:n2z);
        float dp1=f1a*dp0+f1b*dp3,dp2=f2a*dp3+f2b*dp0;
        grad[ic*dim+d]+=dEdw*dp0; grad[i0*dim+d]+=dEdw*dp1;
        grad[i1*dim+d]+=dEdw*dp2; grad[i2*dim+d]+=dEdw*dp3;
    }
}

// ---- 1-4 distance constraint: flat-bottom harmonic ----
inline float dist14_e(const device float* pos, int a, int b,
    float lb, float ub, float wt, int dim
) {
    float d2=0.0f;
    for (int d=0;d<3;d++){float df=pos[a*dim+d]-pos[b*dim+d]; d2+=df*df;}
    float dist=sqrt(d2+1e-12f);
    if (dist<lb){float v=dist-lb; return wt*v*v;}
    if (dist>ub){float v=dist-ub; return wt*v*v;}
    return 0.0f;
}

inline void dist14_g(const device float* pos, device float* grad,
    int a, int b, float lb, float ub, float wt, int dim
) {
    float df[3]; float d2=0.0f;
    for (int d=0;d<3;d++){df[d]=pos[a*dim+d]-pos[b*dim+d]; d2+=df[d]*df[d];}
    float dist=sqrt(d2+1e-12f);
    float pf=0.0f;
    if (dist<lb) pf=wt*2.0f*(dist-lb)/dist;
    else if (dist>ub) pf=wt*2.0f*(dist-ub)/dist;
    if (pf!=0.0f) {
        for (int d=0;d<3;d++){float g=pf*df[d]; grad[a*dim+d]+=g; grad[b*dim+d]-=g;}
    }
}

// ---- Threadgroup primitives (same as DG kernel) ----
inline float tg_reduce_sum(threadgroup float* s, uint tid, uint n) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride=n/2;stride>0;stride>>=1){if(tid<stride)s[tid]+=s[tid+stride];threadgroup_barrier(mem_flags::mem_threadgroup);}
    float r=s[0]; threadgroup_barrier(mem_flags::mem_threadgroup); return r;
}
inline float tg_reduce_max(threadgroup float* s, uint tid, uint n) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride=n/2;stride>0;stride>>=1){if(tid<stride)s[tid]=max(s[tid],s[tid+stride]);threadgroup_barrier(mem_flags::mem_threadgroup);}
    float r=s[0]; threadgroup_barrier(mem_flags::mem_threadgroup); return r;
}
inline float parallel_dot(const device float* a, const device float* b, int n, uint tid, uint tpm, threadgroup float* s) {
    float sum=0.0f; for (int i=(int)tid;i<n;i+=(int)tpm) sum+=a[i]*b[i]; s[tid]=sum; return tg_reduce_sum(s,tid,tpm);
}
inline void parallel_saxpy(device float* a, float alpha, const device float* b, int n, uint tid, uint tpm) {
    for (int i=(int)tid;i<n;i+=(int)tpm) a[i]+=alpha*b[i]; threadgroup_barrier(mem_flags::mem_device);
}
inline void parallel_scale(device float* a, float alpha, int n, uint tid, uint tpm) {
    for (int i=(int)tid;i<n;i+=(int)tpm) a[i]*=alpha; threadgroup_barrier(mem_flags::mem_device);
}
inline void parallel_copy(device float* d, const device float* s, int n, uint tid, uint tpm) {
    for (int i=(int)tid;i<n;i+=(int)tpm) d[i]=s[i]; threadgroup_barrier(mem_flags::mem_device);
}
inline void parallel_set(device float* a, float v, int n, uint tid, uint tpm) {
    for (int i=(int)tid;i<n;i+=(int)tpm) a[i]=v; threadgroup_barrier(mem_flags::mem_device);
}
inline void parallel_neg_copy(device float* d, const device float* s, int n, uint tid, uint tpm) {
    for (int i=(int)tid;i<n;i+=(int)tpm) d[i]=-s[i]; threadgroup_barrier(mem_flags::mem_device);
}
"""

# ---------------------------------------------------------------------------
# ETK kernel body — conf_to_mol + atom_off adaptation
# Pre-adds atom_off to LOCAL constraint indices before calling helpers
# ---------------------------------------------------------------------------

_ETK_BODY = """
    uint tid = thread_position_in_threadgroup.x;
    uint conf_idx = threadgroup_position_in_grid.x;
    const uint tpm = TPM;
    const int lbfgs_m = LBFGS_M;
    threadgroup float shared[TPM];

    int n_confs_cfg = (int)config[0];
    int max_iters = (int)config[1];
    float grad_tol_v = config[2];
    int dim = (int)config[3];
    int total_pos_size = (int)config[4];

    if ((int)conf_idx >= n_confs_cfg) return;

    // Shared constraint indirection
    int mol_idx = conf_to_mol[conf_idx];
    int atom_off = conf_atom_starts[conf_idx];
    int n_atoms = mol_n_atoms[mol_idx];
    int n_vars = n_atoms * dim;

    // Constraint ranges (per molecule — SHARED)
    int tor_s = torsion_starts[mol_idx], tor_e = torsion_starts[mol_idx+1];
    int imp_s = improper_starts[mol_idx], imp_e = improper_starts[mol_idx+1];
    int d14_s = dist14_starts[mol_idx], d14_e = dist14_starts[mol_idx+1];

    int lbfgs_start = lbfgs_history_starts[conf_idx];

    parallel_copy(&out_pos[atom_off*dim], &pos[atom_off*dim], n_vars, tid, tpm);

    device float* my_pos = &out_pos[atom_off*dim];
    device float* my_grad = &work_grad[atom_off*dim];
    device float* my_dir = &work_dir[atom_off*dim];
    device float* my_old_pos = &work_scratch[atom_off*dim];
    device float* my_old_grad = &work_scratch[total_pos_size + atom_off*dim];
    device float* my_q = &work_scratch[2*total_pos_size + atom_off*dim];
    device float* my_S = &work_lbfgs[lbfgs_start];
    device float* my_Y = &work_lbfgs[lbfgs_start + lbfgs_m*n_vars];
    device float* my_rho = &work_rho[conf_idx * lbfgs_m];

    // ---- Energy (parallel) + gradient (thread 0) ----
    parallel_set(my_grad, 0.0f, n_vars, tid, tpm);
    float local_e = 0.0f;

    // Torsion energy (pre-add atom_off to LOCAL indices)
    for (int t=tor_s+(int)tid; t<tor_e; t+=(int)tpm) {
        int a1=torsion_quads[t*4]+atom_off, a2=torsion_quads[t*4+1]+atom_off;
        int a3=torsion_quads[t*4+2]+atom_off, a4=torsion_quads[t*4+3]+atom_off;
        float cp=calc_cos_phi(out_pos, a1,a2,a3,a4, dim);
        local_e += torsion_e(cp, torsion_V[t*6],torsion_V[t*6+1],torsion_V[t*6+2],
            torsion_V[t*6+3],torsion_V[t*6+4],torsion_V[t*6+5],
            torsion_signs_arr[t*6],torsion_signs_arr[t*6+1],torsion_signs_arr[t*6+2],
            torsion_signs_arr[t*6+3],torsion_signs_arr[t*6+4],torsion_signs_arr[t*6+5]);
    }
    // Improper energy
    for (int t=imp_s+(int)tid; t<imp_e; t+=(int)tpm) {
        int ic=improper_quads[t*4]+atom_off, i0=improper_quads[t*4+1]+atom_off;
        int i1=improper_quads[t*4+2]+atom_off, i2=improper_quads[t*4+3]+atom_off;
        local_e += improper_e(out_pos, ic,i0,i1,i2, improper_w[t], dim);
    }
    // 1-4 distance energy
    for (int t=d14_s+(int)tid; t<d14_e; t+=(int)tpm) {
        int a=d14_pairs[t*2]+atom_off, b=d14_pairs[t*2+1]+atom_off;
        local_e += dist14_e(out_pos, a,b, d14_bounds[t*3],d14_bounds[t*3+1],d14_bounds[t*3+2], dim);
    }
    shared[tid] = local_e;
    float energy = tg_reduce_sum(shared, tid, tpm);

    // Gradient (thread 0, serial)
    if (tid == 0) {
        for (int t=tor_s;t<tor_e;t++) {
            int a1=torsion_quads[t*4]+atom_off,a2=torsion_quads[t*4+1]+atom_off;
            int a3=torsion_quads[t*4+2]+atom_off,a4=torsion_quads[t*4+3]+atom_off;
            torsion_g(out_pos,work_grad,a1,a2,a3,a4,
                torsion_V[t*6],torsion_V[t*6+1],torsion_V[t*6+2],
                torsion_V[t*6+3],torsion_V[t*6+4],torsion_V[t*6+5],
                torsion_signs_arr[t*6],torsion_signs_arr[t*6+1],torsion_signs_arr[t*6+2],
                torsion_signs_arr[t*6+3],torsion_signs_arr[t*6+4],torsion_signs_arr[t*6+5],dim);
        }
        for (int t=imp_s;t<imp_e;t++) {
            int ic=improper_quads[t*4]+atom_off,i0=improper_quads[t*4+1]+atom_off;
            int i1=improper_quads[t*4+2]+atom_off,i2=improper_quads[t*4+3]+atom_off;
            improper_g(out_pos,work_grad,ic,i0,i1,i2,improper_w[t],dim);
        }
        for (int t=d14_s;t<d14_e;t++) {
            int a=d14_pairs[t*2]+atom_off,b=d14_pairs[t*2+1]+atom_off;
            dist14_g(out_pos,work_grad,a,b,d14_bounds[t*3],d14_bounds[t*3+1],d14_bounds[t*3+2],dim);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ---- L-BFGS loop (identical to DG kernel) ----
    parallel_neg_copy(my_dir, my_grad, n_vars, tid, tpm);
    float lss=0.0f; for (int i=(int)tid;i<n_vars;i+=(int)tpm) lss+=my_pos[i]*my_pos[i];
    shared[tid]=lss; float max_step=MAX_STEP_FACTOR*max(sqrt(tg_reduce_sum(shared,tid,tpm)),(float)n_vars);
    int status=1, hist_count=0, hist_idx=0;

    for (int iter=0; iter<max_iters && status==1; iter++) {
        parallel_copy(my_old_pos, my_pos, n_vars, tid, tpm);
        float old_energy = energy;

        float ld2=0.0f; for (int i=(int)tid;i<n_vars;i+=(int)tpm) ld2+=my_dir[i]*my_dir[i];
        shared[tid]=ld2; float dn=sqrt(tg_reduce_sum(shared,tid,tpm));
        if (dn>max_step) parallel_scale(my_dir,max_step/dn,n_vars,tid,tpm);
        float slope=parallel_dot(my_dir,my_grad,n_vars,tid,tpm,shared);
        float ltm=0.0f; for (int i=(int)tid;i<n_vars;i+=(int)tpm){float t=abs(my_dir[i])/max(abs(my_pos[i]),1.0f);if(t>ltm)ltm=t;}
        shared[tid]=ltm; float lmin=MOVETOL/max(tg_reduce_max(shared,tid,tpm),1e-30f);

        float lam=1.0f,prev_lam=1.0f,prev_e=old_energy; bool ls_done=false;
        for (int ls=0; ls<MAX_LS_ITERS && !ls_done; ls++) {
            if (lam<lmin){parallel_copy(my_pos,my_old_pos,n_vars,tid,tpm);ls_done=true;break;}
            for (int i=(int)tid;i<n_vars;i+=(int)tpm) my_pos[i]=my_old_pos[i]+lam*my_dir[i];
            threadgroup_barrier(mem_flags::mem_device);

            // Trial energy (parallel)
            float lte=0.0f;
            for (int t=tor_s+(int)tid;t<tor_e;t+=(int)tpm){
                int a1=torsion_quads[t*4]+atom_off,a2=torsion_quads[t*4+1]+atom_off,a3=torsion_quads[t*4+2]+atom_off,a4=torsion_quads[t*4+3]+atom_off;
                float cp=calc_cos_phi(out_pos,a1,a2,a3,a4,dim);
                lte+=torsion_e(cp,torsion_V[t*6],torsion_V[t*6+1],torsion_V[t*6+2],torsion_V[t*6+3],torsion_V[t*6+4],torsion_V[t*6+5],
                    torsion_signs_arr[t*6],torsion_signs_arr[t*6+1],torsion_signs_arr[t*6+2],torsion_signs_arr[t*6+3],torsion_signs_arr[t*6+4],torsion_signs_arr[t*6+5]);}
            for (int t=imp_s+(int)tid;t<imp_e;t+=(int)tpm){
                int ic=improper_quads[t*4]+atom_off,i0=improper_quads[t*4+1]+atom_off,i1=improper_quads[t*4+2]+atom_off,i2=improper_quads[t*4+3]+atom_off;
                lte+=improper_e(out_pos,ic,i0,i1,i2,improper_w[t],dim);}
            for (int t=d14_s+(int)tid;t<d14_e;t+=(int)tpm){
                int a=d14_pairs[t*2]+atom_off,b=d14_pairs[t*2+1]+atom_off;
                lte+=dist14_e(out_pos,a,b,d14_bounds[t*3],d14_bounds[t*3+1],d14_bounds[t*3+2],dim);}
            shared[tid]=lte; float trial_e=tg_reduce_sum(shared,tid,tpm);

            if (trial_e-old_energy<=FUNCTOL*lam*slope){energy=trial_e;ls_done=true;}
            else {
                float tl;
                if (ls==0) tl=-slope/(2.0f*(trial_e-old_energy-slope));
                else {float r1=trial_e-old_energy-lam*slope,r2=prev_e-old_energy-prev_lam*slope;
                    float ls2=lam*lam,l2s=prev_lam*prev_lam,dv=lam-prev_lam;
                    if(abs(dv)<1e-30f)tl=0.5f*lam;
                    else{float a=(r1/ls2-r2/l2s)/dv,b=(-prev_lam*r1/ls2+lam*r2/l2s)/dv;
                        if(abs(a)<1e-30f)tl=(abs(b)>1e-30f)?-slope/(2.0f*b):0.5f*lam;
                        else{float disc=b*b-3.0f*a*slope;if(disc<0.0f)tl=0.5f*lam;else if(b<=0.0f)tl=(-b+sqrt(disc))/(3.0f*a);else tl=-slope/(b+sqrt(disc));}}}
                tl=clamp(tl,0.1f*lam,0.5f*lam);prev_lam=lam;prev_e=trial_e;lam=tl;
            }
        }
        if (!ls_done) parallel_copy(my_pos,my_old_pos,n_vars,tid,tpm);

        for (int i=(int)tid;i<n_vars;i+=(int)tpm) my_old_pos[i]=my_pos[i]-my_old_pos[i];
        threadgroup_barrier(mem_flags::mem_device);
        float ltx=0.0f; for (int i=(int)tid;i<n_vars;i+=(int)tpm){float t=abs(my_old_pos[i])/max(abs(my_pos[i]),1.0f);if(t>ltx)ltx=t;}
        shared[tid]=ltx; if(tg_reduce_max(shared,tid,tpm)<TOLX){status=0;break;}

        parallel_copy(my_old_grad,my_grad,n_vars,tid,tpm);
        parallel_set(my_grad,0.0f,n_vars,tid,tpm);

        // New energy (parallel) + gradient (thread 0)
        float lne=0.0f;
        for (int t=tor_s+(int)tid;t<tor_e;t+=(int)tpm){int a1=torsion_quads[t*4]+atom_off,a2=torsion_quads[t*4+1]+atom_off,a3=torsion_quads[t*4+2]+atom_off,a4=torsion_quads[t*4+3]+atom_off;
            float cp=calc_cos_phi(out_pos,a1,a2,a3,a4,dim);
            lne+=torsion_e(cp,torsion_V[t*6],torsion_V[t*6+1],torsion_V[t*6+2],torsion_V[t*6+3],torsion_V[t*6+4],torsion_V[t*6+5],
                torsion_signs_arr[t*6],torsion_signs_arr[t*6+1],torsion_signs_arr[t*6+2],torsion_signs_arr[t*6+3],torsion_signs_arr[t*6+4],torsion_signs_arr[t*6+5]);}
        for (int t=imp_s+(int)tid;t<imp_e;t+=(int)tpm){int ic=improper_quads[t*4]+atom_off,i0=improper_quads[t*4+1]+atom_off,i1=improper_quads[t*4+2]+atom_off,i2=improper_quads[t*4+3]+atom_off;
            lne+=improper_e(out_pos,ic,i0,i1,i2,improper_w[t],dim);}
        for (int t=d14_s+(int)tid;t<d14_e;t+=(int)tpm){int a=d14_pairs[t*2]+atom_off,b=d14_pairs[t*2+1]+atom_off;
            lne+=dist14_e(out_pos,a,b,d14_bounds[t*3],d14_bounds[t*3+1],d14_bounds[t*3+2],dim);}
        shared[tid]=lne; energy=tg_reduce_sum(shared,tid,tpm);

        if (tid==0) {
            for (int t=tor_s;t<tor_e;t++){int a1=torsion_quads[t*4]+atom_off,a2=torsion_quads[t*4+1]+atom_off,a3=torsion_quads[t*4+2]+atom_off,a4=torsion_quads[t*4+3]+atom_off;
                torsion_g(out_pos,work_grad,a1,a2,a3,a4,torsion_V[t*6],torsion_V[t*6+1],torsion_V[t*6+2],torsion_V[t*6+3],torsion_V[t*6+4],torsion_V[t*6+5],
                    torsion_signs_arr[t*6],torsion_signs_arr[t*6+1],torsion_signs_arr[t*6+2],torsion_signs_arr[t*6+3],torsion_signs_arr[t*6+4],torsion_signs_arr[t*6+5],dim);}
            for (int t=imp_s;t<imp_e;t++){int ic=improper_quads[t*4]+atom_off,i0=improper_quads[t*4+1]+atom_off,i1=improper_quads[t*4+2]+atom_off,i2=improper_quads[t*4+3]+atom_off;
                improper_g(out_pos,work_grad,ic,i0,i1,i2,improper_w[t],dim);}
            for (int t=d14_s;t<d14_e;t++){int a=d14_pairs[t*2]+atom_off,b=d14_pairs[t*2+1]+atom_off;
                dist14_g(out_pos,work_grad,a,b,d14_bounds[t*3],d14_bounds[t*3+1],d14_bounds[t*3+2],dim);}
        }
        threadgroup_barrier(mem_flags::mem_device);

        float lgt=0.0f; for (int i=(int)tid;i<n_vars;i+=(int)tpm){float t=abs(my_grad[i])*max(abs(my_pos[i]),1.0f);if(t>lgt)lgt=t;}
        shared[tid]=lgt; if(tg_reduce_max(shared,tid,tpm)/max(energy,1.0f)<grad_tol_v){status=0;break;}

        // L-BFGS update
        for (int i=(int)tid;i<n_vars;i+=(int)tpm) my_q[i]=my_grad[i]-my_old_grad[i];
        threadgroup_barrier(mem_flags::mem_device);
        float ys=parallel_dot(my_q,my_old_pos,n_vars,tid,tpm,shared);
        if (ys>1e-10f){int sl=hist_idx%lbfgs_m;parallel_copy(&my_S[sl*n_vars],my_old_pos,n_vars,tid,tpm);
            parallel_copy(&my_Y[sl*n_vars],my_q,n_vars,tid,tpm);if(tid==0)my_rho[sl]=1.0f/ys;
            threadgroup_barrier(mem_flags::mem_device);hist_idx++;if(hist_count<lbfgs_m)hist_count++;}

        parallel_copy(my_q,my_grad,n_vars,tid,tpm);
        device float* my_alpha=&work_alpha[conf_idx*lbfgs_m];
        for (int j=hist_count-1;j>=0;j--){int sl=(hist_idx-1-(hist_count-1-j))%lbfgs_m;if(sl<0)sl+=lbfgs_m;
            float aj=my_rho[sl]*parallel_dot(&my_S[sl*n_vars],my_q,n_vars,tid,tpm,shared);
            if(tid==0)my_alpha[j]=aj;threadgroup_barrier(mem_flags::mem_device);
            parallel_saxpy(my_q,-aj,&my_Y[sl*n_vars],n_vars,tid,tpm);}
        if (hist_count>0){int nw=(hist_idx-1)%lbfgs_m;if(nw<0)nw+=lbfgs_m;
            float sy=parallel_dot(&my_S[nw*n_vars],&my_Y[nw*n_vars],n_vars,tid,tpm,shared);
            float yy=parallel_dot(&my_Y[nw*n_vars],&my_Y[nw*n_vars],n_vars,tid,tpm,shared);
            parallel_scale(my_q,sy/max(yy,1e-30f),n_vars,tid,tpm);}
        for (int j=0;j<hist_count;j++){int sl=(hist_idx-1-(hist_count-1-j))%lbfgs_m;if(sl<0)sl+=lbfgs_m;
            float bj=my_rho[sl]*parallel_dot(&my_Y[sl*n_vars],my_q,n_vars,tid,tpm,shared);
            parallel_saxpy(my_q,my_alpha[j]-bj,&my_S[sl*n_vars],n_vars,tid,tpm);}
        parallel_neg_copy(my_dir,my_q,n_vars,tid,tpm);
    }
    if (tid==0){out_energies[conf_idx]=energy;out_statuses[conf_idx]=status;}
"""

# ---------------------------------------------------------------------------
# Kernel build + dispatch
# ---------------------------------------------------------------------------

_etk_kernel_cache: dict[tuple, object] = {}


def _get_etk_kernel(tpm: int = DEFAULT_TPM, lbfgs_m: int = DEFAULT_LBFGS_M):
    key = (tpm, lbfgs_m)
    if key not in _etk_kernel_cache:
        header = _ETK_HEADER.replace("TPM", str(tpm)).replace("LBFGS_M", str(lbfgs_m))
        source = _ETK_BODY.replace("TPM", str(tpm)).replace("LBFGS_M", str(lbfgs_m))
        _etk_kernel_cache[key] = mx.fast.metal_kernel(
            name="etk_lbfgs_shared",
            input_names=[
                "pos", "config",
                "conf_to_mol", "conf_atom_starts", "mol_n_atoms",
                "torsion_starts", "torsion_quads", "torsion_V", "torsion_signs_arr",
                "improper_starts", "improper_quads", "improper_w",
                "dist14_starts", "d14_pairs", "d14_bounds",
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
    return _etk_kernel_cache[key]


def etk_minimize_shared(
    batch: SharedConstraintBatch,
    positions: np.ndarray,
    *,
    max_iters: int = 300,
    grad_tol: float = 1e-4,
    tpm: int = DEFAULT_TPM,
    lbfgs_m: int = DEFAULT_LBFGS_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ETK L-BFGS on all C conformers in parallel with shared constraints.

    Requires ETK fields in *batch* (call ``add_etk_to_batch`` first).

    Parameters
    ----------
    batch : SharedConstraintBatch
    positions : np.ndarray, shape (n_atoms_total * 3,)
        3D positions from DG stage (after 4D→3D extraction).

    Returns
    -------
    out_positions, energies, statuses
    """
    C = batch.n_confs_total
    dim = 3  # ETK is always 3D
    total_pos_size = int(batch.conf_atom_starts[-1]) * dim

    config = np.array([C, max_iters, grad_tol, dim, total_pos_size], dtype=np.float32)

    # Pack torsion terms
    nt = len(batch.etk_torsion_idx) if batch.etk_torsion_idx is not None else 0
    if nt > 0:
        tor_quads = batch.etk_torsion_idx.flatten().astype(np.int32)
        tor_V = batch.etk_torsion_V.flatten().astype(np.float32)
        tor_signs = batch.etk_torsion_signs.flatten().astype(np.float32)
    else:
        tor_quads = np.zeros(4, dtype=np.int32)
        tor_V = np.zeros(6, dtype=np.float32)
        tor_signs = np.zeros(6, dtype=np.float32)

    # Pack improper terms
    ni = len(batch.etk_improper_idx) if batch.etk_improper_idx is not None else 0
    if ni > 0:
        imp_quads = batch.etk_improper_idx.flatten().astype(np.int32)
        imp_w = batch.etk_improper_weight.astype(np.float32)
    else:
        imp_quads = np.zeros(4, dtype=np.int32)
        imp_w = np.zeros(1, dtype=np.float32)

    # Pack 1-4 distance terms
    nd = len(batch.etk_dist14_idx1) if batch.etk_dist14_idx1 is not None else 0
    if nd > 0:
        d14_pairs = np.stack([batch.etk_dist14_idx1, batch.etk_dist14_idx2], axis=1).flatten().astype(np.int32)
        d14_bounds = np.stack([batch.etk_dist14_lb, batch.etk_dist14_ub, batch.etk_dist14_weight], axis=1).flatten().astype(np.float32)
    else:
        d14_pairs = np.zeros(2, dtype=np.int32)
        d14_bounds = np.zeros(3, dtype=np.float32)

    # L-BFGS history
    lbfgs_starts = np.zeros(C + 1, dtype=np.int32)
    for c in range(C):
        n_atoms = batch.mol_n_atoms[batch.conf_to_mol[c]]
        n_vars = n_atoms * dim
        lbfgs_starts[c + 1] = lbfgs_starts[c] + 2 * lbfgs_m * n_vars
    total_lbfgs = int(lbfgs_starts[-1])

    kernel = _get_etk_kernel(tpm, lbfgs_m)
    results = kernel(
        inputs=[
            mx.array(positions),
            mx.array(config),
            mx.array(batch.conf_to_mol),
            mx.array(batch.conf_atom_starts),
            mx.array(batch.mol_n_atoms),
            mx.array(batch.etk_torsion_term_starts if batch.etk_torsion_term_starts is not None else np.zeros(batch.n_mols + 1, dtype=np.int32)),
            mx.array(tor_quads),
            mx.array(tor_V),
            mx.array(tor_signs),
            mx.array(batch.etk_improper_term_starts if batch.etk_improper_term_starts is not None else np.zeros(batch.n_mols + 1, dtype=np.int32)),
            mx.array(imp_quads),
            mx.array(imp_w),
            mx.array(batch.etk_dist14_term_starts if batch.etk_dist14_term_starts is not None else np.zeros(batch.n_mols + 1, dtype=np.int32)),
            mx.array(d14_pairs),
            mx.array(d14_bounds),
            mx.array(lbfgs_starts[:-1]),
        ],
        grid=(C * tpm, 1, 1),
        threadgroup=(tpm, 1, 1),
        output_shapes=[
            (total_pos_size,), (C,), (C,),
            (total_pos_size,), (total_pos_size,), (3 * total_pos_size,),
            (max(1, total_lbfgs),), (max(1, C * lbfgs_m),), (max(1, C * lbfgs_m),),
        ],
        output_dtypes=[
            mx.float32, mx.float32, mx.int32,
            mx.float32, mx.float32, mx.float32,
            mx.float32, mx.float32, mx.float32,
        ],
    )
    mx.eval(results[0], results[1], results[2])
    return np.array(results[0]), np.array(results[1]), np.array(results[2])
