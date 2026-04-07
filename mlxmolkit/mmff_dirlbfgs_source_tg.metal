
    uint mol_idx = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    uint tg_size = threads_per_threadgroup.x;

    int n_mols_cfg = (int)config[0];
    int max_iters = (int)config[1];
    float grad_tol = config[2];
    int lbfgs_m = LBFGS_M_VAL;

    // DirL-BFGS: per-history-entry counts
    int NOPUSH = 3;   // rank-1 terms per (s,y) pair (non-gamma)
    int GPUSH  = 5;   // rank-1 terms per (s,y) pair (gamma-aware)

    if ((int)mol_idx >= n_mols_cfg) return;

    int atom_start = atom_starts[mol_idx];
    int atom_end = atom_starts[mol_idx + 1];
    int n_atoms = atom_end - atom_start;
    int n_terms = n_atoms * 3;

    int ts_stride = n_mols_cfg + 1;
    int b_s = all_term_starts[0*ts_stride+mol_idx], b_e = all_term_starts[0*ts_stride+mol_idx+1];
    int a_s = all_term_starts[1*ts_stride+mol_idx], a_e = all_term_starts[1*ts_stride+mol_idx+1];
    int sb_s = all_term_starts[2*ts_stride+mol_idx], sb_e = all_term_starts[2*ts_stride+mol_idx+1];
    int o_s = all_term_starts[3*ts_stride+mol_idx], o_e = all_term_starts[3*ts_stride+mol_idx+1];
    int t_s = all_term_starts[4*ts_stride+mol_idx], t_e = all_term_starts[4*ts_stride+mol_idx+1];
    int v_s = all_term_starts[5*ts_stride+mol_idx], v_e = all_term_starts[5*ts_stride+mol_idx+1];
    int e_s = all_term_starts[6*ts_stride+mol_idx], e_e = all_term_starts[6*ts_stride+mol_idx+1];

    // Copy initial positions (parallel)
    for (int i = (int)tid; i < n_terms; i += (int)tg_size)
        out_pos[atom_start * 3 + i] = pos[atom_start * 3 + i];
    threadgroup_barrier(mem_flags::mem_device);

    device float* my_pos = &out_pos[atom_start * 3];
    device float* my_grad = &work_grad[atom_start * 3];
    device float* my_dir = &work_dir[atom_start * 3];
    device float* my_old_pos = &work_scratch[atom_start * 3];
    device float* my_old_grad = &work_scratch[total_pos_size + atom_start * 3];
    device float* my_q = &work_scratch[2 * total_pos_size + atom_start * 3];

    // DirL-BFGS vector storage: all packed into work_dirvecs
    // Layout: [U_all | V_all | GU_all | GV_all | SY_all]
    // Compute per-molecule offsets by scanning atom_starts
    int uv_per_mol = NOPUSH * lbfgs_m * n_terms;   // per-mol U (or V) size
    int guv_per_mol = GPUSH * lbfgs_m * n_terms;    // per-mol GU (or GV) size
    int sy_per_mol = 2 * lbfgs_m * n_terms;          // per-mol S+Y size

    // Scan to find offsets for this molecule
    int my_uv_off = 0, my_guv_off = 0, my_sy_off = 0;
    int total_uv_all = 0, total_guv_all = 0, total_sy_all = 0;
    for (int m = 0; m < n_mols_cfg; m++) {
        int na_m = atom_starts[m + 1] - atom_starts[m];
        int nt_m = na_m * 3;
        if (m < (int)mol_idx) {
            my_uv_off  += NOPUSH * lbfgs_m * nt_m;
            my_guv_off += GPUSH  * lbfgs_m * nt_m;
            my_sy_off  += 2 * lbfgs_m * nt_m;
        }
        total_uv_all  += NOPUSH * lbfgs_m * nt_m;
        total_guv_all += GPUSH  * lbfgs_m * nt_m;
        total_sy_all  += 2 * lbfgs_m * nt_m;
    }

    // Offsets into the packed work_dirvecs buffer
    int off_U  = my_uv_off;
    int off_V  = total_uv_all + my_uv_off;
    int off_GU = 2 * total_uv_all + my_guv_off;
    int off_GV = 2 * total_uv_all + total_guv_all + my_guv_off;
    int off_SY = 2 * total_uv_all + 2 * total_guv_all + my_sy_off;

    device float* my_U  = &work_dirvecs[off_U];
    device float* my_V  = &work_dirvecs[off_V];
    device float* my_GU = &work_dirvecs[off_GU];
    device float* my_GV = &work_dirvecs[off_GV];
    device float* my_S  = &work_dirvecs[off_SY];
    device float* my_Y  = &work_dirvecs[off_SY + lbfgs_m * n_terms];

    // Scalar storage: packed into work_dirscalars
    // Layout: [beta_all | gbeta_all | gamma_all | rho_all]
    int sc_beta_off  = mol_idx * NOPUSH * lbfgs_m;
    int sc_gbeta_off = n_mols_cfg * NOPUSH * lbfgs_m + mol_idx * GPUSH * lbfgs_m;
    int sc_gamma_off = n_mols_cfg * (NOPUSH + GPUSH) * lbfgs_m + mol_idx * lbfgs_m;
    int sc_rho_off   = n_mols_cfg * (NOPUSH + GPUSH + 1) * lbfgs_m + mol_idx * lbfgs_m;

    device float* my_beta       = &work_dirscalars[sc_beta_off];
    device float* my_gbeta      = &work_dirscalars[sc_gbeta_off];
    device float* my_gamma_list = &work_dirscalars[sc_gamma_off];
    device float* my_rho        = &work_dirscalars[sc_rho_off];

    // Scratch vectors for tempv/gtempv
    device float* my_tempv  = &work_tempvec[atom_start * 3];
    device float* my_gtempv = &work_tempvec[total_pos_size + atom_start * 3];

    threadgroup float tg_reduce[TG_SIZE_VAL];
    threadgroup float tg_grad_scale_shared;
    threadgroup int tg_status_shared;

    // Serial energy+gradient macro (same as BFGS variant)
    #define SEQ_COMPUTE_EG(OUT_E) \
        if (tid == 0) { \
            for (int i = 0; i < n_terms; i++) my_grad[i] = 0.0f; \
            OUT_E = 0.0f; \
            for (int t = b_s; t < b_e; t++) { \
                OUT_E += bond_stretch_e(out_pos, bond_pairs[t*2], bond_pairs[t*2+1], bond_params[t*2], bond_params[t*2+1]); \
                bond_stretch_g(out_pos, work_grad, bond_pairs[t*2], bond_pairs[t*2+1], bond_params[t*2], bond_params[t*2+1]); \
            } \
            for (int t = a_s; t < a_e; t++) { \
                bool lin = angle_params[t*3+2] > 0.5f; \
                OUT_E += angle_bend_e(out_pos, angle_trips[t*3], angle_trips[t*3+1], angle_trips[t*3+2], angle_params[t*3], angle_params[t*3+1], lin); \
                angle_bend_g(out_pos, work_grad, angle_trips[t*3], angle_trips[t*3+1], angle_trips[t*3+2], angle_params[t*3], angle_params[t*3+1], lin); \
            } \
            for (int t = sb_s; t < sb_e; t++) { \
                OUT_E += stretch_bend_e(out_pos, sb_trips[t*3], sb_trips[t*3+1], sb_trips[t*3+2], sb_params[t*5], sb_params[t*5+1], sb_params[t*5+2], sb_params[t*5+3], sb_params[t*5+4]); \
                stretch_bend_g(out_pos, work_grad, sb_trips[t*3], sb_trips[t*3+1], sb_trips[t*3+2], sb_params[t*5], sb_params[t*5+1], sb_params[t*5+2], sb_params[t*5+3], sb_params[t*5+4]); \
            } \
            for (int t = o_s; t < o_e; t++) { \
                OUT_E += oop_bend_e(out_pos, oop_quads[t*4], oop_quads[t*4+1], oop_quads[t*4+2], oop_quads[t*4+3], oop_params[t]); \
                oop_bend_g(out_pos, work_grad, oop_quads[t*4], oop_quads[t*4+1], oop_quads[t*4+2], oop_quads[t*4+3], oop_params[t]); \
            } \
            for (int t = t_s; t < t_e; t++) { \
                OUT_E += torsion_e(out_pos, tor_quads[t*4], tor_quads[t*4+1], tor_quads[t*4+2], tor_quads[t*4+3], tor_params[t*3], tor_params[t*3+1], tor_params[t*3+2]); \
                torsion_g(out_pos, work_grad, tor_quads[t*4], tor_quads[t*4+1], tor_quads[t*4+2], tor_quads[t*4+3], tor_params[t*3], tor_params[t*3+1], tor_params[t*3+2]); \
            } \
            for (int t = v_s; t < v_e; t++) { \
                OUT_E += vdw_e(out_pos, vdw_pairs[t*2], vdw_pairs[t*2+1], vdw_params[t*2], vdw_params[t*2+1]); \
                vdw_g(out_pos, work_grad, vdw_pairs[t*2], vdw_pairs[t*2+1], vdw_params[t*2], vdw_params[t*2+1]); \
            } \
            for (int t = e_s; t < e_e; t++) { \
                OUT_E += ele_e(out_pos, ele_pairs[t*2], ele_pairs[t*2+1], ele_params[t*3], (int)ele_params[t*3+1], ele_params[t*3+2]>0.5f); \
                ele_g(out_pos, work_grad, ele_pairs[t*2], ele_pairs[t*2+1], ele_params[t*3], (int)ele_params[t*3+1], ele_params[t*3+2]>0.5f); \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_device);

    // Parallel energy-only macro
    #define PAR_COMPUTE_E(OUT_E) \
        { \
            float _local_e = 0.0f; \
            for (int t = b_s + (int)tid; t < b_e; t += (int)tg_size) \
                _local_e += bond_stretch_e(out_pos, bond_pairs[t*2], bond_pairs[t*2+1], bond_params[t*2], bond_params[t*2+1]); \
            for (int t = a_s + (int)tid; t < a_e; t += (int)tg_size) { \
                bool lin = angle_params[t*3+2] > 0.5f; \
                _local_e += angle_bend_e(out_pos, angle_trips[t*3], angle_trips[t*3+1], angle_trips[t*3+2], angle_params[t*3], angle_params[t*3+1], lin); \
            } \
            for (int t = sb_s + (int)tid; t < sb_e; t += (int)tg_size) \
                _local_e += stretch_bend_e(out_pos, sb_trips[t*3], sb_trips[t*3+1], sb_trips[t*3+2], sb_params[t*5], sb_params[t*5+1], sb_params[t*5+2], sb_params[t*5+3], sb_params[t*5+4]); \
            for (int t = o_s + (int)tid; t < o_e; t += (int)tg_size) \
                _local_e += oop_bend_e(out_pos, oop_quads[t*4], oop_quads[t*4+1], oop_quads[t*4+2], oop_quads[t*4+3], oop_params[t]); \
            for (int t = t_s + (int)tid; t < t_e; t += (int)tg_size) \
                _local_e += torsion_e(out_pos, tor_quads[t*4], tor_quads[t*4+1], tor_quads[t*4+2], tor_quads[t*4+3], tor_params[t*3], tor_params[t*3+1], tor_params[t*3+2]); \
            for (int t = v_s + (int)tid; t < v_e; t += (int)tg_size) \
                _local_e += vdw_e(out_pos, vdw_pairs[t*2], vdw_pairs[t*2+1], vdw_params[t*2], vdw_params[t*2+1]); \
            for (int t = e_s + (int)tid; t < e_e; t += (int)tg_size) \
                _local_e += ele_e(out_pos, ele_pairs[t*2], ele_pairs[t*2+1], ele_params[t*3], (int)ele_params[t*3+1], ele_params[t*3+2]>0.5f); \
            tg_reduce[tid] = _local_e; \
            OUT_E = tg_reduce_sum(tg_reduce, tid, tg_size); \
        }

    // ---- Initial energy + gradient ----
    float energy = 0.0f;
    SEQ_COMPUTE_EG(energy);
    if (tid == 0) {
        float grad_scale = 1.0f;
        scale_grad_serial(my_grad, n_terms, grad_scale, true);
        tg_grad_scale_shared = grad_scale;
        tg_reduce[0] = energy;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    energy = tg_reduce[0];

    parallel_neg_copy(my_dir, my_grad, n_terms, tid, tg_size);

    float local_sum_sq = 0.0f;
    for (int i = (int)tid; i < n_terms; i += (int)tg_size) local_sum_sq += my_pos[i] * my_pos[i];
    tg_reduce[tid] = local_sum_sq;
    float max_step = MAX_STEP_FACTOR * max(sqrt(tg_reduce_sum(tg_reduce, tid, tg_size)), (float)n_terms);

    if (tid == 0) tg_status_shared = 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int hist_count = 0;  // number of (s,y) pairs stored
    int n_uv  = 0;       // number of (u,v,beta) triplets
    int n_guv = 0;       // number of (gu,gv,gbeta) triplets
    float cur_gamma = 1.0f;

    // ---- Main DirL-BFGS loop ----
    for (int iter = 0; iter < max_iters && tg_status_shared == 1; iter++) {

        // === LINE SEARCH (same as L-BFGS variant) ===
        parallel_copy(my_old_pos, my_pos, n_terms, tid, tg_size);
        float old_energy = energy;

        float local_dir_sq = 0.0f;
        for (int i = (int)tid; i < n_terms; i += (int)tg_size) local_dir_sq += my_dir[i] * my_dir[i];
        tg_reduce[tid] = local_dir_sq;
        float dir_norm = sqrt(tg_reduce_sum(tg_reduce, tid, tg_size));
        if (dir_norm > max_step) {
            float sc = max_step / dir_norm;
            for (int i = (int)tid; i < n_terms; i += (int)tg_size) my_dir[i] *= sc;
            threadgroup_barrier(mem_flags::mem_device);
        }

        float slope = parallel_dot(my_dir, my_grad, n_terms, tid, tg_size, tg_reduce);
        if (slope >= 0.0f) {
            if (tid == 0) tg_status_shared = 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            break;
        }

        float local_test_max = 0.0f;
        for (int i = (int)tid; i < n_terms; i += (int)tg_size) {
            float tv = abs(my_dir[i]) / max(abs(my_pos[i]), 1.0f);
            if (tv > local_test_max) local_test_max = tv;
        }
        tg_reduce[tid] = local_test_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size/2; s>0; s>>=1) { if(tid<s) tg_reduce[tid]=max(tg_reduce[tid],tg_reduce[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup); }
        float lambda_min = MOVETOL / max(tg_reduce[0], 1e-30f);

        float lam = 1.0f, prev_lam = 1.0f, prev_e = old_energy;
        bool ls_done = false;
        for (int ls = 0; ls < MAX_LS_ITERS && !ls_done; ls++) {
            if (lam < lambda_min) { parallel_copy(my_pos, my_old_pos, n_terms, tid, tg_size); ls_done = true; break; }
            for (int i = (int)tid; i < n_terms; i += (int)tg_size) my_pos[i] = my_old_pos[i] + lam * my_dir[i];
            threadgroup_barrier(mem_flags::mem_device);
            float trial_e; PAR_COMPUTE_E(trial_e);
            if (trial_e - old_energy <= FUNCTOL * lam * slope) { energy = trial_e; ls_done = true; }
            else {
                float tmp_lam;
                if (ls==0) tmp_lam = -slope/(2.0f*(trial_e-old_energy-slope));
                else { float r1=trial_e-old_energy-lam*slope,r2=prev_e-old_energy-prev_lam*slope,ls2=lam*lam,l2s=prev_lam*prev_lam,dv=lam-prev_lam;
                    if(abs(dv)<1e-30f)tmp_lam=0.5f*lam;
                    else{float a=(r1/ls2-r2/l2s)/dv,b=(-prev_lam*r1/ls2+lam*r2/l2s)/dv;
                        if(abs(a)<1e-30f)tmp_lam=(abs(b)>1e-30f)?-slope/(2.0f*b):0.5f*lam;
                        else{float disc=b*b-3.0f*a*slope;if(disc<0.0f)tmp_lam=0.5f*lam;else if(b<=0.0f)tmp_lam=(-b+sqrt(disc))/(3.0f*a);else tmp_lam=-slope/(b+sqrt(disc));}}}
                tmp_lam=clamp(tmp_lam,0.1f*lam,0.5f*lam);prev_lam=lam;prev_e=trial_e;lam=tmp_lam;
            }
        }
        if (!ls_done) parallel_copy(my_pos, my_old_pos, n_terms, tid, tg_size);

        // s_k = pos - old_pos  (stored in my_old_pos)
        for (int i = (int)tid; i < n_terms; i += (int)tg_size) my_old_pos[i] = my_pos[i] - my_old_pos[i];
        threadgroup_barrier(mem_flags::mem_device);

        // TOLX check
        float local_tolx = 0.0f;
        for (int i = (int)tid; i < n_terms; i += (int)tg_size) {
            float tv = abs(my_old_pos[i]) / max(abs(my_pos[i]), 1.0f);
            if (tv > local_tolx) local_tolx = tv;
        }
        tg_reduce[tid] = local_tolx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s=tg_size/2;s>0;s>>=1){if(tid<s)tg_reduce[tid]=max(tg_reduce[tid],tg_reduce[tid+s]);threadgroup_barrier(mem_flags::mem_threadgroup);}
        if (tg_reduce[0] < TOLX) { if(tid==0)tg_status_shared=0; threadgroup_barrier(mem_flags::mem_threadgroup); break; }

        // Save old grad, recompute energy+gradient
        for (int i = (int)tid; i < n_terms; i += (int)tg_size) my_old_grad[i] = my_grad[i];
        threadgroup_barrier(mem_flags::mem_device);
        float new_e = 0.0f;
        SEQ_COMPUTE_EG(new_e);
        if (tid == 0) {
            float grad_scale = tg_grad_scale_shared;
            scale_grad_serial(my_grad, n_terms, grad_scale, false);
            tg_grad_scale_shared = grad_scale;
            tg_reduce[0] = new_e;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        energy = tg_reduce[0];

        // Grad convergence check
        float local_grad_test = 0.0f;
        for (int i = (int)tid; i < n_terms; i += (int)tg_size) {
            float tv = abs(my_grad[i]) * max(abs(my_pos[i]), 1.0f);
            if (tv > local_grad_test) local_grad_test = tv;
        }
        tg_reduce[tid] = local_grad_test;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s=tg_size/2;s>0;s>>=1){if(tid<s)tg_reduce[tid]=max(tg_reduce[tid],tg_reduce[tid+s]);threadgroup_barrier(mem_flags::mem_threadgroup);}
        if (tg_reduce[0] / max(energy * tg_grad_scale_shared, 1.0f) < grad_tol) {
            if(tid==0)tg_status_shared=0; threadgroup_barrier(mem_flags::mem_threadgroup); break;
        }

        // ==== DirL-BFGS UPDATE ====
        // y_k = grad_new - grad_old  (stored in my_q)
        for (int i = (int)tid; i < n_terms; i += (int)tg_size)
            my_q[i] = my_grad[i] - my_old_grad[i];
        threadgroup_barrier(mem_flags::mem_device);

        // s_k is in my_old_pos, y_k is in my_q
        float ys = parallel_dot(my_q, my_old_pos, n_terms, tid, tg_size, tg_reduce);

        if (ys > 1e-10f) {
            float rho_k = 1.0f / ys;

            // Store s_k and y_k in circular buffer
            int slot = hist_count % lbfgs_m;
            parallel_copy(&my_S[slot * n_terms], my_old_pos, n_terms, tid, tg_size);
            parallel_copy(&my_Y[slot * n_terms], my_q, n_terms, tid, tg_size);
            if (tid == 0) my_rho[slot] = rho_k;
            threadgroup_barrier(mem_flags::mem_device);

            // Evict oldest entries if at capacity
            if (hist_count >= lbfgs_m) {
                // Shift u/v/beta arrays left by NOPUSH
                for (int j = (int)tid; j < (n_uv - NOPUSH) * n_terms; j += (int)tg_size) {
                    my_U[j] = my_U[j + NOPUSH * n_terms];
                    my_V[j] = my_V[j + NOPUSH * n_terms];
                }
                if (tid == 0) {
                    for (int j = 0; j < n_uv - NOPUSH; j++) my_beta[j] = my_beta[j + NOPUSH];
                }
                threadgroup_barrier(mem_flags::mem_device);
                n_uv -= NOPUSH;

                // Shift gu/gv/gbeta arrays left by GPUSH
                for (int j = (int)tid; j < (n_guv - GPUSH) * n_terms; j += (int)tg_size) {
                    my_GU[j] = my_GU[j + GPUSH * n_terms];
                    my_GV[j] = my_GV[j + GPUSH * n_terms];
                }
                if (tid == 0) {
                    for (int j = 0; j < n_guv - GPUSH; j++) my_gbeta[j] = my_gbeta[j + GPUSH];
                    // Shift gamma_list
                    for (int j = 0; j < lbfgs_m - 1; j++) my_gamma_list[j] = my_gamma_list[j + 1];
                }
                threadgroup_barrier(mem_flags::mem_device);
                n_guv -= GPUSH;
            }

            // Update gamma (H_diag = ys / yy)
            float yy = parallel_dot(my_q, my_q, n_terms, tid, tg_size, tg_reduce);
            cur_gamma = ys / max(yy, 1e-30f);

            // Store gamma
            int gidx = min(hist_count, lbfgs_m - 1);
            if (tid == 0) my_gamma_list[gidx] = cur_gamma;
            threadgroup_barrier(mem_flags::mem_device);

            // --- Compute tempv[i] = sum_j(beta[j] * dot(U[j], y_k) * V[j][i]) ---
            // --- Compute gtempv[i] = sum_j(gbeta[j] * dot(GU[j], y_k) * GV[j][i]) ---
            // First: compute all dot products dot(U[j], y_k) and dot(GU[j], y_k)

            // Zero tempv and gtempv
            for (int i = (int)tid; i < n_terms; i += (int)tg_size) {
                my_tempv[i] = 0.0f;
                my_gtempv[i] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_device);

            // Also need tempb = sum_j(beta[j] * dot(U[j], y_k) * dot(V[j], y_k))
            // and gtempb = sum_j(gbeta[j] * dot(GU[j], y_k) * dot(GV[j], y_k))
            float tempb = 0.0f;
            float gtempb = 0.0f;

            // Process non-gamma terms
            for (int j = 0; j < n_uv; j++) {
                float uj_dot_yk = parallel_dot(&my_U[j * n_terms], my_q, n_terms, tid, tg_size, tg_reduce);
                float vj_dot_yk = parallel_dot(&my_V[j * n_terms], my_q, n_terms, tid, tg_size, tg_reduce);
                float buj = my_beta[j] * uj_dot_yk;
                tempb += buj * vj_dot_yk;
                // tempv[i] += beta[j] * uj_dot_yk * V[j][i]
                for (int i = (int)tid; i < n_terms; i += (int)tg_size)
                    my_tempv[i] += buj * my_V[j * n_terms + i];
                threadgroup_barrier(mem_flags::mem_device);
            }

            // Process gamma terms
            for (int j = 0; j < n_guv; j++) {
                float guj_dot_yk = parallel_dot(&my_GU[j * n_terms], my_q, n_terms, tid, tg_size, tg_reduce);
                float gvj_dot_yk = parallel_dot(&my_GV[j * n_terms], my_q, n_terms, tid, tg_size, tg_reduce);
                float gbuj = my_gbeta[j] * guj_dot_yk;
                gtempb += gbuj * gvj_dot_yk;
                // gtempv[i] += gbeta[j] * guj_dot_yk * GV[j][i]
                for (int i = (int)tid; i < n_terms; i += (int)tg_size)
                    my_gtempv[i] += gbuj * my_GV[j * n_terms + i];
                threadgroup_barrier(mem_flags::mem_device);
            }

            // --- Push 5 new gamma-aware triplets (gu, gv, gbeta) ---
            int ge = n_guv;
            // gbeta[ge..ge+3] = -rho_k
            if (tid == 0) {
                my_gbeta[ge]   = -rho_k;
                my_gbeta[ge+1] = -rho_k;
                my_gbeta[ge+2] = -rho_k;
                my_gbeta[ge+3] = -rho_k;
                my_gbeta[ge+4] = rho_k * rho_k * (yy + gtempb);
            }
            // gu[ge+0] = s_k, gv[ge+0] = y_k
            parallel_copy(&my_GU[ge * n_terms], my_old_pos, n_terms, tid, tg_size);
            parallel_copy(&my_GV[ge * n_terms], my_q, n_terms, tid, tg_size);
            // gu[ge+1] = y_k, gv[ge+1] = s_k
            parallel_copy(&my_GU[(ge+1) * n_terms], my_q, n_terms, tid, tg_size);
            parallel_copy(&my_GV[(ge+1) * n_terms], my_old_pos, n_terms, tid, tg_size);
            // gu[ge+2] = s_k, gv[ge+2] = gtempv
            parallel_copy(&my_GU[(ge+2) * n_terms], my_old_pos, n_terms, tid, tg_size);
            parallel_copy(&my_GV[(ge+2) * n_terms], my_gtempv, n_terms, tid, tg_size);
            // gu[ge+3] = gtempv (=gtempu), gv[ge+3] = s_k
            parallel_copy(&my_GU[(ge+3) * n_terms], my_gtempv, n_terms, tid, tg_size);
            parallel_copy(&my_GV[(ge+3) * n_terms], my_old_pos, n_terms, tid, tg_size);
            // gu[ge+4] = s_k, gv[ge+4] = s_k
            parallel_copy(&my_GU[(ge+4) * n_terms], my_old_pos, n_terms, tid, tg_size);
            parallel_copy(&my_GV[(ge+4) * n_terms], my_old_pos, n_terms, tid, tg_size);
            n_guv += GPUSH;

            // --- Push 3 new non-gamma triplets (u, v, beta) ---
            int ne = n_uv;
            if (tid == 0) {
                my_beta[ne]   = -rho_k;
                my_beta[ne+1] = -rho_k;
                my_beta[ne+2] = rho_k * rho_k * tempb + rho_k;
            }
            // u[ne+0] = s_k, v[ne+0] = tempv
            parallel_copy(&my_U[ne * n_terms], my_old_pos, n_terms, tid, tg_size);
            parallel_copy(&my_V[ne * n_terms], my_tempv, n_terms, tid, tg_size);
            // u[ne+1] = tempv (=tempu), v[ne+1] = s_k
            parallel_copy(&my_U[(ne+1) * n_terms], my_tempv, n_terms, tid, tg_size);
            parallel_copy(&my_V[(ne+1) * n_terms], my_old_pos, n_terms, tid, tg_size);
            // u[ne+2] = s_k, v[ne+2] = s_k
            parallel_copy(&my_U[(ne+2) * n_terms], my_old_pos, n_terms, tid, tg_size);
            parallel_copy(&my_V[(ne+2) * n_terms], my_old_pos, n_terms, tid, tg_size);
            n_uv += NOPUSH;

            if (hist_count < lbfgs_m) hist_count++;
        }

        // ==== DirL-BFGS DIRECTION COMPUTATION ====
        // d = -(sum_j beta[j]*(V[j].g)*U[j] + cur_gamma*(sum_j gbeta[j]*(GV[j].g)*GU[j] + g))
        if (hist_count == 0) {
            // First iteration: steepest descent
            parallel_neg_copy(my_dir, my_grad, n_terms, tid, tg_size);
        } else {
            // Phase 1: compute all dot products V[j].grad (independent!)
            // Phase 2: weighted sum

            // Zero accumulator in my_q
            for (int i = (int)tid; i < n_terms; i += (int)tg_size) my_q[i] = 0.0f;
            threadgroup_barrier(mem_flags::mem_device);

            // Non-gamma terms: sum_j beta[j] * (V[j].g) * U[j]
            for (int j = 0; j < n_uv; j++) {
                float vj_dot_g = parallel_dot(&my_V[j * n_terms], my_grad, n_terms, tid, tg_size, tg_reduce);
                float bv = my_beta[j] * vj_dot_g;
                for (int i = (int)tid; i < n_terms; i += (int)tg_size)
                    my_q[i] += bv * my_U[j * n_terms + i];
                threadgroup_barrier(mem_flags::mem_device);
            }

            // Gamma terms: cur_gamma * (sum_j gbeta[j] * (GV[j].g) * GU[j] + g)
            // Accumulate gamma part in my_dir as scratch
            for (int i = (int)tid; i < n_terms; i += (int)tg_size) my_dir[i] = my_grad[i];
            threadgroup_barrier(mem_flags::mem_device);

            for (int j = 0; j < n_guv; j++) {
                float gvj_dot_g = parallel_dot(&my_GV[j * n_terms], my_grad, n_terms, tid, tg_size, tg_reduce);
                float gbv = my_gbeta[j] * gvj_dot_g;
                for (int i = (int)tid; i < n_terms; i += (int)tg_size)
                    my_dir[i] += gbv * my_GU[j * n_terms + i];
                threadgroup_barrier(mem_flags::mem_device);
            }

            // Final: d = -(non_gamma_sum + cur_gamma * gamma_sum)
            for (int i = (int)tid; i < n_terms; i += (int)tg_size)
                my_dir[i] = -(my_q[i] + cur_gamma * my_dir[i]);
            threadgroup_barrier(mem_flags::mem_device);
        }
    }

    if (tid == 0) {
        out_energies[mol_idx] = energy;
        out_statuses[mol_idx] = tg_status_shared;
    }
