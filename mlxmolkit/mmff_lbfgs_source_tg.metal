
    uint mol_idx = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    uint tg_size = threads_per_threadgroup.x;

    int n_mols_cfg = (int)config[0];
    int max_iters = (int)config[1];
    float grad_tol = config[2];
    int lbfgs_m = LBFGS_M_VAL;

    if ((int)mol_idx >= n_mols_cfg) return;

    int atom_start = atom_starts[mol_idx];
    int atom_end = atom_starts[mol_idx + 1];
    int n_atoms = atom_end - atom_start;
    int n_terms = n_atoms * 3;
    int lbfgs_start = lbfgs_starts[mol_idx];

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

    // L-BFGS history: S and Y vectors (m pairs), rho scalars
    device float* my_S = &work_lbfgs[lbfgs_start];
    device float* my_Y = &work_lbfgs[lbfgs_start + lbfgs_m * n_terms];
    device float* my_rho = &work_rho[mol_idx * lbfgs_m];
    device float* my_alpha = &work_alpha[mol_idx * lbfgs_m];

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

    int hist_count = 0, hist_idx = 0;

    // ---- Main L-BFGS loop ----
    for (int iter = 0; iter < max_iters && tg_status_shared == 1; iter++) {

        // === LINE SEARCH (same as BFGS variant) ===
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

        // s_k = pos - old_pos
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

        // ==== L-BFGS UPDATE (replaces dense Hessian) ====
        // y_k = grad_new - grad_old
        for (int i = (int)tid; i < n_terms; i += (int)tg_size)
            my_q[i] = my_grad[i] - my_old_grad[i];
        threadgroup_barrier(mem_flags::mem_device);

        float ys = parallel_dot(my_q, my_old_pos, n_terms, tid, tg_size, tg_reduce);
        if (ys > 1e-10f) {
            int slot = hist_idx % lbfgs_m;
            parallel_copy(&my_S[slot * n_terms], my_old_pos, n_terms, tid, tg_size);
            parallel_copy(&my_Y[slot * n_terms], my_q, n_terms, tid, tg_size);
            if (tid == 0) my_rho[slot] = 1.0f / ys;
            threadgroup_barrier(mem_flags::mem_device);
            hist_idx++;
            if (hist_count < lbfgs_m) hist_count++;
        }

        // L-BFGS two-loop recursion → direction
        parallel_copy(my_q, my_grad, n_terms, tid, tg_size);

        for (int j = hist_count - 1; j >= 0; j--) {
            int slot = (hist_idx - 1 - (hist_count - 1 - j)) % lbfgs_m;
            if (slot < 0) slot += lbfgs_m;
            float aj = my_rho[slot] * parallel_dot(&my_S[slot*n_terms], my_q, n_terms, tid, tg_size, tg_reduce);
            if (tid == 0) my_alpha[j] = aj;
            threadgroup_barrier(mem_flags::mem_device);
            for (int i = (int)tid; i < n_terms; i += (int)tg_size) my_q[i] -= aj * my_Y[slot*n_terms+i];
            threadgroup_barrier(mem_flags::mem_device);
        }
        if (hist_count > 0) {
            int nw = (hist_idx - 1) % lbfgs_m; if (nw < 0) nw += lbfgs_m;
            float sy = parallel_dot(&my_S[nw*n_terms], &my_Y[nw*n_terms], n_terms, tid, tg_size, tg_reduce);
            float yy = parallel_dot(&my_Y[nw*n_terms], &my_Y[nw*n_terms], n_terms, tid, tg_size, tg_reduce);
            float gamma = sy / max(yy, 1e-30f);
            for (int i = (int)tid; i < n_terms; i += (int)tg_size) my_q[i] *= gamma;
            threadgroup_barrier(mem_flags::mem_device);
        }
        for (int j = 0; j < hist_count; j++) {
            int slot = (hist_idx - 1 - (hist_count - 1 - j)) % lbfgs_m;
            if (slot < 0) slot += lbfgs_m;
            float bj = my_rho[slot] * parallel_dot(&my_Y[slot*n_terms], my_q, n_terms, tid, tg_size, tg_reduce);
            float aj = my_alpha[j];
            for (int i = (int)tid; i < n_terms; i += (int)tg_size) my_q[i] += (aj - bj) * my_S[slot*n_terms+i];
            threadgroup_barrier(mem_flags::mem_device);
        }
        parallel_neg_copy(my_dir, my_q, n_terms, tid, tg_size);
    }

    if (tid == 0) {
        out_energies[mol_idx] = energy;
        out_statuses[mol_idx] = tg_status_shared;
    }
