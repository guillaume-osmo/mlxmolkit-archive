#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

namespace {

void augmented_overlap_axis(
    double p,
    double P,
    double A,
    double B,
    int la_max,
    int lb_max,
    double S[6][6]) {
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      S[i][j] = 0.0;
    }
  }

  const double PA = P - A;
  const double PB = P - B;
  const double inv2p = 1.0 / (2.0 * p);
  S[0][0] = 1.0;

  for (int i = 1; i <= la_max; ++i) {
    S[i][0] = PA * S[i - 1][0];
    if (i >= 2) {
      S[i][0] += inv2p * static_cast<double>(i - 1) * S[i - 2][0];
    }
  }
  for (int j = 1; j <= lb_max; ++j) {
    for (int i = 0; i <= la_max; ++i) {
      double term = PB * S[i][j - 1];
      if (j >= 2) {
        term += inv2p * static_cast<double>(j - 1) * S[i][j - 2];
      }
      if (i >= 1) {
        term += inv2p * static_cast<double>(i) * S[i - 1][j - 1];
      }
      S[i][j] = term;
    }
  }
}

void multipole_primitive(
    double alpha_a,
    const double* A,
    const int32_t* la,
    double alpha_b,
    const double* B,
    const int32_t* lb,
    double& S_out,
    double D[3],
    double Q[6]) {
  const double p = alpha_a + alpha_b;
  const double mu = alpha_a * alpha_b / p;
  const double P[3] = {
      (alpha_a * A[0] + alpha_b * B[0]) / p,
      (alpha_a * A[1] + alpha_b * B[1]) / p,
      (alpha_a * A[2] + alpha_b * B[2]) / p,
  };
  const double dx = A[0] - B[0];
  const double dy = A[1] - B[1];
  const double dz = A[2] - B[2];
  const double R2 = dx * dx + dy * dy + dz * dz;
  const double base = std::pow(M_PI / p, 1.5) * std::exp(-mu * R2);

  double Sx[6][6];
  double Sy[6][6];
  double Sz[6][6];
  augmented_overlap_axis(p, P[0], A[0], B[0], la[0] + 2, lb[0], Sx);
  augmented_overlap_axis(p, P[1], A[1], B[1], la[1] + 2, lb[1], Sy);
  augmented_overlap_axis(p, P[2], A[2], B[2], la[2] + 2, lb[2], Sz);

  const double sx = Sx[la[0]][lb[0]];
  const double sy = Sy[la[1]][lb[1]];
  const double sz = Sz[la[2]][lb[2]];

  S_out = base * sx * sy * sz;

  const double sx1 = Sx[la[0] + 1][lb[0]];
  const double sy1 = Sy[la[1] + 1][lb[1]];
  const double sz1 = Sz[la[2] + 1][lb[2]];
  const double sx2 = Sx[la[0] + 2][lb[0]];
  const double sy2 = Sy[la[1] + 2][lb[1]];
  const double sz2 = Sz[la[2] + 2][lb[2]];

  const double dx_axis = sx1 + A[0] * sx;
  const double dy_axis = sy1 + A[1] * sy;
  const double dz_axis = sz1 + A[2] * sz;

  D[0] = base * dx_axis * sy * sz;
  D[1] = base * sx * dy_axis * sz;
  D[2] = base * sx * sy * dz_axis;

  Q[0] = base * (sx2 + 2.0 * A[0] * sx1 + A[0] * A[0] * sx) * sy * sz;
  Q[1] = base * sx * (sy2 + 2.0 * A[1] * sy1 + A[1] * A[1] * sy) * sz;
  Q[2] = base * sx * sy * (sz2 + 2.0 * A[2] * sz1 + A[2] * A[2] * sz);
  Q[3] = base * dx_axis * dy_axis * sz;
  Q[4] = base * dx_axis * sy * dz_axis;
  Q[5] = base * sx * dy_axis * dz_axis;
}

void moment_at(
    const double Sx[6][6],
    const double Sy[6][6],
    const double Sz[6][6],
    double base,
    const double* A,
    const int32_t* la,
    const int32_t* lb,
    const int da[3],
    const int db[3],
    double& S_out,
    double D[3],
    double Q[6]) {
  const int ila[3] = {la[0] + da[0], la[1] + da[1], la[2] + da[2]};
  const int ilb[3] = {lb[0] + db[0], lb[1] + db[1], lb[2] + db[2]};
  if (ila[0] < 0 || ila[1] < 0 || ila[2] < 0 ||
      ilb[0] < 0 || ilb[1] < 0 || ilb[2] < 0) {
    S_out = 0.0;
    for (int k = 0; k < 3; ++k) {
      D[k] = 0.0;
    }
    for (int k = 0; k < 6; ++k) {
      Q[k] = 0.0;
    }
    return;
  }

  const double sx = Sx[ila[0]][ilb[0]];
  const double sy = Sy[ila[1]][ilb[1]];
  const double sz = Sz[ila[2]][ilb[2]];
  S_out = base * sx * sy * sz;

  const double sx1 = Sx[ila[0] + 1][ilb[0]];
  const double sy1 = Sy[ila[1] + 1][ilb[1]];
  const double sz1 = Sz[ila[2] + 1][ilb[2]];
  const double sx2 = Sx[ila[0] + 2][ilb[0]];
  const double sy2 = Sy[ila[1] + 2][ilb[1]];
  const double sz2 = Sz[ila[2] + 2][ilb[2]];

  const double dx_axis = sx1 + A[0] * sx;
  const double dy_axis = sy1 + A[1] * sy;
  const double dz_axis = sz1 + A[2] * sz;

  D[0] = base * dx_axis * sy * sz;
  D[1] = base * sx * dy_axis * sz;
  D[2] = base * sx * sy * dz_axis;

  Q[0] = base * (sx2 + 2.0 * A[0] * sx1 + A[0] * A[0] * sx) * sy * sz;
  Q[1] = base * sx * (sy2 + 2.0 * A[1] * sy1 + A[1] * A[1] * sy) * sz;
  Q[2] = base * sx * sy * (sz2 + 2.0 * A[2] * sz1 + A[2] * A[2] * sz);
  Q[3] = base * dx_axis * dy_axis * sz;
  Q[4] = base * dx_axis * sy * dz_axis;
  Q[5] = base * sx * dy_axis * dz_axis;
}

void multipole_primitive_grad(
    double alpha_a,
    const double* A,
    const int32_t* la,
    double alpha_b,
    const double* B,
    const int32_t* lb,
    double dDA[3][3],
    double dDB[3][3],
    double dQA[3][6],
    double dQB[3][6]) {
  const double p = alpha_a + alpha_b;
  const double mu = alpha_a * alpha_b / p;
  const double P[3] = {
      (alpha_a * A[0] + alpha_b * B[0]) / p,
      (alpha_a * A[1] + alpha_b * B[1]) / p,
      (alpha_a * A[2] + alpha_b * B[2]) / p,
  };
  const double dx = A[0] - B[0];
  const double dy = A[1] - B[1];
  const double dz = A[2] - B[2];
  const double R2 = dx * dx + dy * dy + dz * dz;
  const double base = std::pow(M_PI / p, 1.5) * std::exp(-mu * R2);

  double Sx[6][6];
  double Sy[6][6];
  double Sz[6][6];
  augmented_overlap_axis(p, P[0], A[0], B[0], la[0] + 3, lb[0] + 3, Sx);
  augmented_overlap_axis(p, P[1], A[1], B[1], la[1] + 3, lb[1] + 3, Sy);
  augmented_overlap_axis(p, P[2], A[2], B[2], la[2] + 3, lb[2] + 3, Sz);

  for (int beta = 0; beta < 3; ++beta) {
    int dap[3] = {0, 0, 0};
    int dam[3] = {0, 0, 0};
    int dbp[3] = {0, 0, 0};
    int dbm[3] = {0, 0, 0};
    dap[beta] = 1;
    dam[beta] = -1;
    dbp[beta] = 1;
    dbm[beta] = -1;

    double s_dummy;
    double Dp[3], Dm[3], Qp[6], Qm[6];
    const int zero[3] = {0, 0, 0};

    moment_at(Sx, Sy, Sz, base, A, la, lb, dap, zero, s_dummy, Dp, Qp);
    moment_at(Sx, Sy, Sz, base, A, la, lb, dam, zero, s_dummy, Dm, Qm);
    for (int k = 0; k < 3; ++k) {
      dDA[beta][k] = 2.0 * alpha_a * Dp[k] - static_cast<double>(la[beta]) * Dm[k];
    }
    for (int k = 0; k < 6; ++k) {
      dQA[beta][k] = 2.0 * alpha_a * Qp[k] - static_cast<double>(la[beta]) * Qm[k];
    }

    moment_at(Sx, Sy, Sz, base, A, la, lb, zero, dbp, s_dummy, Dp, Qp);
    moment_at(Sx, Sy, Sz, base, A, la, lb, zero, dbm, s_dummy, Dm, Qm);
    for (int k = 0; k < 3; ++k) {
      dDB[beta][k] = 2.0 * alpha_b * Dp[k] - static_cast<double>(lb[beta]) * Dm[k];
    }
    for (int k = 0; k < 6; ++k) {
      dQB[beta][k] = 2.0 * alpha_b * Qp[k] - static_cast<double>(lb[beta]) * Qm[k];
    }
  }
}

void primitive_overlap_grad(
    double alpha_a,
    const double* A,
    const int32_t* la,
    double alpha_b,
    const double* B,
    const int32_t* lb,
    double dA[3],
    double dB[3]) {
  const double p = alpha_a + alpha_b;
  const double mu = alpha_a * alpha_b / p;
  const double P[3] = {
      (alpha_a * A[0] + alpha_b * B[0]) / p,
      (alpha_a * A[1] + alpha_b * B[1]) / p,
      (alpha_a * A[2] + alpha_b * B[2]) / p,
  };
  const double dx = A[0] - B[0];
  const double dy = A[1] - B[1];
  const double dz = A[2] - B[2];
  const double R2 = dx * dx + dy * dy + dz * dz;
  const double base = std::pow(M_PI / p, 1.5) * std::exp(-mu * R2);

  double Sx[6][6];
  double Sy[6][6];
  double Sz[6][6];
  augmented_overlap_axis(p, P[0], A[0], B[0], la[0] + 1, lb[0] + 1, Sx);
  augmented_overlap_axis(p, P[1], A[1], B[1], la[1] + 1, lb[1] + 1, Sy);
  augmented_overlap_axis(p, P[2], A[2], B[2], la[2] + 1, lb[2] + 1, Sz);

  const double sx = Sx[la[0]][lb[0]];
  const double sy = Sy[la[1]][lb[1]];
  const double sz = Sz[la[2]][lb[2]];

  const double sxp = Sx[la[0] + 1][lb[0]];
  const double syp = Sy[la[1] + 1][lb[1]];
  const double szp = Sz[la[2] + 1][lb[2]];
  const double sxm = la[0] >= 1 ? Sx[la[0] - 1][lb[0]] : 0.0;
  const double sym = la[1] >= 1 ? Sy[la[1] - 1][lb[1]] : 0.0;
  const double szm = la[2] >= 1 ? Sz[la[2] - 1][lb[2]] : 0.0;

  const double sxbp = Sx[la[0]][lb[0] + 1];
  const double sybp = Sy[la[1]][lb[1] + 1];
  const double szbp = Sz[la[2]][lb[2] + 1];
  const double sxbm = lb[0] >= 1 ? Sx[la[0]][lb[0] - 1] : 0.0;
  const double sybm = lb[1] >= 1 ? Sy[la[1]][lb[1] - 1] : 0.0;
  const double szbm = lb[2] >= 1 ? Sz[la[2]][lb[2] - 1] : 0.0;

  const double twoa = 2.0 * alpha_a;
  const double twob = 2.0 * alpha_b;
  dA[0] = base * (twoa * sxp - static_cast<double>(la[0]) * sxm) * sy * sz;
  dA[1] = base * sx * (twoa * syp - static_cast<double>(la[1]) * sym) * sz;
  dA[2] = base * sx * sy * (twoa * szp - static_cast<double>(la[2]) * szm);

  dB[0] = base * (twob * sxbp - static_cast<double>(lb[0]) * sxbm) * sy * sz;
  dB[1] = base * sx * (twob * sybp - static_cast<double>(lb[1]) * sybm) * sz;
  dB[2] = base * sx * sy * (twob * szbp - static_cast<double>(lb[2]) * szbm);
}

PyObject* multipole_matrices_from_arrays(PyObject*, PyObject* args) {
  PyObject* centers_obj = nullptr;
  PyObject* lxyz_obj = nullptr;
  PyObject* offsets_obj = nullptr;
  PyObject* alphas_obj = nullptr;
  PyObject* coeffs_obj = nullptr;

  if (!PyArg_ParseTuple(
          args,
          "OOOOO",
          &centers_obj,
          &lxyz_obj,
          &offsets_obj,
          &alphas_obj,
          &coeffs_obj)) {
    return nullptr;
  }

  PyArrayObject* centers = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(centers_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* lxyz = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(lxyz_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* offsets = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(offsets_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* alphas = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(alphas_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* coeffs = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));

  if (!centers || !lxyz || !offsets || !alphas || !coeffs) {
    Py_XDECREF(centers);
    Py_XDECREF(lxyz);
    Py_XDECREF(offsets);
    Py_XDECREF(alphas);
    Py_XDECREF(coeffs);
    return nullptr;
  }

  const auto fail = [&](const char* msg) -> PyObject* {
    PyErr_SetString(PyExc_ValueError, msg);
    Py_DECREF(centers);
    Py_DECREF(lxyz);
    Py_DECREF(offsets);
    Py_DECREF(alphas);
    Py_DECREF(coeffs);
    return nullptr;
  };

  if (PyArray_NDIM(centers) != 2 || PyArray_DIM(centers, 1) != 3) {
    return fail("centers must have shape (n_basis, 3)");
  }
  if (PyArray_NDIM(lxyz) != 2 || PyArray_DIM(lxyz, 1) != 3) {
    return fail("lxyz must have shape (n_basis, 3)");
  }
  if (PyArray_NDIM(offsets) != 1) {
    return fail("offsets must be one-dimensional");
  }
  if (PyArray_NDIM(alphas) != 1 || PyArray_NDIM(coeffs) != 1) {
    return fail("alphas and coeffs must be one-dimensional");
  }

  const npy_intp n = PyArray_DIM(centers, 0);
  if (PyArray_DIM(lxyz, 0) != n || PyArray_DIM(offsets, 0) != n + 1) {
    return fail("basis array leading dimensions are inconsistent");
  }
  if (PyArray_DIM(alphas, 0) != PyArray_DIM(coeffs, 0)) {
    return fail("alphas and coeffs lengths differ");
  }

  const auto* centers_data = static_cast<const double*>(PyArray_DATA(centers));
  const auto* lxyz_data = static_cast<const int32_t*>(PyArray_DATA(lxyz));
  const auto* offsets_data = static_cast<const npy_intp*>(PyArray_DATA(offsets));
  const auto* alphas_data = static_cast<const double*>(PyArray_DATA(alphas));
  const auto* coeffs_data = static_cast<const double*>(PyArray_DATA(coeffs));

  if (offsets_data[0] != 0 || offsets_data[n] != PyArray_DIM(alphas, 0)) {
    return fail("offsets do not span the primitive arrays");
  }
  for (npy_intp mu = 0; mu < n; ++mu) {
    const int32_t* lmu = lxyz_data + 3 * mu;
    if (lmu[0] + 2 >= 6 || lmu[1] + 2 >= 6 || lmu[2] + 2 >= 6) {
      return fail("angular momentum is too large for this C++ kernel");
    }
  }

  npy_intp s_dims[2] = {n, n};
  npy_intp dp_dims[3] = {3, n, n};
  npy_intp qp_dims[3] = {6, n, n};
  PyObject* S_obj = PyArray_SimpleNew(2, s_dims, NPY_DOUBLE);
  PyObject* dp_obj = PyArray_SimpleNew(3, dp_dims, NPY_DOUBLE);
  PyObject* qp_obj = PyArray_SimpleNew(3, qp_dims, NPY_DOUBLE);
  if (!S_obj || !dp_obj || !qp_obj) {
    Py_XDECREF(S_obj);
    Py_XDECREF(dp_obj);
    Py_XDECREF(qp_obj);
    Py_DECREF(centers);
    Py_DECREF(lxyz);
    Py_DECREF(offsets);
    Py_DECREF(alphas);
    Py_DECREF(coeffs);
    return nullptr;
  }

  auto* S = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(S_obj)));
  auto* dp = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(dp_obj)));
  auto* qp = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(qp_obj)));

  for (npy_intp i = 0; i < n * n; ++i) {
    S[i] = 0.0;
  }
  for (npy_intp i = 0; i < 3 * n * n; ++i) {
    dp[i] = 0.0;
  }
  for (npy_intp i = 0; i < 6 * n * n; ++i) {
    qp[i] = 0.0;
  }

  for (npy_intp mu = 0; mu < n; ++mu) {
    const double* A = centers_data + 3 * mu;
    const int32_t* la = lxyz_data + 3 * mu;
    const npy_intp mu0 = offsets_data[mu];
    const npy_intp mu1 = offsets_data[mu + 1];

    for (npy_intp nu = mu; nu < n; ++nu) {
      const double* B = centers_data + 3 * nu;
      const int32_t* lb = lxyz_data + 3 * nu;
      const npy_intp nu0 = offsets_data[nu];
      const npy_intp nu1 = offsets_data[nu + 1];

      double S_mn = 0.0;
      double D_mn[3] = {0.0, 0.0, 0.0};
      double Q_mn[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      for (npy_intp i = mu0; i < mu1; ++i) {
        for (npy_intp j = nu0; j < nu1; ++j) {
          double s = 0.0;
          double d[3];
          double q[6];
          multipole_primitive(
              alphas_data[i],
              A,
              la,
              alphas_data[j],
              B,
              lb,
              s,
              d,
              q);
          const double c = coeffs_data[i] * coeffs_data[j];
          S_mn += c * s;
          D_mn[0] += c * d[0];
          D_mn[1] += c * d[1];
          D_mn[2] += c * d[2];
          Q_mn[0] += c * q[0];
          Q_mn[1] += c * q[1];
          Q_mn[2] += c * q[2];
          Q_mn[3] += c * q[3];
          Q_mn[4] += c * q[4];
          Q_mn[5] += c * q[5];
        }
      }

      S[mu * n + nu] = S_mn;
      S[nu * n + mu] = S_mn;
      for (int k = 0; k < 3; ++k) {
        dp[(k * n + mu) * n + nu] = D_mn[k];
        dp[(k * n + nu) * n + mu] = D_mn[k];
      }
      for (int k = 0; k < 6; ++k) {
        qp[(k * n + mu) * n + nu] = Q_mn[k];
        qp[(k * n + nu) * n + mu] = Q_mn[k];
      }
    }
  }

  Py_DECREF(centers);
  Py_DECREF(lxyz);
  Py_DECREF(offsets);
  Py_DECREF(alphas);
  Py_DECREF(coeffs);

  PyObject* out = PyTuple_New(3);
  PyTuple_SET_ITEM(out, 0, S_obj);
  PyTuple_SET_ITEM(out, 1, dp_obj);
  PyTuple_SET_ITEM(out, 2, qp_obj);
  return out;
}

PyObject* multipole_gradients_from_arrays(PyObject*, PyObject* args) {
  PyObject* centers_obj = nullptr;
  PyObject* lxyz_obj = nullptr;
  PyObject* offsets_obj = nullptr;
  PyObject* alphas_obj = nullptr;
  PyObject* coeffs_obj = nullptr;

  if (!PyArg_ParseTuple(
          args,
          "OOOOO",
          &centers_obj,
          &lxyz_obj,
          &offsets_obj,
          &alphas_obj,
          &coeffs_obj)) {
    return nullptr;
  }

  PyArrayObject* centers = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(centers_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* lxyz = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(lxyz_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* offsets = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(offsets_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* alphas = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(alphas_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* coeffs = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));

  if (!centers || !lxyz || !offsets || !alphas || !coeffs) {
    Py_XDECREF(centers);
    Py_XDECREF(lxyz);
    Py_XDECREF(offsets);
    Py_XDECREF(alphas);
    Py_XDECREF(coeffs);
    return nullptr;
  }

  const npy_intp n = PyArray_DIM(centers, 0);
  const auto* centers_data = static_cast<const double*>(PyArray_DATA(centers));
  const auto* lxyz_data = static_cast<const int32_t*>(PyArray_DATA(lxyz));
  const auto* offsets_data = static_cast<const npy_intp*>(PyArray_DATA(offsets));
  const auto* alphas_data = static_cast<const double*>(PyArray_DATA(alphas));
  const auto* coeffs_data = static_cast<const double*>(PyArray_DATA(coeffs));

  npy_intp d3_dims[4] = {3, 3, n, n};
  npy_intp d6_dims[4] = {3, 6, n, n};
  PyObject* dDA_obj = PyArray_ZEROS(4, d3_dims, NPY_DOUBLE, 0);
  PyObject* dDB_obj = PyArray_ZEROS(4, d3_dims, NPY_DOUBLE, 0);
  PyObject* dQA_obj = PyArray_ZEROS(4, d6_dims, NPY_DOUBLE, 0);
  PyObject* dQB_obj = PyArray_ZEROS(4, d6_dims, NPY_DOUBLE, 0);
  if (!dDA_obj || !dDB_obj || !dQA_obj || !dQB_obj) {
    Py_XDECREF(dDA_obj);
    Py_XDECREF(dDB_obj);
    Py_XDECREF(dQA_obj);
    Py_XDECREF(dQB_obj);
    Py_DECREF(centers);
    Py_DECREF(lxyz);
    Py_DECREF(offsets);
    Py_DECREF(alphas);
    Py_DECREF(coeffs);
    return nullptr;
  }

  auto* dDA = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(dDA_obj)));
  auto* dDB = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(dDB_obj)));
  auto* dQA = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(dQA_obj)));
  auto* dQB = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(dQB_obj)));

  for (npy_intp mu = 0; mu < n; ++mu) {
    const double* A = centers_data + 3 * mu;
    const int32_t* la = lxyz_data + 3 * mu;
    const npy_intp mu0 = offsets_data[mu];
    const npy_intp mu1 = offsets_data[mu + 1];

    for (npy_intp nu = 0; nu < n; ++nu) {
      const double* B = centers_data + 3 * nu;
      const int32_t* lb = lxyz_data + 3 * nu;
      const npy_intp nu0 = offsets_data[nu];
      const npy_intp nu1 = offsets_data[nu + 1];

      for (npy_intp i = mu0; i < mu1; ++i) {
        for (npy_intp j = nu0; j < nu1; ++j) {
          double pDA[3][3];
          double pDB[3][3];
          double pQA[3][6];
          double pQB[3][6];
          multipole_primitive_grad(
              alphas_data[i],
              A,
              la,
              alphas_data[j],
              B,
              lb,
              pDA,
              pDB,
              pQA,
              pQB);
          const double c = coeffs_data[i] * coeffs_data[j];
          for (int beta = 0; beta < 3; ++beta) {
            for (int k = 0; k < 3; ++k) {
              dDA[((beta * 3 + k) * n + mu) * n + nu] += c * pDA[beta][k];
              dDB[((beta * 3 + k) * n + mu) * n + nu] += c * pDB[beta][k];
            }
            for (int k = 0; k < 6; ++k) {
              dQA[((beta * 6 + k) * n + mu) * n + nu] += c * pQA[beta][k];
              dQB[((beta * 6 + k) * n + mu) * n + nu] += c * pQB[beta][k];
            }
          }
        }
      }
    }
  }

  Py_DECREF(centers);
  Py_DECREF(lxyz);
  Py_DECREF(offsets);
  Py_DECREF(alphas);
  Py_DECREF(coeffs);

  PyObject* out = PyTuple_New(4);
  PyTuple_SET_ITEM(out, 0, dDA_obj);
  PyTuple_SET_ITEM(out, 1, dDB_obj);
  PyTuple_SET_ITEM(out, 2, dQA_obj);
  PyTuple_SET_ITEM(out, 3, dQB_obj);
  return out;
}

PyObject* mmompop_chain_gradient(PyObject*, PyObject* args) {
  PyObject *P_obj, *S_obj, *dp_obj, *qp_obj, *aoat_obj, *coords_obj;
  PyObject *dSA_obj, *dSB_obj, *dDA_obj, *dDB_obj, *dQA_obj, *dQB_obj;
  PyObject *dEdip_obj, *dEqp_obj;

  if (!PyArg_ParseTuple(
          args,
          "OOOOOOOOOOOOOO",
          &P_obj,
          &S_obj,
          &dp_obj,
          &qp_obj,
          &aoat_obj,
          &coords_obj,
          &dSA_obj,
          &dSB_obj,
          &dDA_obj,
          &dDB_obj,
          &dQA_obj,
          &dQB_obj,
          &dEdip_obj,
          &dEqp_obj)) {
    return nullptr;
  }

  PyArrayObject* P = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(P_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* S = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(S_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* dp = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(dp_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* qp = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(qp_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* aoat = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(aoat_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* coords = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(coords_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* dSA = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(dSA_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* dSB = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(dSB_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* dDA = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(dDA_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* dDB = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(dDB_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* dQA = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(dQA_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* dQB = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(dQB_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* dEdip = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(dEdip_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* dEqp = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(dEqp_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));

  if (!P || !S || !dp || !qp || !aoat || !coords || !dSA || !dSB ||
      !dDA || !dDB || !dQA || !dQB || !dEdip || !dEqp) {
    Py_XDECREF(P); Py_XDECREF(S); Py_XDECREF(dp); Py_XDECREF(qp);
    Py_XDECREF(aoat); Py_XDECREF(coords); Py_XDECREF(dSA); Py_XDECREF(dSB);
    Py_XDECREF(dDA); Py_XDECREF(dDB); Py_XDECREF(dQA); Py_XDECREF(dQB);
    Py_XDECREF(dEdip); Py_XDECREF(dEqp);
    return nullptr;
  }

  const npy_intp nao = PyArray_DIM(S, 0);
  const npy_intp nat = PyArray_DIM(coords, 0);
  npy_intp gdims[2] = {nat, 3};
  PyObject* grad_obj = PyArray_ZEROS(2, gdims, NPY_DOUBLE, 0);
  if (!grad_obj) {
    Py_DECREF(P); Py_DECREF(S); Py_DECREF(dp); Py_DECREF(qp);
    Py_DECREF(aoat); Py_DECREF(coords); Py_DECREF(dSA); Py_DECREF(dSB);
    Py_DECREF(dDA); Py_DECREF(dDB); Py_DECREF(dQA); Py_DECREF(dQB);
    Py_DECREF(dEdip); Py_DECREF(dEqp);
    return nullptr;
  }

  const double* pP = static_cast<const double*>(PyArray_DATA(P));
  const double* pS = static_cast<const double*>(PyArray_DATA(S));
  const double* pDp = static_cast<const double*>(PyArray_DATA(dp));
  const npy_intp* pAo = static_cast<const npy_intp*>(PyArray_DATA(aoat));
  const double* pCoords = static_cast<const double*>(PyArray_DATA(coords));
  const double* pDSA = static_cast<const double*>(PyArray_DATA(dSA));
  const double* pDSB = static_cast<const double*>(PyArray_DATA(dSB));
  const double* pDDA = static_cast<const double*>(PyArray_DATA(dDA));
  const double* pDDB = static_cast<const double*>(PyArray_DATA(dDB));
  const double* pDQA = static_cast<const double*>(PyArray_DATA(dQA));
  const double* pDQB = static_cast<const double*>(PyArray_DATA(dQB));
  const double* pDEdip = static_cast<const double*>(PyArray_DATA(dEdip));
  const double* pDEqp = static_cast<const double*>(PyArray_DATA(dEqp));
  double* pGrad = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(grad_obj)));

  const auto mat = [nao](const double* a, npy_intp i, npy_intp j) -> double {
    return a[i * nao + j];
  };
  const auto arr3 = [nao](const double* a, int k, npy_intp i, npy_intp j) -> double {
    return a[(static_cast<npy_intp>(k) * nao + i) * nao + j];
  };
  const auto arr33 = [nao](const double* a, int beta, int k, npy_intp i, npy_intp j) -> double {
    return a[((static_cast<npy_intp>(beta) * 3 + k) * nao + i) * nao + j];
  };
  const auto arr36 = [nao](const double* a, int beta, int k, npy_intp i, npy_intp j) -> double {
    return a[((static_cast<npy_intp>(beta) * 6 + k) * nao + i) * nao + j];
  };

  std::vector<double> ddip(static_cast<size_t>(3 * nat), 0.0);
  std::vector<double> dqp_raw(static_cast<size_t>(6 * nat), 0.0);
  std::vector<double> dqp(static_cast<size_t>(6 * nat), 0.0);

  for (npy_intp atom = 0; atom < nat; ++atom) {
    for (int beta = 0; beta < 3; ++beta) {
      std::fill(ddip.begin(), ddip.end(), 0.0);
      std::fill(dqp_raw.begin(), dqp_raw.end(), 0.0);
      std::fill(dqp.begin(), dqp.end(), 0.0);

      for (npy_intp i = 0; i < nao; ++i) {
        const npy_intp ii = pAo[i];
        const double* ra = pCoords + 3 * ii;
        for (npy_intp j = 0; j < i; ++j) {
          const npy_intp jj = pAo[j];
          const double* rb = pCoords + 3 * jj;
          const double pij = mat(pP, j, i);
          const double ps = pij * mat(pS, j, i);

          double dS = 0.0;
          if (jj == atom) {
            dS += arr3(pDSA, beta, j, i);
          }
          if (ii == atom) {
            dS += arr3(pDSB, beta, j, i);
          }
          const double dps = pij * dS;

          double dD[3] = {0.0, 0.0, 0.0};
          double dQ[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
          for (int k = 0; k < 3; ++k) {
            if (jj == atom) {
              dD[k] += arr33(pDDA, beta, k, j, i);
            }
            if (ii == atom) {
              dD[k] += arr33(pDDB, beta, k, j, i);
            }
          }
          for (int k = 0; k < 6; ++k) {
            if (jj == atom) {
              dQ[k] += arr36(pDQA, beta, k, j, i);
            }
            if (ii == atom) {
              dQ[k] += arr36(pDQB, beta, k, j, i);
            }
          }

          for (int k = 0; k < 3; ++k) {
            const double pdmk = pij * arr3(pDp, k, j, i);
            const double dpdmk = pij * dD[k];
            ddip[k * nat + ii] +=
                ((ii == atom && k == beta) ? ps : 0.0) + ra[k] * dps - dpdmk;
            ddip[k * nat + jj] +=
                ((jj == atom && k == beta) ? ps : 0.0) + rb[k] * dps - dpdmk;

            for (int l = 0; l < k; ++l) {
              int qint_idx;
              int mm_idx;
              if (k == 1 && l == 0) {
                qint_idx = 3; mm_idx = 1;
              } else if (k == 2 && l == 0) {
                qint_idx = 4; mm_idx = 3;
              } else {
                qint_idx = 5; mm_idx = 4;
              }
              const double pdml = pij * arr3(pDp, l, j, i);
              const double dpdml = pij * dD[l];
              const double dpqm = pij * dQ[qint_idx];
              const double* rs[2] = {ra, rb};
              const npy_intp targets[2] = {ii, jj};
              for (int t = 0; t < 2; ++t) {
                const npy_intp target = targets[t];
                const double* r = rs[t];
                dqp_raw[mm_idx * nat + target] +=
                    dpdmk * r[l]
                    + pdmk * ((target == atom && l == beta) ? 1.0 : 0.0)
                    + dpdml * r[k]
                    + pdml * ((target == atom && k == beta) ? 1.0 : 0.0)
                    - r[l] * r[k] * dps
                    - (((target == atom && l == beta) ? r[k] : 0.0)
                       + ((target == atom && k == beta) ? r[l] : 0.0)) * ps
                    - dpqm;
              }
            }

            const int qint_idx = k;
            const int mm_idx = (k == 0) ? 0 : ((k == 1) ? 2 : 5);
            const double dpqm = pij * dQ[qint_idx];
            const double* rs[2] = {ra, rb};
            const npy_intp targets[2] = {ii, jj};
            for (int t = 0; t < 2; ++t) {
              const npy_intp target = targets[t];
              const double* r = rs[t];
              const double delta = (target == atom && k == beta) ? 1.0 : 0.0;
              dqp_raw[mm_idx * nat + target] +=
                  2.0 * dpdmk * r[k]
                  + 2.0 * pdmk * delta
                  - r[k] * r[k] * dps
                  - 2.0 * r[k] * delta * ps
                  - dpqm;
            }
          }
        }
      }

      for (npy_intp i = 0; i < nao; ++i) {
        const npy_intp ii = pAo[i];
        const double* ra = pCoords + 3 * ii;
        const double pij = mat(pP, i, i);
        const double ps = pij * mat(pS, i, i);

        double dS = 0.0;
        double dD[3] = {0.0, 0.0, 0.0};
        double dQ[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        if (ii == atom) {
          dS = arr3(pDSA, beta, i, i) + arr3(pDSB, beta, i, i);
          for (int k = 0; k < 3; ++k) {
            dD[k] = arr33(pDDA, beta, k, i, i) + arr33(pDDB, beta, k, i, i);
          }
          for (int k = 0; k < 6; ++k) {
            dQ[k] = arr36(pDQA, beta, k, i, i) + arr36(pDQB, beta, k, i, i);
          }
        }
        const double dps = pij * dS;

        for (int k = 0; k < 3; ++k) {
          const double pdmk = pij * arr3(pDp, k, i, i);
          const double dpdmk = pij * dD[k];
          const double delta_k = (ii == atom && k == beta) ? 1.0 : 0.0;
          ddip[k * nat + ii] += delta_k * ps + ra[k] * dps - dpdmk;

          for (int l = 0; l < k; ++l) {
            int qint_idx;
            int mm_idx;
            if (k == 1 && l == 0) {
              qint_idx = 3; mm_idx = 1;
            } else if (k == 2 && l == 0) {
              qint_idx = 4; mm_idx = 3;
            } else {
              qint_idx = 5; mm_idx = 4;
            }
            const double pdml = pij * arr3(pDp, l, i, i);
            const double dpdml = pij * dD[l];
            const double dpqm = pij * dQ[qint_idx];
            const double delta_l = (ii == atom && l == beta) ? 1.0 : 0.0;
            dqp_raw[mm_idx * nat + ii] +=
                dpdmk * ra[l]
                + pdmk * delta_l
                + dpdml * ra[k]
                + pdml * delta_k
                - ra[l] * ra[k] * dps
                - (delta_l * ra[k] + ra[l] * delta_k) * ps
                - dpqm;
          }

          const int qint_idx = k;
          const int mm_idx = (k == 0) ? 0 : ((k == 1) ? 2 : 5);
          const double dpqm = pij * dQ[qint_idx];
          dqp_raw[mm_idx * nat + ii] +=
              2.0 * dpdmk * ra[k]
              + 2.0 * pdmk * delta_k
              - ra[k] * ra[k] * dps
              - 2.0 * ra[k] * delta_k * ps
              - dpqm;
        }
      }

      for (npy_intp a = 0; a < nat; ++a) {
        for (int k = 0; k < 6; ++k) {
          dqp[k * nat + a] = 1.5 * dqp_raw[k * nat + a];
        }
        const double tr = 0.5 * (
            dqp_raw[0 * nat + a] + dqp_raw[2 * nat + a] + dqp_raw[5 * nat + a]);
        dqp[0 * nat + a] -= tr;
        dqp[2 * nat + a] -= tr;
        dqp[5 * nat + a] -= tr;
      }

      double value = 0.0;
      for (npy_intp a = 0; a < nat; ++a) {
        for (int k = 0; k < 3; ++k) {
          value += pDEdip[k * nat + a] * ddip[k * nat + a];
        }
        for (int k = 0; k < 6; ++k) {
          value += pDEqp[k * nat + a] * dqp[k * nat + a];
        }
      }
      pGrad[atom * 3 + beta] = value;
    }
  }

  Py_DECREF(P); Py_DECREF(S); Py_DECREF(dp); Py_DECREF(qp);
  Py_DECREF(aoat); Py_DECREF(coords); Py_DECREF(dSA); Py_DECREF(dSB);
  Py_DECREF(dDA); Py_DECREF(dDB); Py_DECREF(dQA); Py_DECREF(dQB);
  Py_DECREF(dEdip); Py_DECREF(dEqp);
  return grad_obj;
}

PyObject* mmompop_from_arrays(PyObject*, PyObject* args) {
  PyObject *P_obj, *S_obj, *dp_obj, *qp_obj, *aoat_obj, *coords_obj;

  if (!PyArg_ParseTuple(
          args,
          "OOOOOO",
          &P_obj,
          &S_obj,
          &dp_obj,
          &qp_obj,
          &aoat_obj,
          &coords_obj)) {
    return nullptr;
  }

  PyArrayObject* P = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(P_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* S = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(S_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* dp = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(dp_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* qp = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(qp_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* aoat = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(aoat_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* coords = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(coords_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));

  if (!P || !S || !dp || !qp || !aoat || !coords) {
    Py_XDECREF(P); Py_XDECREF(S); Py_XDECREF(dp); Py_XDECREF(qp);
    Py_XDECREF(aoat); Py_XDECREF(coords);
    return nullptr;
  }

  const auto fail = [&](const char* msg) -> PyObject* {
    PyErr_SetString(PyExc_ValueError, msg);
    Py_DECREF(P); Py_DECREF(S); Py_DECREF(dp); Py_DECREF(qp);
    Py_DECREF(aoat); Py_DECREF(coords);
    return nullptr;
  };

  if (PyArray_NDIM(S) != 2 || PyArray_DIM(S, 0) != PyArray_DIM(S, 1)) {
    return fail("S must have shape (nao, nao)");
  }
  const npy_intp nao = PyArray_DIM(S, 0);
  if (PyArray_NDIM(P) != 2 || PyArray_DIM(P, 0) != nao || PyArray_DIM(P, 1) != nao) {
    return fail("P must have shape (nao, nao)");
  }
  if (PyArray_NDIM(dp) != 3 || PyArray_DIM(dp, 0) != 3 ||
      PyArray_DIM(dp, 1) != nao || PyArray_DIM(dp, 2) != nao) {
    return fail("dpint must have shape (3, nao, nao)");
  }
  if (PyArray_NDIM(qp) != 3 || PyArray_DIM(qp, 0) != 6 ||
      PyArray_DIM(qp, 1) != nao || PyArray_DIM(qp, 2) != nao) {
    return fail("qpint must have shape (6, nao, nao)");
  }
  if (PyArray_NDIM(aoat) != 1 || PyArray_DIM(aoat, 0) != nao) {
    return fail("aoat must have shape (nao,)");
  }
  if (PyArray_NDIM(coords) != 2 || PyArray_DIM(coords, 1) != 3) {
    return fail("coords_bohr must have shape (nat, 3)");
  }
  const npy_intp nat = PyArray_DIM(coords, 0);

  npy_intp dip_dims[2] = {3, nat};
  npy_intp q_dims[2] = {6, nat};
  PyObject* dip_obj = PyArray_ZEROS(2, dip_dims, NPY_DOUBLE, 0);
  PyObject* qmom_obj = PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 0);
  if (!dip_obj || !qmom_obj) {
    Py_XDECREF(dip_obj);
    Py_XDECREF(qmom_obj);
    Py_DECREF(P); Py_DECREF(S); Py_DECREF(dp); Py_DECREF(qp);
    Py_DECREF(aoat); Py_DECREF(coords);
    return nullptr;
  }

  const double* pP = static_cast<const double*>(PyArray_DATA(P));
  const double* pS = static_cast<const double*>(PyArray_DATA(S));
  const double* pDp = static_cast<const double*>(PyArray_DATA(dp));
  const double* pQp = static_cast<const double*>(PyArray_DATA(qp));
  const npy_intp* pAo = static_cast<const npy_intp*>(PyArray_DATA(aoat));
  const double* pCoords = static_cast<const double*>(PyArray_DATA(coords));
  double* dip = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(dip_obj)));
  double* qmom = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(qmom_obj)));

  const auto mat = [nao](const double* a, npy_intp i, npy_intp j) -> double {
    return a[i * nao + j];
  };
  const auto arr3 = [nao](const double* a, int k, npy_intp i, npy_intp j) -> double {
    return a[(static_cast<npy_intp>(k) * nao + i) * nao + j];
  };

  for (npy_intp i = 0; i < nao; ++i) {
    const npy_intp ii = pAo[i];
    if (ii < 0 || ii >= nat) {
      Py_DECREF(dip_obj);
      Py_DECREF(qmom_obj);
      return fail("aoat contains an out-of-range atom index");
    }
    const double* ra = pCoords + 3 * ii;
    for (npy_intp j = 0; j < i; ++j) {
      const npy_intp jj = pAo[j];
      if (jj < 0 || jj >= nat) {
        Py_DECREF(dip_obj);
        Py_DECREF(qmom_obj);
        return fail("aoat contains an out-of-range atom index");
      }
      const double* rb = pCoords + 3 * jj;
      const double pij = mat(pP, j, i);
      const double ps = pij * mat(pS, j, i);

      for (int k = 0; k < 3; ++k) {
        const double xk1 = ra[k];
        const double xk2 = rb[k];
        const double pdmk = pij * arr3(pDp, k, j, i);
        dip[k * nat + ii] += xk1 * ps - pdmk;
        dip[k * nat + jj] += xk2 * ps - pdmk;

        for (int l = 0; l < k; ++l) {
          int qint_idx;
          int mm_idx;
          if (k == 1 && l == 0) {
            qint_idx = 3; mm_idx = 1;
          } else if (k == 2 && l == 0) {
            qint_idx = 4; mm_idx = 3;
          } else {
            qint_idx = 5; mm_idx = 4;
          }
          const double xl1 = ra[l];
          const double xl2 = rb[l];
          const double pdml = pij * arr3(pDp, l, j, i);
          const double pqm = pij * arr3(pQp, qint_idx, j, i);
          qmom[mm_idx * nat + ii] += pdmk * xl1 + pdml * xk1 - xl1 * xk1 * ps - pqm;
          qmom[mm_idx * nat + jj] += pdmk * xl2 + pdml * xk2 - xl2 * xk2 * ps - pqm;
        }

        const int mm_idx = (k == 0) ? 0 : ((k == 1) ? 2 : 5);
        const double pqm = pij * arr3(pQp, k, j, i);
        qmom[mm_idx * nat + ii] += 2.0 * pdmk * xk1 - xk1 * xk1 * ps - pqm;
        qmom[mm_idx * nat + jj] += 2.0 * pdmk * xk2 - xk2 * xk2 * ps - pqm;
      }
    }
  }

  for (npy_intp i = 0; i < nao; ++i) {
    const npy_intp ii = pAo[i];
    const double* ra = pCoords + 3 * ii;
    const double pij = mat(pP, i, i);
    const double ps = pij * mat(pS, i, i);

    for (int k = 0; k < 3; ++k) {
      const double xk1 = ra[k];
      const double pdmk = pij * arr3(pDp, k, i, i);
      dip[k * nat + ii] += xk1 * ps - pdmk;

      for (int l = 0; l < k; ++l) {
        int qint_idx;
        int mm_idx;
        if (k == 1 && l == 0) {
          qint_idx = 3; mm_idx = 1;
        } else if (k == 2 && l == 0) {
          qint_idx = 4; mm_idx = 3;
        } else {
          qint_idx = 5; mm_idx = 4;
        }
        const double xl1 = ra[l];
        const double pdml = pij * arr3(pDp, l, i, i);
        const double pqm = pij * arr3(pQp, qint_idx, i, i);
        qmom[mm_idx * nat + ii] += pdmk * xl1 + pdml * xk1 - xl1 * xk1 * ps - pqm;
      }

      const int mm_idx = (k == 0) ? 0 : ((k == 1) ? 2 : 5);
      const double pqm = pij * arr3(pQp, k, i, i);
      qmom[mm_idx * nat + ii] += 2.0 * pdmk * xk1 - xk1 * xk1 * ps - pqm;
    }
  }

  for (npy_intp atom = 0; atom < nat; ++atom) {
    const double tr = 0.5 * (
        qmom[0 * nat + atom] + qmom[2 * nat + atom] + qmom[5 * nat + atom]);
    for (int k = 0; k < 6; ++k) {
      qmom[k * nat + atom] *= 1.5;
    }
    qmom[0 * nat + atom] -= tr;
    qmom[2 * nat + atom] -= tr;
    qmom[5 * nat + atom] -= tr;
  }

  Py_DECREF(P); Py_DECREF(S); Py_DECREF(dp); Py_DECREF(qp);
  Py_DECREF(aoat); Py_DECREF(coords);

  PyObject* out = PyTuple_New(2);
  PyTuple_SET_ITEM(out, 0, dip_obj);
  PyTuple_SET_ITEM(out, 1, qmom_obj);
  return out;
}

PyObject* overlap_gradients_from_arrays(PyObject*, PyObject* args) {
  PyObject* centers_obj = nullptr;
  PyObject* lxyz_obj = nullptr;
  PyObject* offsets_obj = nullptr;
  PyObject* alphas_obj = nullptr;
  PyObject* coeffs_obj = nullptr;

  if (!PyArg_ParseTuple(
          args,
          "OOOOO",
          &centers_obj,
          &lxyz_obj,
          &offsets_obj,
          &alphas_obj,
          &coeffs_obj)) {
    return nullptr;
  }

  PyArrayObject* centers = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(centers_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* lxyz = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(lxyz_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* offsets = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(offsets_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* alphas = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(alphas_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* coeffs = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));

  if (!centers || !lxyz || !offsets || !alphas || !coeffs) {
    Py_XDECREF(centers);
    Py_XDECREF(lxyz);
    Py_XDECREF(offsets);
    Py_XDECREF(alphas);
    Py_XDECREF(coeffs);
    return nullptr;
  }

  const npy_intp n = PyArray_DIM(centers, 0);
  const auto* centers_data = static_cast<const double*>(PyArray_DATA(centers));
  const auto* lxyz_data = static_cast<const int32_t*>(PyArray_DATA(lxyz));
  const auto* offsets_data = static_cast<const npy_intp*>(PyArray_DATA(offsets));
  const auto* alphas_data = static_cast<const double*>(PyArray_DATA(alphas));
  const auto* coeffs_data = static_cast<const double*>(PyArray_DATA(coeffs));

  npy_intp dims[3] = {3, n, n};
  PyObject* dSA_obj = PyArray_ZEROS(3, dims, NPY_DOUBLE, 0);
  PyObject* dSB_obj = PyArray_ZEROS(3, dims, NPY_DOUBLE, 0);
  if (!dSA_obj || !dSB_obj) {
    Py_XDECREF(dSA_obj);
    Py_XDECREF(dSB_obj);
    Py_DECREF(centers);
    Py_DECREF(lxyz);
    Py_DECREF(offsets);
    Py_DECREF(alphas);
    Py_DECREF(coeffs);
    return nullptr;
  }

  auto* dSA = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(dSA_obj)));
  auto* dSB = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(dSB_obj)));

  for (npy_intp mu = 0; mu < n; ++mu) {
    const double* A = centers_data + 3 * mu;
    const int32_t* la = lxyz_data + 3 * mu;
    const npy_intp mu0 = offsets_data[mu];
    const npy_intp mu1 = offsets_data[mu + 1];

    for (npy_intp nu = 0; nu < n; ++nu) {
      if (mu == nu) {
        continue;
      }
      const double* B = centers_data + 3 * nu;
      const int32_t* lb = lxyz_data + 3 * nu;
      const npy_intp nu0 = offsets_data[nu];
      const npy_intp nu1 = offsets_data[nu + 1];

      for (npy_intp i = mu0; i < mu1; ++i) {
        for (npy_intp j = nu0; j < nu1; ++j) {
          double pDA[3];
          double pDB[3];
          primitive_overlap_grad(
              alphas_data[i],
              A,
              la,
              alphas_data[j],
              B,
              lb,
              pDA,
              pDB);
          const double c = coeffs_data[i] * coeffs_data[j];
          for (int beta = 0; beta < 3; ++beta) {
            dSA[(static_cast<npy_intp>(beta) * n + mu) * n + nu] += c * pDA[beta];
            dSB[(static_cast<npy_intp>(beta) * n + mu) * n + nu] += c * pDB[beta];
          }
        }
      }
    }
  }

  Py_DECREF(centers);
  Py_DECREF(lxyz);
  Py_DECREF(offsets);
  Py_DECREF(alphas);
  Py_DECREF(coeffs);

  PyObject* out = PyTuple_New(2);
  PyTuple_SET_ITEM(out, 0, dSA_obj);
  PyTuple_SET_ITEM(out, 1, dSB_obj);
  return out;
}

PyMethodDef methods[] = {
    {
        "multipole_matrices_from_arrays",
        multipole_matrices_from_arrays,
        METH_VARARGS,
        "Compute float64 CAO overlap/dipole/quadrupole multipole matrices.",
    },
    {
        "multipole_gradients_from_arrays",
        multipole_gradients_from_arrays,
        METH_VARARGS,
        "Compute float64 CAO dipole/quadrupole derivative tensors.",
    },
    {
        "mmompop_chain_gradient",
        mmompop_chain_gradient,
        METH_VARARGS,
        "Contract Mulliken multipole derivatives into a float64 AES gradient.",
    },
    {
        "mmompop_from_arrays",
        mmompop_from_arrays,
        METH_VARARGS,
        "Compute float64 Mulliken cumulative atomic multipole moments.",
    },
    {
        "overlap_gradients_from_arrays",
        overlap_gradients_from_arrays,
        METH_VARARGS,
        "Compute float64 CAO overlap derivative tensors.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_multipole_cpp",
    "Native float64 CPU multipole integral kernels.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__multipole_cpp() {
  import_array();
  return PyModule_Create(&module);
}
