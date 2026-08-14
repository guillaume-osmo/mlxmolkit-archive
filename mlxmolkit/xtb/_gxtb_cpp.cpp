#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <cmath>
#include <cstdint>

namespace {

struct PyArrayHolder {
  PyArrayObject* ptr = nullptr;

  PyArrayHolder() = default;
  explicit PyArrayHolder(PyArrayObject* value) : ptr(value) {}
  ~PyArrayHolder() { Py_XDECREF(ptr); }

  PyArrayHolder(const PyArrayHolder&) = delete;
  PyArrayHolder& operator=(const PyArrayHolder&) = delete;

  PyArrayObject* get() const { return ptr; }
  PyArrayObject* release() {
    PyArrayObject* out = ptr;
    ptr = nullptr;
    return out;
  }
};

PyArrayHolder as_array(PyObject* obj, int typenum, int flags) {
  return PyArrayHolder(reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(obj, typenum, flags)));
}

bool require_1d(PyArrayObject* arr, const char* name) {
  if (PyArray_NDIM(arr) != 1) {
    PyErr_Format(PyExc_ValueError, "%s must be a 1D array", name);
    return false;
  }
  return true;
}

bool require_2d(PyArrayObject* arr, const char* name) {
  if (PyArray_NDIM(arr) != 2) {
    PyErr_Format(PyExc_ValueError, "%s must be a 2D array", name);
    return false;
  }
  return true;
}

bool require_coords(PyArrayObject* arr, const char* name) {
  if (!require_2d(arr, name)) {
    return false;
  }
  if (PyArray_DIM(arr, 1) != 3) {
    PyErr_Format(PyExc_ValueError, "%s must have shape (n, 3)", name);
    return false;
  }
  return true;
}

bool require_same_length(PyArrayObject* a, PyArrayObject* b, const char* a_name, const char* b_name) {
  if (PyArray_DIM(a, 0) != PyArray_DIM(b, 0)) {
    PyErr_Format(
        PyExc_ValueError,
        "%s and %s must have the same length, got %zd and %zd",
        a_name,
        b_name,
        static_cast<Py_ssize_t>(PyArray_DIM(a, 0)),
        static_cast<Py_ssize_t>(PyArray_DIM(b, 0)));
    return false;
  }
  return true;
}

bool require_pair_matrix(PyArrayObject* arr, const char* name, npy_intp n) {
  if (!require_2d(arr, name)) {
    return false;
  }
  if (PyArray_DIM(arr, 0) != n || PyArray_DIM(arr, 1) != n) {
    PyErr_Format(
        PyExc_ValueError,
        "%s must have shape (%zd, %zd), got (%zd, %zd)",
        name,
        static_cast<Py_ssize_t>(n),
        static_cast<Py_ssize_t>(n),
        static_cast<Py_ssize_t>(PyArray_DIM(arr, 0)),
        static_cast<Py_ssize_t>(PyArray_DIM(arr, 1)));
    return false;
  }
  return true;
}

bool require_min_z_table(PyArrayObject* arr, const char* name, npy_intp max_z) {
  if (PyArray_DIM(arr, 0) < max_z) {
    PyErr_Format(
        PyExc_ValueError,
        "%s must contain at least max(Z) entries; got %zd need %zd",
        name,
        static_cast<Py_ssize_t>(PyArray_DIM(arr, 0)),
        static_cast<Py_ssize_t>(max_z));
    return false;
  }
  return true;
}

PyObject* scaled_zeff(PyObject*, PyObject* args) {
  PyObject* atomic_numbers_obj = nullptr;
  PyObject* zeff_by_z_obj = nullptr;
  PyObject* scale_by_z_obj = nullptr;
  PyObject* descriptor_obj = nullptr;
  if (!PyArg_ParseTuple(
          args,
          "OOOO",
          &atomic_numbers_obj,
          &zeff_by_z_obj,
          &scale_by_z_obj,
          &descriptor_obj)) {
    return nullptr;
  }

  PyArrayHolder atomic_numbers = as_array(atomic_numbers_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder zeff_by_z = as_array(zeff_by_z_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder scale_by_z = as_array(scale_by_z_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder descriptor = as_array(descriptor_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!atomic_numbers.get() || !zeff_by_z.get() || !scale_by_z.get() || !descriptor.get()) {
    return nullptr;
  }
  if (!require_1d(atomic_numbers.get(), "atomic_numbers") ||
      !require_1d(zeff_by_z.get(), "zeff_by_z") ||
      !require_1d(scale_by_z.get(), "scale_by_z") ||
      !require_1d(descriptor.get(), "descriptor") ||
      !require_same_length(atomic_numbers.get(), descriptor.get(), "atomic_numbers", "descriptor")) {
    return nullptr;
  }

  const npy_intp n = PyArray_DIM(atomic_numbers.get(), 0);
  const auto* z_data = static_cast<const npy_intp*>(PyArray_DATA(atomic_numbers.get()));
  npy_intp max_z = 0;
  for (npy_intp i = 0; i < n; ++i) {
    if (z_data[i] < 1) {
      PyErr_Format(PyExc_ValueError, "atomic_numbers must be >= 1; got %zd at index %zd",
                   static_cast<Py_ssize_t>(z_data[i]), static_cast<Py_ssize_t>(i));
      return nullptr;
    }
    if (z_data[i] > max_z) {
      max_z = z_data[i];
    }
  }
  if (!require_min_z_table(zeff_by_z.get(), "zeff_by_z", max_z) ||
      !require_min_z_table(scale_by_z.get(), "scale_by_z", max_z)) {
    return nullptr;
  }

  npy_intp dims[1] = {n};
  PyObject* out_obj = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
  if (!out_obj) {
    return nullptr;
  }
  auto* out = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out_obj)));
  const auto* zeff = static_cast<const double*>(PyArray_DATA(zeff_by_z.get()));
  const auto* scale = static_cast<const double*>(PyArray_DATA(scale_by_z.get()));
  const auto* desc = static_cast<const double*>(PyArray_DATA(descriptor.get()));
  for (npy_intp i = 0; i < n; ++i) {
    const npy_intp z = z_data[i] - 1;
    out[i] = zeff[z] * (1.0 - scale[z] * desc[i]);
  }
  return out_obj;
}

PyObject* cn_scaled_parameter(PyObject*, PyObject* args) {
  PyObject* atomic_numbers_obj = nullptr;
  PyObject* base_by_z_obj = nullptr;
  PyObject* slope_by_z_obj = nullptr;
  PyObject* cn_obj = nullptr;
  double eps2 = 1.0e-12;
  double eps = 1.0e-6;
  if (!PyArg_ParseTuple(
          args,
          "OOOO|dd",
          &atomic_numbers_obj,
          &base_by_z_obj,
          &slope_by_z_obj,
          &cn_obj,
          &eps2,
          &eps)) {
    return nullptr;
  }

  PyArrayHolder atomic_numbers = as_array(atomic_numbers_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder base_by_z = as_array(base_by_z_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder slope_by_z = as_array(slope_by_z_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder cn = as_array(cn_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!atomic_numbers.get() || !base_by_z.get() || !slope_by_z.get() || !cn.get()) {
    return nullptr;
  }
  if (!require_1d(atomic_numbers.get(), "atomic_numbers") ||
      !require_1d(base_by_z.get(), "base_by_z") ||
      !require_1d(slope_by_z.get(), "slope_by_z") ||
      !require_1d(cn.get(), "cn") ||
      !require_same_length(atomic_numbers.get(), cn.get(), "atomic_numbers", "cn")) {
    return nullptr;
  }

  const npy_intp n = PyArray_DIM(atomic_numbers.get(), 0);
  const auto* z_data = static_cast<const npy_intp*>(PyArray_DATA(atomic_numbers.get()));
  npy_intp max_z = 0;
  for (npy_intp i = 0; i < n; ++i) {
    if (z_data[i] < 1) {
      PyErr_Format(PyExc_ValueError, "atomic_numbers must be >= 1; got %zd at index %zd",
                   static_cast<Py_ssize_t>(z_data[i]), static_cast<Py_ssize_t>(i));
      return nullptr;
    }
    if (z_data[i] > max_z) {
      max_z = z_data[i];
    }
  }
  if (!require_min_z_table(base_by_z.get(), "base_by_z", max_z) ||
      !require_min_z_table(slope_by_z.get(), "slope_by_z", max_z)) {
    return nullptr;
  }

  npy_intp dims[1] = {n};
  PyObject* values_obj = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
  PyObject* derivs_obj = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
  if (!values_obj || !derivs_obj) {
    Py_XDECREF(values_obj);
    Py_XDECREF(derivs_obj);
    return nullptr;
  }

  auto* values = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(values_obj)));
  auto* derivs = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(derivs_obj)));
  const auto* base = static_cast<const double*>(PyArray_DATA(base_by_z.get()));
  const auto* slope = static_cast<const double*>(PyArray_DATA(slope_by_z.get()));
  const auto* cn_data = static_cast<const double*>(PyArray_DATA(cn.get()));

  for (npy_intp i = 0; i < n; ++i) {
    const npy_intp z = z_data[i] - 1;
    const double root = std::sqrt(cn_data[i] + eps2);
    values[i] = base[z] * (1.0 + slope[z] * (root - eps));
    derivs[i] = base[z] * slope[z] / (2.0 * root);
  }

  PyObject* out = PyTuple_New(2);
  if (!out) {
    Py_DECREF(values_obj);
    Py_DECREF(derivs_obj);
    return nullptr;
  }
  PyTuple_SET_ITEM(out, 0, values_obj);
  PyTuple_SET_ITEM(out, 1, derivs_obj);
  return out;
}

PyObject* repulsion_energy_from_matvec(PyObject*, PyObject* args) {
  PyObject* scaled_zeff_obj = nullptr;
  PyObject* matvec_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &scaled_zeff_obj, &matvec_obj)) {
    return nullptr;
  }

  PyArrayHolder scaled = as_array(scaled_zeff_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder matvec = as_array(matvec_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!scaled.get() || !matvec.get()) {
    return nullptr;
  }
  if (!require_1d(scaled.get(), "scaled_zeff") ||
      !require_1d(matvec.get(), "matvec") ||
      !require_same_length(scaled.get(), matvec.get(), "scaled_zeff", "matvec")) {
    return nullptr;
  }

  const npy_intp n = PyArray_DIM(scaled.get(), 0);
  const auto* s = static_cast<const double*>(PyArray_DATA(scaled.get()));
  const auto* m = static_cast<const double*>(PyArray_DATA(matvec.get()));
  double energy = 0.0;
  for (npy_intp i = 0; i < n; ++i) {
    energy += s[i] * m[i];
  }
  return PyFloat_FromDouble(energy);
}

PyObject* repulsion_descriptor_potential(PyObject*, PyObject* args) {
  PyObject* atomic_numbers_obj = nullptr;
  PyObject* base_by_z_obj = nullptr;
  PyObject* scale_by_z_obj = nullptr;
  PyObject* matvec_obj = nullptr;
  if (!PyArg_ParseTuple(
          args,
          "OOOO",
          &atomic_numbers_obj,
          &base_by_z_obj,
          &scale_by_z_obj,
          &matvec_obj)) {
    return nullptr;
  }

  PyArrayHolder atomic_numbers = as_array(atomic_numbers_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder base_by_z = as_array(base_by_z_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder scale_by_z = as_array(scale_by_z_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder matvec = as_array(matvec_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!atomic_numbers.get() || !base_by_z.get() || !scale_by_z.get() || !matvec.get()) {
    return nullptr;
  }
  if (!require_1d(atomic_numbers.get(), "atomic_numbers") ||
      !require_1d(base_by_z.get(), "base_by_z") ||
      !require_1d(scale_by_z.get(), "scale_by_z") ||
      !require_1d(matvec.get(), "matvec") ||
      !require_same_length(atomic_numbers.get(), matvec.get(), "atomic_numbers", "matvec")) {
    return nullptr;
  }

  const npy_intp n = PyArray_DIM(atomic_numbers.get(), 0);
  const auto* z_data = static_cast<const npy_intp*>(PyArray_DATA(atomic_numbers.get()));
  npy_intp max_z = 0;
  for (npy_intp i = 0; i < n; ++i) {
    if (z_data[i] < 1) {
      PyErr_Format(PyExc_ValueError, "atomic_numbers must be >= 1; got %zd at index %zd",
                   static_cast<Py_ssize_t>(z_data[i]), static_cast<Py_ssize_t>(i));
      return nullptr;
    }
    if (z_data[i] > max_z) {
      max_z = z_data[i];
    }
  }
  if (!require_min_z_table(base_by_z.get(), "base_by_z", max_z) ||
      !require_min_z_table(scale_by_z.get(), "scale_by_z", max_z)) {
    return nullptr;
  }

  npy_intp dims[1] = {n};
  PyObject* out_obj = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
  if (!out_obj) {
    return nullptr;
  }
  auto* out = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out_obj)));
  const auto* base = static_cast<const double*>(PyArray_DATA(base_by_z.get()));
  const auto* scale = static_cast<const double*>(PyArray_DATA(scale_by_z.get()));
  const auto* mv = static_cast<const double*>(PyArray_DATA(matvec.get()));
  for (npy_intp i = 0; i < n; ++i) {
    const npy_intp z = z_data[i] - 1;
    out[i] = -base[z] * scale[z] * mv[i];
  }
  return out_obj;
}

double repulsion_pair_value_impl(
    double r,
    double alpha_a,
    double alpha_b,
    double roffset,
    const double* coeffs,
    double exp_power_1,
    double exp_power_2,
    double exp2_scale,
    double exp2_weight,
    double* deriv) {
  const double inv_r = 1.0 / r;
  const double inv_r2 = inv_r * inv_r;
  const double inv_r3 = inv_r2 * inv_r;
  const double inv_r4 = inv_r2 * inv_r2;
  const double poly =
      1.0 + coeffs[0] * inv_r + coeffs[1] * inv_r2 + coeffs[2] * inv_r3 + coeffs[3] * inv_r4;
  const double alpha = (alpha_a * alpha_b) / (alpha_a + alpha_b);
  const double rho = r + roffset;
  const double rho_p1 = std::pow(rho, exp_power_1);
  const double rho_p2 = std::pow(rho, exp_power_2);
  const double exp1 = std::exp(-alpha * rho_p1);
  const double exp2 = std::exp(-alpha * exp2_scale * rho_p2);
  const double damp = exp1 + exp2_weight * exp2;
  const double value = poly * damp;

  if (deriv != nullptr) {
    const double dpoly =
        -coeffs[0] * inv_r2
        - 2.0 * coeffs[1] * inv_r3
        - 3.0 * coeffs[2] * inv_r4
        - 4.0 * coeffs[3] * inv_r4 * inv_r;
    const double dexp1 =
        exp1 * (-alpha * exp_power_1 * std::pow(rho, exp_power_1 - 1.0));
    const double dexp2 =
        exp2 * (-alpha * exp2_scale * exp_power_2 * std::pow(rho, exp_power_2 - 1.0));
    *deriv = dpoly * damp + poly * (dexp1 + exp2_weight * dexp2);
  }
  return value;
}

double repulsion_pair_value_asm_impl(
    double r,
    double alpha_a,
    double alpha_b,
    double pair_rvdw,
    double roffset,
    double linear_coeff,
    double quadratic_coeff,
    double cubic_coeff,
    double quartic_coeff,
    double exp_power_1,
    double exp_power_2,
    double exp2_scale,
    double exp2_weight,
    double* deriv) {
  const double inv_r = 1.0 / r;
  const double x = pair_rvdw * inv_r;
  const double x2 = x * x;
  const double x3 = x2 * x;
  const double x4 = x2 * x2;
  const double poly =
      1.0 + linear_coeff * inv_r + quadratic_coeff * x2 + cubic_coeff * x3 + quartic_coeff * x4;
  const double alpha = (alpha_a * alpha_b) / (alpha_a + alpha_b);
  const double rho = r + roffset;
  const double rho_p1 = std::pow(rho, exp_power_1);
  const double rho_p2 = std::pow(rho, exp_power_2);
  const double exp1 = std::exp(-alpha * rho_p1);
  const double exp2 = std::exp(-alpha * exp2_scale * rho_p2);
  const double damp = exp1 + exp2_weight * exp2;
  const double value = poly * damp;

  if (deriv != nullptr) {
    const double dpoly =
        -linear_coeff * inv_r * inv_r
        - 2.0 * quadratic_coeff * x2 * inv_r
        - 3.0 * cubic_coeff * x3 * inv_r
        - 4.0 * quartic_coeff * x4 * inv_r;
    const double dexp1 =
        exp1 * (-alpha * exp_power_1 * std::pow(rho, exp_power_1 - 1.0));
    const double dexp2 =
        exp2 * (-alpha * exp2_scale * exp_power_2 * std::pow(rho, exp_power_2 - 1.0));
    *deriv = dpoly * damp + poly * (dexp1 + exp2_weight * dexp2);
  }
  return value;
}

PyObject* repulsion_pair_value(PyObject*, PyObject* args) {
  double r = 0.0;
  double alpha_a = 0.0;
  double alpha_b = 0.0;
  double roffset = 0.0;
  PyObject* coeffs_obj = nullptr;
  double exp_power_1 = 1.0;
  double exp_power_2 = 1.0;
  double exp2_scale = 1.0;
  double exp2_weight = 0.0;
  if (!PyArg_ParseTuple(
          args,
          "ddddOdddd",
          &r,
          &alpha_a,
          &alpha_b,
          &roffset,
          &coeffs_obj,
          &exp_power_1,
          &exp_power_2,
          &exp2_scale,
          &exp2_weight)) {
    return nullptr;
  }

  PyArrayHolder coeffs = as_array(coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!coeffs.get()) {
    return nullptr;
  }
  if (!require_1d(coeffs.get(), "coeffs") || PyArray_DIM(coeffs.get(), 0) != 4) {
    PyErr_SetString(PyExc_ValueError, "coeffs must be a 1D array of length 4");
    return nullptr;
  }
  if (!(r > 0.0) || !(alpha_a > 0.0) || !(alpha_b > 0.0) || !(r + roffset > 0.0)) {
    PyErr_SetString(PyExc_ValueError, "r, alpha_a, alpha_b, and r + roffset must be positive");
    return nullptr;
  }

  const double* coeff_data = static_cast<const double*>(PyArray_DATA(coeffs.get()));
  return PyFloat_FromDouble(repulsion_pair_value_impl(
      r,
      alpha_a,
      alpha_b,
      roffset,
      coeff_data,
      exp_power_1,
      exp_power_2,
      exp2_scale,
      exp2_weight,
      nullptr));
}

PyObject* repulsion_pair_value_deriv(PyObject*, PyObject* args) {
  double r = 0.0;
  double alpha_a = 0.0;
  double alpha_b = 0.0;
  double roffset = 0.0;
  PyObject* coeffs_obj = nullptr;
  double exp_power_1 = 1.0;
  double exp_power_2 = 1.0;
  double exp2_scale = 1.0;
  double exp2_weight = 0.0;
  if (!PyArg_ParseTuple(
          args,
          "ddddOdddd",
          &r,
          &alpha_a,
          &alpha_b,
          &roffset,
          &coeffs_obj,
          &exp_power_1,
          &exp_power_2,
          &exp2_scale,
          &exp2_weight)) {
    return nullptr;
  }

  PyArrayHolder coeffs = as_array(coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!coeffs.get()) {
    return nullptr;
  }
  if (!require_1d(coeffs.get(), "coeffs") || PyArray_DIM(coeffs.get(), 0) != 4) {
    PyErr_SetString(PyExc_ValueError, "coeffs must be a 1D array of length 4");
    return nullptr;
  }
  if (!(r > 0.0) || !(alpha_a > 0.0) || !(alpha_b > 0.0) || !(r + roffset > 0.0)) {
    PyErr_SetString(PyExc_ValueError, "r, alpha_a, alpha_b, and r + roffset must be positive");
    return nullptr;
  }

  double deriv = 0.0;
  const double* coeff_data = static_cast<const double*>(PyArray_DATA(coeffs.get()));
  const double value = repulsion_pair_value_impl(
      r,
      alpha_a,
      alpha_b,
      roffset,
      coeff_data,
      exp_power_1,
      exp_power_2,
      exp2_scale,
      exp2_weight,
      &deriv);
  return Py_BuildValue("(dd)", value, deriv);
}

PyObject* repulsion_pair_value_asm(PyObject*, PyObject* args) {
  double r = 0.0;
  double alpha_a = 0.0;
  double alpha_b = 0.0;
  double pair_rvdw = 0.0;
  double roffset = 0.0;
  double linear_coeff = 0.0;
  double quadratic_coeff = 0.0;
  double cubic_coeff = 0.0;
  double quartic_coeff = 0.0;
  double exp_power_1 = 1.0;
  double exp_power_2 = 1.0;
  double exp2_scale = 1.0;
  double exp2_weight = 0.0;
  if (!PyArg_ParseTuple(
          args,
          "ddddddddddddd",
          &r,
          &alpha_a,
          &alpha_b,
          &pair_rvdw,
          &roffset,
          &linear_coeff,
          &quadratic_coeff,
          &cubic_coeff,
          &quartic_coeff,
          &exp_power_1,
          &exp_power_2,
          &exp2_scale,
          &exp2_weight)) {
    return nullptr;
  }
  if (!(r > 0.0) || !(alpha_a > 0.0) || !(alpha_b > 0.0) || !(pair_rvdw > 0.0) ||
      !(r + roffset > 0.0)) {
    PyErr_SetString(
        PyExc_ValueError,
        "r, alpha_a, alpha_b, pair_rvdw, and r + roffset must be positive");
    return nullptr;
  }

  return PyFloat_FromDouble(repulsion_pair_value_asm_impl(
      r,
      alpha_a,
      alpha_b,
      pair_rvdw,
      roffset,
      linear_coeff,
      quadratic_coeff,
      cubic_coeff,
      quartic_coeff,
      exp_power_1,
      exp_power_2,
      exp2_scale,
      exp2_weight,
      nullptr));
}

PyObject* repulsion_pair_value_asm_deriv(PyObject*, PyObject* args) {
  double r = 0.0;
  double alpha_a = 0.0;
  double alpha_b = 0.0;
  double pair_rvdw = 0.0;
  double roffset = 0.0;
  double linear_coeff = 0.0;
  double quadratic_coeff = 0.0;
  double cubic_coeff = 0.0;
  double quartic_coeff = 0.0;
  double exp_power_1 = 1.0;
  double exp_power_2 = 1.0;
  double exp2_scale = 1.0;
  double exp2_weight = 0.0;
  if (!PyArg_ParseTuple(
          args,
          "ddddddddddddd",
          &r,
          &alpha_a,
          &alpha_b,
          &pair_rvdw,
          &roffset,
          &linear_coeff,
          &quadratic_coeff,
          &cubic_coeff,
          &quartic_coeff,
          &exp_power_1,
          &exp_power_2,
          &exp2_scale,
          &exp2_weight)) {
    return nullptr;
  }
  if (!(r > 0.0) || !(alpha_a > 0.0) || !(alpha_b > 0.0) || !(pair_rvdw > 0.0) ||
      !(r + roffset > 0.0)) {
    PyErr_SetString(
        PyExc_ValueError,
        "r, alpha_a, alpha_b, pair_rvdw, and r + roffset must be positive");
    return nullptr;
  }

  double deriv = 0.0;
  const double value = repulsion_pair_value_asm_impl(
      r,
      alpha_a,
      alpha_b,
      pair_rvdw,
      roffset,
      linear_coeff,
      quadratic_coeff,
      cubic_coeff,
      quartic_coeff,
      exp_power_1,
      exp_power_2,
      exp2_scale,
      exp2_weight,
      &deriv);
  return Py_BuildValue("(dd)", value, deriv);
}

PyObject* repulsion_pair_matrix(PyObject*, PyObject* args) {
  PyObject* coords_obj = nullptr;
  PyObject* alpha_obj = nullptr;
  PyObject* pair_roffset_obj = nullptr;
  PyObject* coeffs_obj = nullptr;
  double exp_power_1 = 1.0;
  double exp_power_2 = 1.0;
  double exp2_scale = 1.0;
  double exp2_weight = 0.0;
  double cutoff = 25.0;
  if (!PyArg_ParseTuple(
          args,
          "OOOOddddd",
          &coords_obj,
          &alpha_obj,
          &pair_roffset_obj,
          &coeffs_obj,
          &exp_power_1,
          &exp_power_2,
          &exp2_scale,
          &exp2_weight,
          &cutoff)) {
    return nullptr;
  }

  PyArrayHolder coords = as_array(coords_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder alpha = as_array(alpha_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder pair_roffset = as_array(pair_roffset_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder coeffs = as_array(coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!coords.get() || !alpha.get() || !pair_roffset.get() || !coeffs.get()) {
    return nullptr;
  }
  if (!require_coords(coords.get(), "coords") ||
      !require_1d(alpha.get(), "alpha") ||
      !require_1d(coeffs.get(), "coeffs")) {
    return nullptr;
  }

  const npy_intp n = PyArray_DIM(coords.get(), 0);
  if (PyArray_DIM(alpha.get(), 0) != n) {
    PyErr_SetString(PyExc_ValueError, "alpha must have length n");
    return nullptr;
  }
  if (!require_pair_matrix(pair_roffset.get(), "pair_roffset", n)) {
    return nullptr;
  }
  if (PyArray_DIM(coeffs.get(), 0) != 4) {
    PyErr_SetString(PyExc_ValueError, "coeffs must be a 1D array of length 4");
    return nullptr;
  }

  npy_intp dims[2] = {n, n};
  PyObject* mat_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
  if (!mat_obj) {
    return nullptr;
  }
  auto* mat = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(mat_obj)));
  const auto* xyz = static_cast<const double*>(PyArray_DATA(coords.get()));
  const auto* alpha_data = static_cast<const double*>(PyArray_DATA(alpha.get()));
  const auto* roffset = static_cast<const double*>(PyArray_DATA(pair_roffset.get()));
  const auto* coeff_data = static_cast<const double*>(PyArray_DATA(coeffs.get()));

  for (npy_intp i = 0; i < n; ++i) {
    for (npy_intp j = i + 1; j < n; ++j) {
      const double dx = xyz[3 * i] - xyz[3 * j];
      const double dy = xyz[3 * i + 1] - xyz[3 * j + 1];
      const double dz = xyz[3 * i + 2] - xyz[3 * j + 2];
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (r <= 0.0 || r > cutoff) {
        continue;
      }
      const double value = repulsion_pair_value_impl(
          r,
          alpha_data[i],
          alpha_data[j],
          roffset[i * n + j],
          coeff_data,
          exp_power_1,
          exp_power_2,
          exp2_scale,
          exp2_weight,
          nullptr);
      mat[i * n + j] = value;
      mat[j * n + i] = value;
    }
  }
  return mat_obj;
}

PyObject* repulsion_energy_gradient(PyObject*, PyObject* args) {
  PyObject* coords_obj = nullptr;
  PyObject* scaled_obj = nullptr;
  PyObject* alpha_obj = nullptr;
  PyObject* pair_roffset_obj = nullptr;
  PyObject* coeffs_obj = nullptr;
  double exp_power_1 = 1.0;
  double exp_power_2 = 1.0;
  double exp2_scale = 1.0;
  double exp2_weight = 0.0;
  double cutoff = 25.0;
  if (!PyArg_ParseTuple(
          args,
          "OOOOOddddd",
          &coords_obj,
          &scaled_obj,
          &alpha_obj,
          &pair_roffset_obj,
          &coeffs_obj,
          &exp_power_1,
          &exp_power_2,
          &exp2_scale,
          &exp2_weight,
          &cutoff)) {
    return nullptr;
  }

  PyArrayHolder coords = as_array(coords_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder scaled = as_array(scaled_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder alpha = as_array(alpha_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder pair_roffset = as_array(pair_roffset_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder coeffs = as_array(coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!coords.get() || !scaled.get() || !alpha.get() || !pair_roffset.get() || !coeffs.get()) {
    return nullptr;
  }
  if (!require_coords(coords.get(), "coords") ||
      !require_1d(scaled.get(), "scaled") ||
      !require_1d(alpha.get(), "alpha") ||
      !require_1d(coeffs.get(), "coeffs")) {
    return nullptr;
  }

  const npy_intp n = PyArray_DIM(coords.get(), 0);
  if (PyArray_DIM(scaled.get(), 0) != n || PyArray_DIM(alpha.get(), 0) != n) {
    PyErr_SetString(PyExc_ValueError, "scaled and alpha must have length n");
    return nullptr;
  }
  if (!require_pair_matrix(pair_roffset.get(), "pair_roffset", n)) {
    return nullptr;
  }
  if (PyArray_DIM(coeffs.get(), 0) != 4) {
    PyErr_SetString(PyExc_ValueError, "coeffs must be a 1D array of length 4");
    return nullptr;
  }

  npy_intp grad_dims[2] = {n, 3};
  npy_intp mv_dims[1] = {n};
  PyObject* grad_obj = PyArray_ZEROS(2, grad_dims, NPY_DOUBLE, 0);
  PyObject* matvec_obj = PyArray_ZEROS(1, mv_dims, NPY_DOUBLE, 0);
  if (!grad_obj || !matvec_obj) {
    Py_XDECREF(grad_obj);
    Py_XDECREF(matvec_obj);
    return nullptr;
  }

  auto* grad = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(grad_obj)));
  auto* matvec = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(matvec_obj)));
  const auto* xyz = static_cast<const double*>(PyArray_DATA(coords.get()));
  const auto* scaled_data = static_cast<const double*>(PyArray_DATA(scaled.get()));
  const auto* alpha_data = static_cast<const double*>(PyArray_DATA(alpha.get()));
  const auto* roffset = static_cast<const double*>(PyArray_DATA(pair_roffset.get()));
  const auto* coeff_data = static_cast<const double*>(PyArray_DATA(coeffs.get()));

  double energy = 0.0;
  for (npy_intp i = 0; i < n; ++i) {
    for (npy_intp j = i + 1; j < n; ++j) {
      const double dx = xyz[3 * i] - xyz[3 * j];
      const double dy = xyz[3 * i + 1] - xyz[3 * j + 1];
      const double dz = xyz[3 * i + 2] - xyz[3 * j + 2];
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (r <= 0.0 || r > cutoff) {
        continue;
      }

      double dvalue_dr = 0.0;
      const double value = repulsion_pair_value_impl(
          r,
          alpha_data[i],
          alpha_data[j],
          roffset[i * n + j],
          coeff_data,
          exp_power_1,
          exp_power_2,
          exp2_scale,
          exp2_weight,
          &dvalue_dr);
      matvec[i] += value * scaled_data[j];
      matvec[j] += value * scaled_data[i];
      const double pair_scale = 2.0 * scaled_data[i] * scaled_data[j];
      energy += pair_scale * value;
      const double pref = pair_scale * dvalue_dr / r;
      grad[3 * i] += pref * dx;
      grad[3 * i + 1] += pref * dy;
      grad[3 * i + 2] += pref * dz;
      grad[3 * j] -= pref * dx;
      grad[3 * j + 1] -= pref * dy;
      grad[3 * j + 2] -= pref * dz;
    }
  }

  PyObject* out = PyTuple_New(3);
  if (!out) {
    Py_DECREF(grad_obj);
    Py_DECREF(matvec_obj);
    return nullptr;
  }
  PyTuple_SET_ITEM(out, 0, PyFloat_FromDouble(energy));
  PyTuple_SET_ITEM(out, 1, grad_obj);
  PyTuple_SET_ITEM(out, 2, matvec_obj);
  return out;
}

PyObject* repulsion_pair_matrix_asm(PyObject*, PyObject* args) {
  PyObject* coords_obj = nullptr;
  PyObject* alpha_obj = nullptr;
  PyObject* pair_rvdw_obj = nullptr;
  PyObject* pair_roffset_obj = nullptr;
  PyObject* linear_coeff_obj = nullptr;
  PyObject* quadratic_coeff_obj = nullptr;
  double cubic_coeff = 0.0;
  double quartic_coeff = 0.0;
  double exp_power_1 = 1.0;
  double exp_power_2 = 1.0;
  double exp2_scale = 1.0;
  double exp2_weight = 0.0;
  double cutoff = 25.0;
  if (!PyArg_ParseTuple(
          args,
          "OOOOOOddddddd",
          &coords_obj,
          &alpha_obj,
          &pair_rvdw_obj,
          &pair_roffset_obj,
          &linear_coeff_obj,
          &quadratic_coeff_obj,
          &cubic_coeff,
          &quartic_coeff,
          &exp_power_1,
          &exp_power_2,
          &exp2_scale,
          &exp2_weight,
          &cutoff)) {
    return nullptr;
  }

  PyArrayHolder coords = as_array(coords_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder alpha = as_array(alpha_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder pair_rvdw = as_array(pair_rvdw_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder pair_roffset = as_array(pair_roffset_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder linear_coeff = as_array(linear_coeff_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder quadratic_coeff = as_array(quadratic_coeff_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!coords.get() || !alpha.get() || !pair_rvdw.get() || !pair_roffset.get() ||
      !linear_coeff.get() || !quadratic_coeff.get()) {
    return nullptr;
  }
  if (!require_coords(coords.get(), "coords") || !require_1d(alpha.get(), "alpha")) {
    return nullptr;
  }

  const npy_intp n = PyArray_DIM(coords.get(), 0);
  if (PyArray_DIM(alpha.get(), 0) != n) {
    PyErr_SetString(PyExc_ValueError, "alpha must have length n");
    return nullptr;
  }
  if (!require_pair_matrix(pair_rvdw.get(), "pair_rvdw", n) ||
      !require_pair_matrix(pair_roffset.get(), "pair_roffset", n) ||
      !require_pair_matrix(linear_coeff.get(), "linear_coeff", n) ||
      !require_pair_matrix(quadratic_coeff.get(), "quadratic_coeff", n)) {
    return nullptr;
  }

  npy_intp dims[2] = {n, n};
  PyObject* mat_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
  if (!mat_obj) {
    return nullptr;
  }
  auto* mat = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(mat_obj)));
  const auto* xyz = static_cast<const double*>(PyArray_DATA(coords.get()));
  const auto* alpha_data = static_cast<const double*>(PyArray_DATA(alpha.get()));
  const auto* rvdw = static_cast<const double*>(PyArray_DATA(pair_rvdw.get()));
  const auto* roffset = static_cast<const double*>(PyArray_DATA(pair_roffset.get()));
  const auto* lin = static_cast<const double*>(PyArray_DATA(linear_coeff.get()));
  const auto* quad = static_cast<const double*>(PyArray_DATA(quadratic_coeff.get()));

  for (npy_intp i = 0; i < n; ++i) {
    for (npy_intp j = i + 1; j < n; ++j) {
      const double dx = xyz[3 * i] - xyz[3 * j];
      const double dy = xyz[3 * i + 1] - xyz[3 * j + 1];
      const double dz = xyz[3 * i + 2] - xyz[3 * j + 2];
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (r <= 0.0 || r > cutoff) {
        continue;
      }
      const npy_intp ij = i * n + j;
      const double value = repulsion_pair_value_asm_impl(
          r,
          alpha_data[i],
          alpha_data[j],
          rvdw[ij],
          roffset[ij],
          lin[ij],
          quad[ij],
          cubic_coeff,
          quartic_coeff,
          exp_power_1,
          exp_power_2,
          exp2_scale,
          exp2_weight,
          nullptr);
      mat[ij] = value;
      mat[j * n + i] = value;
    }
  }
  return mat_obj;
}

PyObject* repulsion_energy_gradient_asm(PyObject*, PyObject* args) {
  PyObject* coords_obj = nullptr;
  PyObject* scaled_obj = nullptr;
  PyObject* alpha_obj = nullptr;
  PyObject* pair_rvdw_obj = nullptr;
  PyObject* pair_roffset_obj = nullptr;
  PyObject* linear_coeff_obj = nullptr;
  PyObject* quadratic_coeff_obj = nullptr;
  double cubic_coeff = 0.0;
  double quartic_coeff = 0.0;
  double exp_power_1 = 1.0;
  double exp_power_2 = 1.0;
  double exp2_scale = 1.0;
  double exp2_weight = 0.0;
  double cutoff = 25.0;
  if (!PyArg_ParseTuple(
          args,
          "OOOOOOOddddddd",
          &coords_obj,
          &scaled_obj,
          &alpha_obj,
          &pair_rvdw_obj,
          &pair_roffset_obj,
          &linear_coeff_obj,
          &quadratic_coeff_obj,
          &cubic_coeff,
          &quartic_coeff,
          &exp_power_1,
          &exp_power_2,
          &exp2_scale,
          &exp2_weight,
          &cutoff)) {
    return nullptr;
  }

  PyArrayHolder coords = as_array(coords_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder scaled = as_array(scaled_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder alpha = as_array(alpha_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder pair_rvdw = as_array(pair_rvdw_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder pair_roffset = as_array(pair_roffset_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder linear_coeff = as_array(linear_coeff_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder quadratic_coeff = as_array(quadratic_coeff_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!coords.get() || !scaled.get() || !alpha.get() || !pair_rvdw.get() ||
      !pair_roffset.get() || !linear_coeff.get() || !quadratic_coeff.get()) {
    return nullptr;
  }
  if (!require_coords(coords.get(), "coords") ||
      !require_1d(scaled.get(), "scaled") ||
      !require_1d(alpha.get(), "alpha")) {
    return nullptr;
  }

  const npy_intp n = PyArray_DIM(coords.get(), 0);
  if (PyArray_DIM(scaled.get(), 0) != n || PyArray_DIM(alpha.get(), 0) != n) {
    PyErr_SetString(PyExc_ValueError, "scaled and alpha must have length n");
    return nullptr;
  }
  if (!require_pair_matrix(pair_rvdw.get(), "pair_rvdw", n) ||
      !require_pair_matrix(pair_roffset.get(), "pair_roffset", n) ||
      !require_pair_matrix(linear_coeff.get(), "linear_coeff", n) ||
      !require_pair_matrix(quadratic_coeff.get(), "quadratic_coeff", n)) {
    return nullptr;
  }

  npy_intp grad_dims[2] = {n, 3};
  npy_intp mv_dims[1] = {n};
  PyObject* grad_obj = PyArray_ZEROS(2, grad_dims, NPY_DOUBLE, 0);
  PyObject* matvec_obj = PyArray_ZEROS(1, mv_dims, NPY_DOUBLE, 0);
  if (!grad_obj || !matvec_obj) {
    Py_XDECREF(grad_obj);
    Py_XDECREF(matvec_obj);
    return nullptr;
  }

  auto* grad = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(grad_obj)));
  auto* matvec = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(matvec_obj)));
  const auto* xyz = static_cast<const double*>(PyArray_DATA(coords.get()));
  const auto* scaled_data = static_cast<const double*>(PyArray_DATA(scaled.get()));
  const auto* alpha_data = static_cast<const double*>(PyArray_DATA(alpha.get()));
  const auto* rvdw = static_cast<const double*>(PyArray_DATA(pair_rvdw.get()));
  const auto* roffset = static_cast<const double*>(PyArray_DATA(pair_roffset.get()));
  const auto* lin = static_cast<const double*>(PyArray_DATA(linear_coeff.get()));
  const auto* quad = static_cast<const double*>(PyArray_DATA(quadratic_coeff.get()));

  double energy = 0.0;
  for (npy_intp i = 0; i < n; ++i) {
    for (npy_intp j = i + 1; j < n; ++j) {
      const double dx = xyz[3 * i] - xyz[3 * j];
      const double dy = xyz[3 * i + 1] - xyz[3 * j + 1];
      const double dz = xyz[3 * i + 2] - xyz[3 * j + 2];
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (r <= 0.0 || r > cutoff) {
        continue;
      }

      const npy_intp ij = i * n + j;
      double dvalue_dr = 0.0;
      const double value = repulsion_pair_value_asm_impl(
          r,
          alpha_data[i],
          alpha_data[j],
          rvdw[ij],
          roffset[ij],
          lin[ij],
          quad[ij],
          cubic_coeff,
          quartic_coeff,
          exp_power_1,
          exp_power_2,
          exp2_scale,
          exp2_weight,
          &dvalue_dr);
      matvec[i] += value * scaled_data[j];
      matvec[j] += value * scaled_data[i];
      const double pair_scale = 2.0 * scaled_data[i] * scaled_data[j];
      energy += pair_scale * value;
      const double pref = pair_scale * dvalue_dr / r;
      grad[3 * i] += pref * dx;
      grad[3 * i + 1] += pref * dy;
      grad[3 * i + 2] += pref * dz;
      grad[3 * j] -= pref * dx;
      grad[3 * j + 1] -= pref * dy;
      grad[3 * j + 2] -= pref * dz;
    }
  }

  PyObject* out = PyTuple_New(3);
  if (!out) {
    Py_DECREF(grad_obj);
    Py_DECREF(matvec_obj);
    return nullptr;
  }
  PyTuple_SET_ITEM(out, 0, PyFloat_FromDouble(energy));
  PyTuple_SET_ITEM(out, 1, grad_obj);
  PyTuple_SET_ITEM(out, 2, matvec_obj);
  return out;
}

PyObject* multipole_damping_pair(PyObject*, PyObject* args) {
  double a = 0.0;
  double b = 0.0;
  PyObject* amplitudes_obj = nullptr;
  PyObject* betas_obj = nullptr;
  if (!PyArg_ParseTuple(args, "ddOO", &a, &b, &amplitudes_obj, &betas_obj)) {
    return nullptr;
  }

  PyArrayHolder amplitudes = as_array(amplitudes_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder betas = as_array(betas_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!amplitudes.get() || !betas.get()) {
    return nullptr;
  }
  if (!require_1d(amplitudes.get(), "amplitudes") ||
      !require_1d(betas.get(), "betas") ||
      PyArray_DIM(amplitudes.get(), 0) != 4 ||
      PyArray_DIM(betas.get(), 0) != 4) {
    PyErr_SetString(PyExc_ValueError, "amplitudes and betas must be 1D arrays of length 4");
    return nullptr;
  }

  npy_intp dims[1] = {4};
  PyObject* out_obj = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
  if (!out_obj) {
    return nullptr;
  }
  auto* out = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out_obj)));
  const auto* amp = static_cast<const double*>(PyArray_DATA(amplitudes.get()));
  const auto* beta = static_cast<const double*>(PyArray_DATA(betas.get()));
  const double delta = a - b;
  for (int k = 0; k < 4; ++k) {
    out[k] = 0.5 * amp[k] * (1.0 + std::erf(delta * beta[k]));
  }
  return out_obj;
}

PyObject* multipole_mrad_pair(PyObject*, PyObject* args) {
  PyObject* table_obj = nullptr;
  Py_ssize_t i = 0;
  Py_ssize_t j = 0;
  if (!PyArg_ParseTuple(args, "Onn", &table_obj, &i, &j)) {
    return nullptr;
  }

  PyArrayHolder table = as_array(table_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!table.get()) {
    return nullptr;
  }
  if (!require_2d(table.get(), "table")) {
    return nullptr;
  }
  const npy_intp n0 = PyArray_DIM(table.get(), 0);
  const npy_intp n1 = PyArray_DIM(table.get(), 1);
  if (i < 0 || j < 0 || i >= n0 || j >= n1) {
    PyErr_Format(
        PyExc_IndexError,
        "mrad index out of range: (%zd, %zd) for shape (%zd, %zd)",
        i,
        j,
        static_cast<Py_ssize_t>(n0),
        static_cast<Py_ssize_t>(n1));
    return nullptr;
  }
  const auto* data = static_cast<const double*>(PyArray_DATA(table.get()));
  return PyFloat_FromDouble(data[static_cast<npy_intp>(i) * n1 + static_cast<npy_intp>(j)]);
}

PyObject* multipole_damping_pair_derivs(PyObject*, PyObject* args) {
  double a = 0.0;
  double b = 0.0;
  PyObject* amplitudes_obj = nullptr;
  PyObject* betas_obj = nullptr;
  if (!PyArg_ParseTuple(args, "ddOO", &a, &b, &amplitudes_obj, &betas_obj)) {
    return nullptr;
  }

  PyArrayHolder amplitudes = as_array(amplitudes_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayHolder betas = as_array(betas_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!amplitudes.get() || !betas.get()) {
    return nullptr;
  }
  if (!require_1d(amplitudes.get(), "amplitudes") ||
      !require_1d(betas.get(), "betas") ||
      PyArray_DIM(amplitudes.get(), 0) != 4 ||
      PyArray_DIM(betas.get(), 0) != 4) {
    PyErr_SetString(PyExc_ValueError, "amplitudes and betas must be 1D arrays of length 4");
    return nullptr;
  }

  npy_intp dims[1] = {4};
  PyObject* values_obj = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
  PyObject* d_delta_obj = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
  if (!values_obj || !d_delta_obj) {
    Py_XDECREF(values_obj);
    Py_XDECREF(d_delta_obj);
    return nullptr;
  }

  auto* values = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(values_obj)));
  auto* d_delta = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(d_delta_obj)));
  const auto* amp = static_cast<const double*>(PyArray_DATA(amplitudes.get()));
  const auto* beta = static_cast<const double*>(PyArray_DATA(betas.get()));
  const double delta = a - b;
  constexpr double inv_sqrt_pi = 0.56418958354775628694807945156077258584;
  for (int k = 0; k < 4; ++k) {
    const double x = delta * beta[k];
    values[k] = 0.5 * amp[k] * (1.0 + std::erf(x));
    d_delta[k] = amp[k] * beta[k] * inv_sqrt_pi * std::exp(-(x * x));
  }

  PyObject* out = PyTuple_New(2);
  if (!out) {
    Py_DECREF(values_obj);
    Py_DECREF(d_delta_obj);
    return nullptr;
  }
  PyTuple_SET_ITEM(out, 0, values_obj);
  PyTuple_SET_ITEM(out, 1, d_delta_obj);
  return out;
}

PyMethodDef methods[] = {
    {
        "scaled_zeff",
        scaled_zeff,
        METH_VARARGS,
        "Compute the g-xTB scaled-Zeff microkernel observed in the release binary.",
    },
    {
        "cn_scaled_parameter",
        cn_scaled_parameter,
        METH_VARARGS,
        "Compute the g-xTB CN-scaled parameter and CN derivative microkernel.",
    },
    {
        "repulsion_energy_from_matvec",
        repulsion_energy_from_matvec,
        METH_VARARGS,
        "Contract scaled-Zeff with the cached repulsion matrix-vector product.",
    },
    {
        "repulsion_descriptor_potential",
        repulsion_descriptor_potential,
        METH_VARARGS,
        "Compute the descriptor-potential increment observed in g-xTB repulsion.",
    },
    {
        "repulsion_pair_value",
        repulsion_pair_value,
        METH_VARARGS,
        "Compute the observed two-exponential inverse-polynomial repulsion pair form.",
    },
    {
        "repulsion_pair_value_deriv",
        repulsion_pair_value_deriv,
        METH_VARARGS,
        "Compute the observed repulsion pair form and d/dR derivative.",
    },
    {
        "repulsion_pair_value_asm",
        repulsion_pair_value_asm,
        METH_VARARGS,
        "Compute the disassembled g-xTB repulsion pair form with rvdw/R polynomial terms.",
    },
    {
        "repulsion_pair_value_asm_deriv",
        repulsion_pair_value_asm_deriv,
        METH_VARARGS,
        "Compute the disassembled g-xTB repulsion pair form and d/dR derivative.",
    },
    {
        "repulsion_pair_matrix",
        repulsion_pair_matrix,
        METH_VARARGS,
        "Build a symmetric pair matrix from the observed g-xTB repulsion pair form.",
    },
    {
        "repulsion_energy_gradient",
        repulsion_energy_gradient,
        METH_VARARGS,
        "Compute energy, gradient, and matvec from the observed repulsion pair form.",
    },
    {
        "repulsion_pair_matrix_asm",
        repulsion_pair_matrix_asm,
        METH_VARARGS,
        "Build a symmetric pair matrix from the disassembled g-xTB repulsion pair form.",
    },
    {
        "repulsion_energy_gradient_asm",
        repulsion_energy_gradient_asm,
        METH_VARARGS,
        "Compute energy, gradient, and matvec from the disassembled g-xTB pair form.",
    },
    {
        "multipole_damping_pair",
        multipole_damping_pair,
        METH_VARARGS,
        "Compute the g-xTB multipole erf damping-pair microkernel.",
    },
    {
        "multipole_mrad_pair",
        multipole_mrad_pair,
        METH_VARARGS,
        "Fetch the g-xTB multipole radius pair table entry.",
    },
    {
        "multipole_damping_pair_derivs",
        multipole_damping_pair_derivs,
        METH_VARARGS,
        "Compute g-xTB multipole damping values and d/d(a-b) derivatives.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_gxtb_cpp",
    "Native float64 CPU kernels for clean-room g-xTB reconstruction.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__gxtb_cpp() {
  import_array();
  return PyModule_Create(&module);
}
