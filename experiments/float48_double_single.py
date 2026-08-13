"""Double-single (pseudo-float48) on Metal via MLX: does it hold, and does it pay?

x is carried as (hi, lo), both float32, with hi = fl(x) and lo = fl(x - hi).
That is ~48 mantissa bits against float32's 24 and float64's 53.

The algorithms (Dekker/Knuth two-sum, two-product) are only valid under exact
IEEE-754 rounding with no contraction or reassociation. A compiler that fuses
`a*b` and the subsequent subtraction, or that applies fast-math, silently
destroys the error term and the whole thing degrades to float32 while still
appearing to run. So precision is measured before speed.
"""
import time, numpy as np, mlx.core as mx

def split_hi_lo(x):
    hi = mx.array(np.asarray(x, dtype=np.float32))
    lo = mx.array(np.asarray(x - np.asarray(hi, dtype=np.float64), dtype=np.float32))
    return hi, lo

def to_f64(hi, lo):
    return np.asarray(hi, dtype=np.float64) + np.asarray(lo, dtype=np.float64)

def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)

def quick_two_sum(a, b):
    s = a + b
    return s, b - (s - a)

SPLIT = mx.array(np.float32(4097.0))          # 2**12 + 1
def dekker_split(a):
    c = SPLIT * a
    hi = c - (c - a)
    return hi, a - hi

def two_prod(a, b):
    p = a * b
    ah, al = dekker_split(a)
    bh, bl = dekker_split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl

def df_add(ah, al, bh, bl):
    s, e = two_sum(ah, bh)
    e = e + (al + bl)
    return quick_two_sum(s, e)

def df_mul(ah, al, bh, bl):
    p, e = two_prod(ah, bh)
    e = e + (ah * bl + al * bh)
    return quick_two_sum(p, e)

# ---------------------------------------------------------------- precision
rng = np.random.default_rng(0)
N = 200_000
a = rng.uniform(0.5, 2.0, N)
b = rng.uniform(0.5, 2.0, N)
ah, al = split_hi_lo(a); bh, bl = split_hi_lo(b)

sh, sl = df_add(ah, al, bh, bl); mx.eval(sh, sl)
ph, pl = df_mul(ah, al, bh, bl); mx.eval(ph, pl)

def rel(got, want):
    return float(np.abs((got - want) / want).max())

e_add_f32 = rel(np.asarray(ah + bh, dtype=np.float64), a + b)
e_add_df  = rel(to_f64(sh, sl), a + b)
e_mul_f32 = rel(np.asarray(ah * bh, dtype=np.float64), a * b)
e_mul_df  = rel(to_f64(ph, pl), a * b)
print(f"  add: float32 {e_add_f32:.3e}   df32 {e_add_df:.3e}   gain {e_add_f32/max(e_add_df,1e-300):.3g}x")
print(f"  mul: float32 {e_mul_f32:.3e}   df32 {e_mul_df:.3e}   gain {e_mul_f32/max(e_mul_df,1e-300):.3g}x")
print(f"  float32 eps {np.finfo(np.float32).eps:.2e}   float64 eps {np.finfo(np.float64).eps:.2e}")
bits = -np.log2(max(e_mul_df, 1e-300))
print(f"  effective mantissa bits from df32 multiply: {bits:.1f}  (f32=24, f64=53, target ~48)")

# ---------------------------------------------------------------- speed
def bench(fn, repeats=7):
    fn(); ts = []
    for _ in range(repeats):
        t = time.perf_counter(); r = fn()
        if isinstance(r, tuple): mx.eval(*r)
        elif isinstance(r, mx.array): mx.eval(r)
        ts.append(time.perf_counter() - t)
    return min(ts)

a64, b64 = a, b
t_np = bench(lambda: (a64 * b64 + a64, None)[1] if False else np.add(np.multiply(a64, b64), a64))
t_f32 = bench(lambda: ah * bh + ah)
def df_chain():
    ph_, pl_ = df_mul(ah, al, bh, bl)
    return df_add(ph_, pl_, ah, al)
t_df = bench(df_chain)
print(f"\n  one a*b+a on {N} elements:")
print(f"    numpy float64      {t_np*1e3:7.3f} ms")
print(f"    mlx float32        {t_f32*1e3:7.3f} ms   {t_np/t_f32:5.2f}x vs numpy")
print(f"    mlx df32 (~f48)    {t_df*1e3:7.3f} ms   {t_np/t_df:5.2f}x vs numpy   "
      f"{t_df/t_f32:5.1f}x the float32 cost")
