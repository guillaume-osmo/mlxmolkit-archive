"""Does MLX beat NumPy at the shapes the integral block actually uses?

Measured before porting anything. Includes host->device and device->host,
because prepare_batch's callers are NumPy on both sides.
"""
import time, numpy as np, mlx.core as mx

def bench(fn, *a, repeats=7):
    fn(*a); mx.eval(mx.zeros(1))
    ts = []
    for _ in range(repeats):
        t = time.perf_counter(); r = fn(*a)
        if isinstance(r, mx.array): mx.eval(r)
        ts.append(time.perf_counter() - t)
    return min(ts)

P = 102_000     # pairs in an 800-molecule mixed batch
print(f"P = {P} pairs (an 800-molecule mixed batch)\n")

# --- 1. the raw transfer floor -------------------------------------------
ri = np.random.rand(P, 22)
r0 = np.random.rand(P, 3)
t_up = bench(lambda: mx.array(ri))
big = mx.array(np.random.rand(P, 4, 4, 4, 4))
mx.eval(big)
t_down = bench(lambda: np.array(big))
print(f"  host->device (P,22)      {t_up*1e3:6.2f} ms")
print(f"  device->host (P,4,4,4,4) {t_down*1e3:6.2f} ms   <- the (P,256) result alone")

# --- 2. a representative elementwise chain (what tetci/overlap do) --------
def np_chain(a, b):
    for _ in range(20):
        a = a * b + np.sqrt(a * a + 1.0)
    return a
def mx_chain(a, b):
    for _ in range(20):
        a = a * b + mx.sqrt(a * a + 1.0)
    return a

a_np, b_np = np.random.rand(P), np.random.rand(P)
a_mx, b_mx = mx.array(a_np), mx.array(b_np)
mx.eval(a_mx, b_mx)
t_np = bench(np_chain, a_np, b_np)
t_mx = bench(mx_chain, a_mx, b_mx)
t_mx_rt = bench(lambda: np.array(mx_chain(mx.array(a_np), mx.array(b_np))))
print(f"\n  20-op elementwise chain on (P,)")
print(f"    numpy                  {t_np*1e3:6.2f} ms")
print(f"    mlx (resident)         {t_mx*1e3:6.2f} ms   {t_np/t_mx:5.2f}x")
print(f"    mlx (round-trip)       {t_mx_rt*1e3:6.2f} ms   {t_np/t_mx_rt:5.2f}x")

# --- 3. the scatter pattern rotate_xx_batch uses -------------------------
combos = np.random.randint(0, 4, size=(60, 4))
vals_np = np.random.rand(P, 60)
def np_scatter():
    w = np.zeros((P, 4, 4, 4, 4))
    kk, ll, mm, nn = combos[:,0], combos[:,1], combos[:,2], combos[:,3]
    w[:, kk, ll, mm, nn] = vals_np
    w[:, ll, kk, mm, nn] = vals_np
    return w
vals_mx = mx.array(vals_np); mx.eval(vals_mx)
kk = mx.array(combos[:,0]); ll = mx.array(combos[:,1])
mm = mx.array(combos[:,2]); nn = mx.array(combos[:,3])
def mx_scatter():
    w = mx.zeros((P, 4, 4, 4, 4))
    w[:, kk, ll, mm, nn] = vals_mx
    w[:, ll, kk, mm, nn] = vals_mx
    return w
t_np_s = bench(np_scatter)
t_mx_s = bench(mx_scatter)
print(f"\n  scatter into (P,4,4,4,4)")
print(f"    numpy                  {t_np_s*1e3:6.2f} ms")
print(f"    mlx (resident)         {t_mx_s*1e3:6.2f} ms   {t_np_s/t_mx_s:5.2f}x")
