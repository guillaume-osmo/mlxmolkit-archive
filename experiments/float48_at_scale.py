"""df32 at the arithmetic intensity and scale the integral block actually has."""
import time, numpy as np, mlx.core as mx
exec(open("experiments/float48_double_single.py").read().split("# ---------------------------------------------------------------- precision")[0])

def bench(fn, repeats=5):
    fn(); ts = []
    for _ in range(repeats):
        t = time.perf_counter(); r = fn()
        if isinstance(r, tuple): mx.eval(*r)
        elif isinstance(r, mx.array): mx.eval(r)
        ts.append(time.perf_counter() - t)
    return min(ts)

print(f"{'P':>9} {'ops':>4} | {'numpy f64':>10} {'mlx f32':>9} {'mlx df32':>9} | "
      f"{'df32 vs numpy':>13} {'df32/f32':>9}")
for P in (102_000, 1_000_000):
    for nops in (5, 20, 60):
        rng = np.random.default_rng(0)
        a = rng.uniform(0.5, 2.0, P); b = rng.uniform(0.5, 2.0, P)
        ah, al = split_hi_lo(a); bh, bl = split_hi_lo(b)
        mx.eval(ah, al, bh, bl)

        def np_chain():
            x = a.copy()
            for _ in range(nops): x = x * b + a
            return x
        def f32_chain():
            x = ah
            for _ in range(nops): x = x * bh + ah
            return x
        def df_chain():
            xh, xl = ah, al
            for _ in range(nops):
                ph, pl = df_mul(xh, xl, bh, bl)
                xh, xl = df_add(ph, pl, ah, al)
            return xh, xl

        t_np, t_f32, t_df = bench(np_chain), bench(f32_chain), bench(df_chain)
        print(f"{P:>9} {nops:>4} | {t_np*1e3:9.2f}m {t_f32*1e3:8.2f}m {t_df*1e3:8.2f}m | "
              f"{t_np/t_df:12.2f}x {t_df/t_f32:8.1f}x")
