import multiprocessing as mp, subprocess, time, os

def do_import(barrier):
    barrier.wait()
    import mlx.core   # noqa: F401
    time.sleep(0.2)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    n = 14
    barrier = mp.Barrier(n + 1)
    procs = [mp.Process(target=do_import, args=(barrier,)) for _ in range(n)]
    for p in procs:
        p.start()
    time.sleep(2.5)
    pids = [p.pid for p in procs]
    sampler = subprocess.Popen(
        ["sample", str(pids[0]), "2", "-f", f"{os.path.dirname(__file__)}/sample2.txt"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    barrier.wait()                      # release the import storm
    out, _ = sampler.communicate(timeout=120)
    for p in procs:
        p.join()
    print("sampler said:", out.strip()[:200])
