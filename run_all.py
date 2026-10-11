"""Run the whole pipeline without Jupyter (works on Windows, macOS and Linux).

    python run_all.py                 # everything, in order
    python run_all.py --from sweeps   # resume from a stage (earlier results are reused)

Stages: data, toy, features, main, sweeps, figures, prune, rf
Run it from this folder. Each step runs as its own process and the run stops at the first failure.
"""
import argparse, os, subprocess, sys

SWEEPS = [
    "--ds catsdogs --noise 0.0 --widths 1,2,4,8,16,32,64,128,256,512",
    "--ds catsdogs --noise 0.2 --widths 4,8,16,32,64,128,512",
    "--ds cifar10 --noise 0.0 --widths 2,4,8,16,32,64,128,512 --max_nodes 100",
    "--ds catsdogs --dontcare paper --widths 2,4,8",
    "--ds catsdogs --trunk random --widths 4,16,64",
    "--ds cifar10 --noise 0.2 --widths 8,32,128,512 --max_nodes 100",
]
STAGES = {
    "data": ["scripts/prepare_data.py"],
    "toy": ["experiments/00_toy_spirals.py"],
    "features": ["experiments/01_features.py"],
    "main": ["experiments/02_main.py --ds catsdogs --root_h 16 --h 16",
             "experiments/02_main.py --ds cifar10 --root_h 16 --h 16"],
    "sweeps": [f"experiments/03_double_descent.py {a}" for a in SWEEPS],
    "figures": ["experiments/06_figures.py"],
    "prune": ["experiments/04_minimise.py"],
    "rf": ["experiments/05_receptive_fields.py", "experiments/06_figures.py"],
}
ORDER = list(STAGES)


def step(cmd):
    print(f"\n=== python {cmd}", flush=True)
    r = subprocess.run([sys.executable, "-u"] + cmd.split())
    if r.returncode != 0:
        sys.exit(f"FAILED: python {cmd}\nFix the error, then resume with:  python run_all.py --from <stage>")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for d in ["data", "results/figs"]:
        os.makedirs(d, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=ORDER, default="data")
    ap.add_argument("--only", choices=ORDER)
    a = ap.parse_args()
    todo = [a.only] if a.only else ORDER[ORDER.index(a.start):]
    for st in todo:
        if st == "sweeps" and os.path.exists("results/dd.jsonl"):
            os.remove("results/dd.jsonl")  # the sweep appends rows: start it clean
        for cmd in STAGES[st]:
            step(cmd)
    print("\nDone. Results in results/")
