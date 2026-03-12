"""
run_remote.py  --  Thin wrapper for SSH cluster execution.
Runs run_batch_gpu_v4 with given args and prints results as JSON to stdout.

Usage:
    py run_remote.py --L 80 --P 0.001 --seeds 4,5,6,7 --nsteps 100000 --rw 5 --hfield 0.0
"""

import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import os
os.chdir(Path(__file__).parent)

parser = argparse.ArgumentParser()
parser.add_argument('--L',      type=int,   required=True)
parser.add_argument('--P',      type=float, required=True)
parser.add_argument('--seeds',  type=str,   required=True)   # comma-separated
parser.add_argument('--nsteps', type=int,   required=True)
parser.add_argument('--rw',     type=int,   default=5)
parser.add_argument('--hfield', type=float, default=0.0)
args = parser.parse_args()

seeds = [int(s) for s in args.seeds.split(',')]

from vcml_gpu_v4 import run_batch_gpu_v4

results = run_batch_gpu_v4(args.L, [args.P], seeds, args.nsteps,
                            r_w=args.rw, h_field=args.hfield)

# Sanitize for JSON (remove lags/acf lists to keep output small)
out = []
for r in results:
    out.append({k: v for k, v in r.items() if k not in ('lags', 'acf')})

# Print marker so coordinator can find the JSON even if there's compile noise
print('RESULT_JSON:' + json.dumps(out), flush=True)
