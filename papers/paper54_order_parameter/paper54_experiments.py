"""
Paper 54: Order Parameter, Correlation Length Law, and Cross-Axis Surfaces.

Three sub-experiments:

Exp A — sg_C as order parameter (Axis I sweep)
  Sweep SUPP_AMP/EXC_AMP at S1. Collect sg4, sigma_w, snr=sg4/sigma_w at each.
  Q: Is snr monotone where sg4 is non-monotone? If yes, snr=sg_C is a better
  order parameter. Non-monotone sg4 with monotone snr would collapse Axis I.

Exp B — Correlation length law (r_wave sweep at S2)
  Fix S2 grid (N=6,400), optimal amplitude. Sweep r_wave: 1,2,4,8,16.
  delta = zone_width / r_wave. As r_wave increases, delta decreases.
  Two-sided boundary: too small r_wave -> patch formation (copy-forward range
  too small to span zone); too large r_wave -> wave bleed (Axis II failure).
  Mapping the peak gives the correlation length of the copy-forward loop.

Exp C — Cross-axis surface: write trigger x perturbation intensity
  Run C_ref and C_perturb at 3 amplitude levels (sub-opt, optimal, over-pert).
  Prediction: C_perturb fails (na_form<1) at low amplitude (no wave events
  to trigger writes); C_ref maintains formation slowly. Confirms trigger
  selectivity axis is contingent on perturbation intensity axis.

Runs: 6x5=30 (A) + 5x5=25 (B) + 3x2x5=30 (C) = 85 total.
"""

import numpy as np, json, os, math, random, multiprocessing as mp

# ── Fixed constants ───────────────────────────────────────────────────────────
HS = 2; IS = 3
WAVE_DUR   = 15
MID_DECAY  = 0.99; FIELD_DECAY = 0.9997; BASE_BETA = 0.005
FA         = 0.16
VALS_DECAY = 0.92; VALS_NAV = 0.08; ADJ_SCALE = 0.03
BOUND_LO = 0.05; BOUND_HI = 0.95; FRAG_NOISE = 0.01
INST_THRESH = 0.45; INST_PROB = 0.03; DIFFUSE = 0.02; DIFFUSE_EVERY = 2
NEAR_BAND   = 0.15

SEEDS   = list(range(5))
STEPS   = 2000
N_ZONES = 4

# S1 default
S1 = dict(W=80,  H=40,  HALF=40,  WR=4.8)
# S2 (N=6400): W=160, H=80, HALF=80
S2 = dict(W=160, H=80,  HALF=80,  WR=19.2)

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "paper54_results.json")

# ── Exp A amplitude levels (supp_amp, exc_amp) ────────────────────────────────
AMP_LEVELS = [
    (0.06, 0.12),   # A1: deep sub-threshold
    (0.12, 0.24),   # A2: near threshold
    (0.25, 0.50),   # A3: optimal (standard)
    (0.45, 0.90),   # A4: over-perturbed
    (0.80, 1.60),   # A5: strongly over-perturbed
    (1.20, 2.40),   # A6: extreme
]
AMP_LABELS = ['A1_deep_sub', 'A2_near_thresh', 'A3_optimal',
              'A4_over', 'A5_strong_over', 'A6_extreme']

# ── Exp B r_wave levels ───────────────────────────────────────────────────────
RWAVE_LEVELS = [1, 2, 4, 8, 16]

# ── Exp C: write modes and amplitude levels ───────────────────────────────────
WRITE_MODES_C = ['C_ref', 'C_perturb']
AMP_LEVELS_C  = [
    (0.06, 0.12),   # deep sub-threshold
    (0.25, 0.50),   # optimal
    (0.80, 1.60),   # over-perturbed
]
AMP_LABELS_C = ['sub_thresh', 'optimal', 'over_pert']


# ── Parametric FastVCML ───────────────────────────────────────────────────────
class FastVCML:
    def __init__(self, seed, ss=10, seed_beta=0.25, half=40, h=40):
        self.rng       = np.random.RandomState(seed)
        self.SS        = ss
        self.seed_beta = seed_beta
        self.HALF      = half
        self.H         = h
        N = half * h
        self.N = N; self.t = 0
        self.ai_x = half + (np.arange(N) % half)
        self.ai_y = np.arange(N) // half
        rng = self.rng; sc = 0.3
        self.Wz=rng.randn(N,HS,IS)*sc; self.Uz=rng.randn(N,HS,HS)*sc; self.bz=np.full((N,HS),.5)
        self.Wr=rng.randn(N,HS,IS)*sc; self.Ur=rng.randn(N,HS,HS)*sc; self.br=np.full((N,HS),.5)
        self.Wh=rng.randn(N,HS,IS)*sc; self.Uh=rng.randn(N,HS,HS)*sc; self.bh=np.zeros((N,HS))
        self.Wo=rng.randn(N,HS)*0.1;   self.bo=np.zeros(N)
        self.vals  = rng.uniform(.3, .7, N)
        self.hid   = rng.randn(N, HS) * .1
        self.age   = rng.randint(0, 50, N).astype(float)
        self.base_h = np.zeros((N, HS))
        self.mid    = np.zeros((N, HS))
        self.fieldM = np.zeros((N, HS))
        self.streak = np.zeros(N, int)
        self.cc     = np.zeros(N, int)
        self._build_nb()

    def _build_nb(self):
        N = self.N; HALF = self.HALF; H = self.H
        rx = np.arange(N) % HALF; y = np.arange(N) // HALF
        nb = np.full((N, 4), -1, int)
        nb[rx > 0,      0] = np.where(rx > 0)[0] - 1
        nb[rx < HALF-1, 1] = np.where(rx < HALF-1)[0] + 1
        nb[y > 0,       2] = np.where(y > 0)[0] - HALF
        nb[y < H-1,     3] = np.where(y < H-1)[0] + HALF
        self.nb = nb
        self.nbc = (nb >= 0).sum(1).astype(float)

    def _nbmean(self):
        ns = np.maximum(self.nb, 0); g = self.vals[ns]; g[self.nb < 0] = 0.0
        return np.where(self.nbc > 0, g.sum(1) / self.nbc, .5)

    def _gru(self, x):
        h = self.hid
        def sig(a): return 1/(1+np.exp(-np.clip(a,-8,8)))
        def tanh(a): e2=np.exp(2*np.clip(a,-8,8)); return (e2-1)/(e2+1)
        x3, h3 = x[:,:,None], h[:,:,None]
        z = sig((self.Wz@x3).squeeze(-1)+(self.Uz@h3).squeeze(-1)+self.bz)
        r = sig((self.Wr@x3).squeeze(-1)+(self.Ur@h3).squeeze(-1)+self.br)
        rh = (r*h)[:,:,None]
        g = tanh((self.Wh@x3).squeeze(-1)+(self.Uh@rh).squeeze(-1)+self.bh)
        hn = (1-z)*h + z*g
        out = np.tanh(np.einsum('ni,ni->n', self.Wo, hn) + self.bo)
        return hn, out

    def _diffuse(self):
        ns = np.maximum(self.nb, 0); nf = self.fieldM[ns]
        nf *= (self.nb >= 0)[:,:,None]
        nm = np.where(self.nbc[:,None] > 0, nf.sum(1)/self.nbc[:,None], self.fieldM)
        self.fieldM += DIFFUSE * (nm - self.fieldM)

    def step(self, wa, wc):
        nb = self._nbmean(); an = np.minimum(1., self.age/300.)
        x = np.stack([nb, np.minimum(1., wa), an], 1)
        hn, out = self._gru(x); self.hid = hn
        self.vals = np.clip(VALS_DECAY*self.vals + VALS_NAV*nb + ADJ_SCALE*out, 0, 1)
        dev = hn - self.base_h; self.base_h += BASE_BETA * dev
        self.streak = np.where(np.sum(dev**2, 1) < .0025, self.streak+1, 0)
        self.mid = (self.mid + FA*dev) * MID_DECAY
        gate = self.streak >= self.SS
        self.fieldM[gate] += FA * (self.mid[gate] - self.fieldM[gate])
        self.fieldM *= FIELD_DECAY
        if self.t % DIFFUSE_EVERY == 0: self._diffuse()
        self.age += 1; self._collapse(); self.t += 1

    def _collapse(self):
        rng = self.rng
        bad  = (self.vals < BOUND_LO) | (self.vals > BOUND_HI)
        inst = (np.sum(np.abs(self.hid-self.base_h),1) > INST_THRESH) & (rng.random(self.N) < INST_PROB)
        ci   = np.where(bad | inst)[0]
        if not len(ci): return
        prev = self.hid.copy()
        for ai in ci:
            self.cc[ai] += 1
            fm = self.fieldM[ai]; mag = np.sqrt(np.dot(fm, fm)); sb = self.seed_beta
            nh = ((1-sb)*prev[ai] + sb*fm if mag > 1e-6 else prev[ai].copy())
            nh += rng.randn(HS) * FRAG_NOISE
            self.hid[ai]=nh; self.vals[ai]=.5; self.age[ai]=0
            self.streak[ai]=0; self.mid[ai]=np.zeros(HS)


# ── FastVCML with write-mode switching (for Exp C) ───────────────────────────
class FastVCML54C(FastVCML):
    def __init__(self, seed, ss=10, seed_beta=0.25, half=40, h=40, write_mode='C_ref'):
        super().__init__(seed, ss, seed_beta, half, h)
        self.write_mode = write_mode

    def step(self, wa, wc):
        nb = self._nbmean(); an = np.minimum(1., self.age/300.)
        x = np.stack([nb, np.minimum(1., wa), an], 1)
        hn, out = self._gru(x); self.hid = hn
        self.vals = np.clip(VALS_DECAY*self.vals + VALS_NAV*nb + ADJ_SCALE*out, 0, 1)
        dev = hn - self.base_h; self.base_h += BASE_BETA * dev
        self.streak = np.where(np.sum(dev**2, 1) < .0025, self.streak+1, 0)
        self.mid = (self.mid + FA*dev) * MID_DECAY
        base_gate = self.streak >= self.SS
        if self.write_mode == 'C_ref':
            wg = base_gate; ws = self.mid
        elif self.write_mode == 'C_perturb':
            wg = base_gate & (wa > 0.05); ws = self.mid
        else:
            wg = base_gate; ws = self.mid
        if wg.any():
            self.fieldM[wg] += FA * (ws[wg] - self.fieldM[wg])
        self.fieldM *= FIELD_DECAY
        if self.t % DIFFUSE_EVERY == 0: self._diffuse()
        self.age += 1; self._collapse(); self.t += 1


# ── Parametric WaveEnvStd ─────────────────────────────────────────────────────
class WaveEnv:
    def __init__(self, vcml, WR, supp_amp=0.25, exc_amp=0.50,
                 r_wave=2, n_zones=4, rng_seed=0):
        self.vcml = vcml; self.WR = WR
        self.supp_amp = supp_amp; self.exc_amp = exc_amp
        self.rwave = r_wave; self.nz = n_zones
        self.zw = vcml.HALF // n_zones
        self.rng = random.Random(rng_seed); self.waves = []
        N = vcml.N; self._wa = np.empty(N); self._wc = np.empty(N, int)

    def _launch(self):
        cls = self.rng.randint(0, self.nz-1)
        sx = self.rng.randint(cls*self.zw, (cls+1)*self.zw - 1)
        cx = self.vcml.HALF + sx
        cy = self.rng.randint(0, self.vcml.H - 1)
        self.waves.append([cx, cy, WAVE_DUR, cls])

    def _apply(self):
        wa = self._wa; wc = self._wc; wa.fill(0.); wc.fill(-1)
        ax = self.vcml.ai_x; ay = self.vcml.ai_y; rw = self.rwave; surv = []
        for wave in self.waves:
            cx, cy, rem, cls = wave
            if rem <= 0: continue
            dist = np.abs(ax-cx) + np.abs(ay-cy)
            act  = np.maximum(0., 1. - dist/rw); act[dist > rw] = 0.
            better = act > wa; wa[better] = act[better]; wc[better] = cls
            wave[2] -= 1
            if wave[2] > 0: surv.append(wave)
        self.waves = surv
        for even in [True, False]:
            amp = self.supp_amp if even else self.exc_amp
            par = (wc%2==0) if even else (wc%2==1)
            idx = np.where((wc >= 0) & par & (wa > .05))[0]
            if not len(idx): continue
            sc = wa[idx] * amp
            if even: self.vcml.vals[idx] = np.maximum(0., self.vcml.vals[idx]*(1.-sc*.5))
            else:    self.vcml.vals[idx] = np.minimum(1., self.vcml.vals[idx]+sc*.5)
        return wa.copy(), wc.copy()

    def step(self):
        exp = self.WR / WAVE_DUR
        nl = int(exp) + (1 if self.rng.random() < exp-int(exp) else 0)
        for _ in range(nl): self._launch()
        return self._apply()


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(vcml):
    HALF = vcml.HALF; n_zones = N_ZONES; zw = HALF // n_zones
    means = []; spreads = []
    for z in range(n_zones):
        xlo = HALF + z*zw; xhi = HALF + (z+1)*zw
        mask = (vcml.ai_x >= xlo) & (vcml.ai_x < xhi)
        fm = vcml.fieldM[mask]
        means.append(fm.mean(0) if len(fm) else np.zeros(HS))
        spreads.append(float(np.std(fm)) if len(fm) > 1 else float('nan'))
    # sg4
    dists = [np.linalg.norm(means[a]-means[b])
             for a in range(n_zones) for b in range(a+1, n_zones)]
    sg4 = float(np.mean(dists)) if dists else 0.
    # sigma_w
    sw_vals = [s for s in spreads if not math.isnan(s)]
    sigma_w = float(np.mean(sw_vals)) if sw_vals else float('nan')
    # snr = sg_C
    snr = float(sg4/sigma_w) if sigma_w and sigma_w > 1e-8 else float('nan')
    # na_ratio
    adj = []; nadj = []
    for a in range(n_zones):
        for b in range(a+1, n_zones):
            d = float(np.linalg.norm(means[a]-means[b]))
            if abs(a-b) == 1: adj.append(d)
            else:              nadj.append(d)
    na = float(np.mean(nadj)/np.mean(adj)) if adj and nadj and np.mean(adj) > 1e-8 else float('nan')
    # coll/site
    coll_site = float(vcml.cc.sum()) / (vcml.N * vcml.t)
    return dict(sg4=sg4, sigma_w=sigma_w, snr=snr, na=na, coll_site=coll_site)


# ── Run functions ─────────────────────────────────────────────────────────────
def run_exp_a(seed, supp_amp, exc_amp):
    vcml = FastVCML(seed, half=S1['HALF'], h=S1['H'])
    env  = WaveEnv(vcml, WR=S1['WR'], supp_amp=supp_amp, exc_amp=exc_amp,
                   r_wave=2, rng_seed=seed*1000)
    for _ in range(STEPS):
        wa, wc = env.step(); vcml.step(wa, wc)
    return compute_metrics(vcml)

def run_exp_b(seed, r_wave):
    vcml = FastVCML(seed, half=S2['HALF'], h=S2['H'])
    env  = WaveEnv(vcml, WR=S2['WR'], supp_amp=0.25, exc_amp=0.50,
                   r_wave=r_wave, rng_seed=seed*1000)
    for _ in range(STEPS):
        wa, wc = env.step(); vcml.step(wa, wc)
    return compute_metrics(vcml)

def run_exp_c(seed, supp_amp, exc_amp, write_mode):
    vcml = FastVCML54C(seed, half=S1['HALF'], h=S1['H'], write_mode=write_mode)
    env  = WaveEnv(vcml, WR=S1['WR'], supp_amp=supp_amp, exc_amp=exc_amp,
                   r_wave=2, rng_seed=seed*1000)
    for _ in range(STEPS):
        wa, wc = env.step(); vcml.step(wa, wc)
    return compute_metrics(vcml)


# ── Worker (module-level for Windows spawn) ───────────────────────────────────
def _worker(args):
    exp, *rest = args
    if exp == 'A':
        seed, amp_idx = rest
        sa, ea = AMP_LEVELS[amp_idx]
        return ('A', AMP_LABELS[amp_idx], seed, run_exp_a(seed, sa, ea))
    elif exp == 'B':
        seed, r_wave = rest
        return ('B', r_wave, seed, run_exp_b(seed, r_wave))
    elif exp == 'C':
        seed, amp_idx, wm = rest
        sa, ea = AMP_LEVELS_C[amp_idx]
        return ('C', AMP_LABELS_C[amp_idx], wm, seed, run_exp_c(seed, sa, ea, wm))


def main():
    mp.freeze_support()

    # Load already-done results
    done = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            done = json.load(f)
        print(f"Loaded {len(done)} existing results")

    # Build task list
    all_tasks = []
    for amp_idx in range(len(AMP_LEVELS)):
        for seed in SEEDS:
            k = f'A|{AMP_LABELS[amp_idx]}|{seed}'
            if k not in done:
                all_tasks.append(('A', seed, amp_idx))

    for r_wave in RWAVE_LEVELS:
        for seed in SEEDS:
            k = f'B|{r_wave}|{seed}'
            if k not in done:
                all_tasks.append(('B', seed, r_wave))

    for amp_idx in range(len(AMP_LEVELS_C)):
        for wm in WRITE_MODES_C:
            for seed in SEEDS:
                k = f'C|{AMP_LABELS_C[amp_idx]}|{wm}|{seed}'
                if k not in done:
                    all_tasks.append(('C', seed, amp_idx, wm))

    print(f"Tasks to run: {len(all_tasks)}")
    if not all_tasks:
        print("All done."); return done

    n_proc = min(len(all_tasks), mp.cpu_count())
    with mp.Pool(processes=n_proc) as pool:
        for res in pool.imap_unordered(_worker, all_tasks):
            if res[0] == 'A':
                _, lbl, seed, metrics = res
                k = f'A|{lbl}|{seed}'
            elif res[0] == 'B':
                _, r_wave, seed, metrics = res
                k = f'B|{r_wave}|{seed}'
            else:
                _, lbl, wm, seed, metrics = res
                k = f'C|{lbl}|{wm}|{seed}'
            done[k] = metrics
            print(f"  {k}: sg4={metrics['sg4']:.4f} snr={metrics['snr']:.3f} "
                  f"na={metrics['na']:.3f} coll={metrics['coll_site']:.5f}")
            with open(RESULTS_FILE, 'w') as f:
                json.dump(done, f, indent=2)

    # Aggregate
    agg = {}
    # Exp A
    for amp_idx, lbl in enumerate(AMP_LABELS):
        sa, ea = AMP_LEVELS[amp_idx]
        keys = [f'A|{lbl}|{s}' for s in SEEDS]
        vals = [done[k] for k in keys if k in done]
        if vals:
            agg[f'A|{lbl}'] = {
                'supp_amp': sa, 'exc_amp': ea,
                'sg4':    float(np.mean([v['sg4']    for v in vals])),
                'sigma_w':float(np.mean([v['sigma_w'] for v in vals])),
                'snr':    float(np.mean([v['snr']    for v in vals])),
                'na':     float(np.mean([v['na']     for v in vals])),
                'coll_site': float(np.mean([v['coll_site'] for v in vals])),
            }
    # Exp B
    for r_wave in RWAVE_LEVELS:
        keys = [f'B|{r_wave}|{s}' for s in SEEDS]
        vals = [done[k] for k in keys if k in done]
        if vals:
            zw = S2['HALF'] // N_ZONES
            agg[f'B|{r_wave}'] = {
                'r_wave': r_wave, 'delta': zw/r_wave,
                'sg4':    float(np.mean([v['sg4']    for v in vals])),
                'sigma_w':float(np.mean([v['sigma_w'] for v in vals])),
                'snr':    float(np.mean([v['snr']    for v in vals])),
                'na':     float(np.mean([v['na']     for v in vals])),
                'coll_site': float(np.mean([v['coll_site'] for v in vals])),
            }
    # Exp C
    for amp_idx, albl in enumerate(AMP_LABELS_C):
        sa, ea = AMP_LEVELS_C[amp_idx]
        for wm in WRITE_MODES_C:
            keys = [f'C|{albl}|{wm}|{s}' for s in SEEDS]
            vals = [done[k] for k in keys if k in done]
            if vals:
                agg[f'C|{albl}|{wm}'] = {
                    'supp_amp': sa, 'exc_amp': ea, 'write_mode': wm,
                    'sg4':    float(np.mean([v['sg4']    for v in vals])),
                    'sigma_w':float(np.mean([v['sigma_w'] for v in vals])),
                    'snr':    float(np.mean([v['snr']    for v in vals])),
                    'na':     float(np.mean([v['na']     for v in vals])),
                    'coll_site': float(np.mean([v['coll_site'] for v in vals])),
                }

    agg_file = RESULTS_FILE.replace('paper54_results.json', 'paper54_analysis.json')
    with open(agg_file, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved: {agg_file}")

    print("\n=== Exp A: sg_C sweep (Axis I) ===")
    for lbl in AMP_LABELS:
        k = f'A|{lbl}'
        if k in agg:
            r = agg[k]
            print(f"  {lbl}: coll={r['coll_site']:.5f} sg4={r['sg4']:.4f} "
                  f"sigma_w={r['sigma_w']:.4f} snr={r['snr']:.3f} na={r['na']:.3f}")

    print("\n=== Exp B: Correlation length (r_wave sweep) ===")
    for rw in RWAVE_LEVELS:
        k = f'B|{rw}'
        if k in agg:
            r = agg[k]
            print(f"  r_wave={rw:2d} delta={r['delta']:.1f}: "
                  f"sg4={r['sg4']:.4f} snr={r['snr']:.3f} na={r['na']:.3f}")

    print("\n=== Exp C: Cross-axis (write trigger x amplitude) ===")
    for albl in AMP_LABELS_C:
        for wm in WRITE_MODES_C:
            k = f'C|{albl}|{wm}'
            if k in agg:
                r = agg[k]
                print(f"  {albl} {wm}: coll={r['coll_site']:.5f} "
                      f"sg4={r['sg4']:.4f} snr={r['snr']:.3f} na={r['na']:.3f}")

    return agg


if __name__ == '__main__':
    main()
