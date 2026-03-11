"""
Paper 55: Lyapunov Functional for VCML Identity Formation.

Tests whether C_order = (D_nonadj - D_adj) / sigma_w has monotone positive
drift toward identity-encoding attractors in the adaptive regime, and
weak/negative drift in parity/patch regimes.

Five behavioral regimes (S1 grid, 5 seeds each = 25 runs):
  ident_opt   : supp=0.25, exc=0.50, r_wave=2, C_ref  -- standard identity
  parity_sub  : supp=0.06, exc=0.12, r_wave=2, C_ref  -- amplitude parity
  parity_over : supp=0.80, exc=1.60, r_wave=2, C_ref  -- over-perturbed parity
  patch       : supp=0.25, exc=0.50, r_wave=1, C_ref  -- correlation-length patch
  ident_cp    : supp=0.06, exc=0.12, r_wave=2, C_perturb -- noise-gated identity

Metrics tracked every TRACK_EVERY steps:
  D_adj, D_nonadj, sg4, sigma_w, na_ratio, c_order = (D_nonadj-D_adj)/sigma_w
"""

import numpy as np, json, os, math, random, multiprocessing as mp

# ── Fixed constants ────────────────────────────────────────────────────────────
HS = 2; IS = 3
WAVE_DUR   = 15
MID_DECAY  = 0.99; FIELD_DECAY = 0.9997; BASE_BETA = 0.005
FA         = 0.16
VALS_DECAY = 0.92; VALS_NAV = 0.08; ADJ_SCALE = 0.03
BOUND_LO = 0.05; BOUND_HI = 0.95; FRAG_NOISE = 0.01
INST_THRESH = 0.45; INST_PROB = 0.03; DIFFUSE = 0.02; DIFFUSE_EVERY = 2

SEEDS       = list(range(5))
STEPS       = 3000
TRACK_EVERY = 25    # record metrics every 25 steps -> 120 timepoints
N_ZONES     = 4

S1 = dict(W=80, H=40, HALF=40, WR=4.8)

# Five regimes to test Lyapunov drift
REGIMES = {
    'ident_opt':   dict(supp=0.25, exc=0.50, r_wave=2, mode='C_ref'),
    'parity_sub':  dict(supp=0.06, exc=0.12, r_wave=2, mode='C_ref'),
    'parity_over': dict(supp=0.80, exc=1.60, r_wave=2, mode='C_ref'),
    'patch':       dict(supp=0.25, exc=0.50, r_wave=1, mode='C_ref'),
    'ident_cp':    dict(supp=0.06, exc=0.12, r_wave=2, mode='C_perturb'),
}

RESULTS_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results', 'paper55_results.json')
ANALYSIS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results', 'paper55_analysis.json')


# ── FastVCML (C_ref write mode) ────────────────────────────────────────────────
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


# ── FastVCML with C_perturb write mode ────────────────────────────────────────
class FastVCMLCP(FastVCML):
    """C_perturb: only write during active wave events (wa > 0.05)."""
    def step(self, wa, wc):
        nb = self._nbmean(); an = np.minimum(1., self.age/300.)
        x = np.stack([nb, np.minimum(1., wa), an], 1)
        hn, out = self._gru(x); self.hid = hn
        self.vals = np.clip(VALS_DECAY*self.vals + VALS_NAV*nb + ADJ_SCALE*out, 0, 1)
        dev = hn - self.base_h; self.base_h += BASE_BETA * dev
        self.streak = np.where(np.sum(dev**2, 1) < .0025, self.streak+1, 0)
        self.mid = (self.mid + FA*dev) * MID_DECAY
        wg = (self.streak >= self.SS) & (wa > 0.05)
        if wg.any():
            self.fieldM[wg] += FA * (self.mid[wg] - self.fieldM[wg])
        self.fieldM *= FIELD_DECAY
        if self.t % DIFFUSE_EVERY == 0: self._diffuse()
        self.age += 1; self._collapse(); self.t += 1


# ── Wave environment ────────────────────────────────────────────────────────────
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


# ── C_order and time-series metrics ────────────────────────────────────────────
def compute_snapshot(vcml):
    """Compute C_order and related metrics at current timestep."""
    HALF = vcml.HALF; n_zones = N_ZONES; zw = HALF // n_zones
    means = []; spreads = []
    for z in range(n_zones):
        xlo = HALF + z*zw; xhi = HALF + (z+1)*zw
        mask = (vcml.ai_x >= xlo) & (vcml.ai_x < xhi)
        fm = vcml.fieldM[mask]
        means.append(fm.mean(0) if len(fm) else np.zeros(HS))
        spreads.append(float(np.std(fm)) if len(fm) > 1 else 0.0)

    adj_d = []; nadj_d = []
    for a in range(n_zones):
        for b in range(a+1, n_zones):
            d = float(np.linalg.norm(means[a] - means[b]))
            (adj_d if abs(a-b) == 1 else nadj_d).append(d)

    D_adj   = float(np.mean(adj_d))   if adj_d  else 0.0
    D_nonadj= float(np.mean(nadj_d))  if nadj_d else 0.0
    sg4     = float(np.mean(adj_d + nadj_d)) if (adj_d or nadj_d) else 0.0
    sigma_w = float(np.mean(spreads)) if spreads else 0.0
    na      = float(D_nonadj / D_adj) if D_adj > 1e-8 else float('nan')
    # C_order: only valid when sigma_w is non-trivial (fieldM non-zero)
    c_order = float((D_nonadj - D_adj) / sigma_w) if sigma_w > 1e-5 else float('nan')
    return dict(D_adj=D_adj, D_nonadj=D_nonadj, sg4=sg4,
                sigma_w=sigma_w, na=na, c_order=c_order)


# ── Run one regime/seed combination ────────────────────────────────────────────
def run_regime(seed, regime_name):
    cfg  = REGIMES[regime_name]
    half = S1['HALF']; h = S1['H']; WR = S1['WR']

    if cfg['mode'] == 'C_perturb':
        vcml = FastVCMLCP(seed, half=half, h=h)
    else:
        vcml = FastVCML(seed, half=half, h=h)

    env = WaveEnv(vcml, WR, supp_amp=cfg['supp'], exc_amp=cfg['exc'],
                  r_wave=cfg['r_wave'], rng_seed=seed * 1000)

    traj = []
    for t in range(STEPS):
        wa, wc = env.step()
        vcml.step(wa, wc)
        if (t + 1) % TRACK_EVERY == 0:
            snap = compute_snapshot(vcml)
            snap['step'] = t + 1
            traj.append(snap)
    return traj


# ── Worker (module-level for Windows spawn) ────────────────────────────────────
def _worker(args):
    seed, regime_name = args
    return (seed, regime_name, run_regime(seed, regime_name))


# ── Aggregation helper ─────────────────────────────────────────────────────────
def mean_safe(vals):
    v = [x for x in vals if x is not None and not math.isnan(x)]
    return float(np.mean(v)) if v else float('nan')

def se_safe(vals):
    v = [x for x in vals if x is not None and not math.isnan(x)]
    return float(np.std(v) / math.sqrt(len(v))) if len(v) > 1 else 0.0


def aggregate(done):
    """Compute per-regime trajectories and drift statistics."""
    summary = {}
    for regime in REGIMES:
        trajs = [done[f'{regime}|{s}'] for s in SEEDS if f'{regime}|{s}' in done]
        if not trajs:
            continue
        n_tp = min(len(t) for t in trajs)
        steps = [trajs[0][tp]['step'] for tp in range(n_tp)]

        # Mean trajectories over seeds
        c_order_mean = [mean_safe([t[tp]['c_order'] for t in trajs]) for tp in range(n_tp)]
        c_order_se   = [se_safe( [t[tp]['c_order'] for t in trajs]) for tp in range(n_tp)]
        na_mean      = [mean_safe([t[tp]['na']      for t in trajs]) for tp in range(n_tp)]
        sg4_mean     = [mean_safe([t[tp]['sg4']     for t in trajs]) for tp in range(n_tp)]
        sigma_w_mean = [mean_safe([t[tp]['sigma_w'] for t in trajs]) for tp in range(n_tp)]

        # Drift per step in three windows: early / mid / late
        thirds = n_tp // 3
        windows = [(0, thirds), (thirds, 2*thirds), (2*thirds, n_tp)]
        drifts = []
        for lo, hi in windows:
            seg = [x for x in c_order_mean[lo:hi] if not math.isnan(x)]
            if len(seg) >= 2:
                delta_c = seg[-1] - seg[0]
                delta_t = (hi - lo) * TRACK_EVERY
                drifts.append(float(delta_c / delta_t))
            else:
                drifts.append(float('nan'))

        # Final-state summary
        final_na = mean_safe([t[-1]['na']      for t in trajs])
        final_co = mean_safe([t[-1]['c_order'] for t in trajs])
        # Time of C_order crossing zero (first tp where c_order_mean > 0)
        cross_t  = next((steps[i] for i, v in enumerate(c_order_mean)
                         if not math.isnan(v) and v > 0), float('nan'))

        summary[regime] = dict(
            steps         = steps,
            c_order_mean  = c_order_mean,
            c_order_se    = c_order_se,
            na_mean       = na_mean,
            sg4_mean      = sg4_mean,
            sigma_w_mean  = sigma_w_mean,
            drift_windows = drifts,   # [early, mid, late] drift per step
            final_na      = final_na,
            final_c_order = final_co,
            cross_step    = cross_t,  # step where C_order first > 0
        )
    return summary


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
    for regime in REGIMES:
        for seed in SEEDS:
            k = f'{regime}|{seed}'
            if k not in done:
                all_tasks.append((seed, regime))

    print(f"Tasks to run: {len(all_tasks)}")
    if not all_tasks:
        print("All cached.")
    else:
        n_proc = min(len(all_tasks), mp.cpu_count())
        with mp.Pool(processes=n_proc) as pool:
            for res in pool.imap_unordered(_worker, all_tasks):
                seed, regime, traj = res
                k = f'{regime}|{seed}'
                done[k] = traj
                final = traj[-1] if traj else {}
                co = final.get('c_order', float('nan'))
                na = final.get('na',      float('nan'))
                co_s = f'{co:.3f}' if not math.isnan(co) else 'nan'
                na_s = f'{na:.3f}' if not math.isnan(na) else 'nan'
                print(f"  {k}: c_order={co_s} na={na_s}")
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(done, f, indent=2)

    # Aggregate
    summary = aggregate(done)
    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {ANALYSIS_FILE}")

    print("\n=== Lyapunov Drift Summary ===")
    print(f"{'Regime':<15} {'final C_order':>14} {'final na':>10} "
          f"{'drift[early]':>14} {'drift[mid]':>12} {'drift[late]':>12} {'cross_step':>12}")
    for regime, s in summary.items():
        dw = s['drift_windows']
        def fs(x): return f'{x:.4f}' if not math.isnan(x) else '  nan '
        cs = f"{s['cross_step']:.0f}" if not math.isnan(s['cross_step']) else 'never'
        print(f"  {regime:<15} {fs(s['final_c_order']):>14} "
              f"{fs(s['final_na']):>10} {fs(dw[0]):>14} {fs(dw[1]):>12} "
              f"{fs(dw[2]):>12} {cs:>12}")


if __name__ == '__main__':
    main()
