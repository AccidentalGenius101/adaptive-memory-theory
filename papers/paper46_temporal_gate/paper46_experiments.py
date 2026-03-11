"""
Paper 46: Temporal Gate Invariant and Causal Coordinate Validation

Two experiments:

Exp A (30 runs): Temporal gate -- does quiescence timing determine what fieldM
  encodes, independent of total write quantity?
  6 conditions x 5 seeds = 30 runs, STEPS=3000.
  Conditions:
    ss0:      SS=0  (always write -- timing removed, quantity maximized)
    ss5:      SS=5  (soft gate)
    ss10:     SS=10 (standard)
    ss15:     SS=15 (tight gate)
    ss20:     SS=20 (tightest gate -- minimum quantity)
    rand_p60: random write p=0.60 (matched quantity, timing destroyed)

  Key comparison: ss10 vs rand_p60.
    Same expected write rate; one is gated to quiescence, one is random.
    Theory predicts: ss10 > rand_p60 (timing determines what survives, not quantity).

  Also measured: n_writes (actual consolidation count per cell per step),
    sg4, nonadj/adj ratio.

Exp B (30 runs): phi_w validation -- does sg4 scale with Omega = WR * phi_w?
  r_wave in {1, 2, 3} x {omega_const, wr_const} x 5 seeds = 30 runs.
    omega_const: WR adjusted so WR * phi_w(r) = 62.4 (standard 4.8*13).
    wr_const:    WR = 4.8 fixed.
  phi_w(r) = 2r^2 + 2r + 1 (Manhattan diamond cell count).

  Theory predicts: sg4(omega_const) approx constant across r_wave.
  Control:         sg4(wr_const)    varies with r_wave.

Analysis output:
  Table A: condition | mean sg4 | mean write_rate | sg4/sg4_ss10
  Table B: r_wave | phi_w | WR(omega) | sg4(omega) | sg4(wr_const)
"""
import numpy as np, json, os, math, random, multiprocessing as mp
from collections import defaultdict

# ── Fixed constants (identical to Papers 43-44) ───────────────────────────────
W=80; H=40; HALF=40
HS=2; IS=3
WAVE_DUR=15
SUPP_AMP=0.25; EXC_AMP=0.50
ZONE_K=320; N_ZONES=4; ZW=HALF//N_ZONES    # ZW=10
MID_DECAY=0.99; FIELD_DECAY=0.9997; BASE_BETA=0.005
FA=0.16
VALS_DECAY=0.92; VALS_NAV=0.08; ADJ_SCALE=0.03
BOUND_LO=0.05; BOUND_HI=0.95; FRAG_NOISE=0.01; SEED_BETA=0.25
INST_THRESH=0.45; INST_PROB=0.03; DIFFUSE=0.02; DIFFUSE_EVERY=2

STEPS     = 3000
SEEDS     = list(range(5))
WARMUP    = 300
TAIL_STEPS= 30           # average sg4 over last TAIL_STEPS samples
SAMPLE_EVERY = 20        # sample metrics every N steps

# Standard wave parameters
WR_STD    = 4.8
R_WAVE_STD = 2           # Manhattan radius

def phi_w(r):
    """Manhattan diamond cell count at radius r: 2r^2 + 2r + 1."""
    return 2*r*r + 2*r + 1

# Omega = WR * phi_w(r)  [wave events * cells per event; proxy for causal dose]
OMEGA_STD = WR_STD * phi_w(R_WAVE_STD)   # 4.8 * 13 = 62.4

def wr_omega_const(r):
    """WR that keeps WR * phi_w = OMEGA_STD."""
    return OMEGA_STD / phi_w(r)

# ── Exp A conditions ──────────────────────────────────────────────────────────
A_CONDITIONS = {
    'ss0':      {'ss': 0,   'rand_gate': False, 'p_write': None},
    'ss5':      {'ss': 5,   'rand_gate': False, 'p_write': None},
    'ss10':     {'ss': 10,  'rand_gate': False, 'p_write': None},  # standard
    'ss15':     {'ss': 15,  'rand_gate': False, 'p_write': None},
    'ss20':     {'ss': 20,  'rand_gate': False, 'p_write': None},
    'rand_p60': {'ss': None,'rand_gate': True,  'p_write': 0.60},
}

# ── Exp B r_wave sweep ────────────────────────────────────────────────────────
R_WAVE_SWEEP = [1, 2, 3]

RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper46_results.json")


# ── FastVCML with parametrised SS and r_wave ──────────────────────────────────
class FastVCML:
    def __init__(self, seed, ss=10, rand_gate=False, p_write=None):
        self.rng       = np.random.RandomState(seed)
        self.SS        = ss
        self.rand_gate = rand_gate
        self.p_write   = p_write if p_write is not None else 0.60
        N = HALF * H
        self.N=N; self.t=0
        self.ai_x = HALF + (np.arange(N) % HALF)
        self.ai_y = np.arange(N) // HALF
        rng=self.rng; sc=0.3
        self.Wz=rng.randn(N,HS,IS)*sc; self.Uz=rng.randn(N,HS,HS)*sc; self.bz=np.full((N,HS),.5)
        self.Wr=rng.randn(N,HS,IS)*sc; self.Ur=rng.randn(N,HS,HS)*sc; self.br=np.full((N,HS),.5)
        self.Wh=rng.randn(N,HS,IS)*sc; self.Uh=rng.randn(N,HS,HS)*sc; self.bh=np.zeros((N,HS))
        self.Wo=rng.randn(N,HS)*0.1;   self.bo=np.zeros(N)
        self.vals=rng.uniform(.3,.7,N); self.hid=rng.randn(N,HS)*.1
        self.age=rng.randint(0,50,N).astype(float)
        self.base_h=np.zeros((N,HS)); self.mid=np.zeros((N,HS))
        self.fieldM=np.zeros((N,HS)); self.streak=np.zeros(N,int)
        self.cc=np.zeros(N,int)
        self.n_writes = 0   # total consolidation events (for write-rate diagnostic)
        self._build_nb()

    def _build_nb(self):
        N=self.N; rx=np.arange(N)%HALF; y=np.arange(N)//HALF
        nb=np.full((N,4),-1,int)
        nb[rx>0,0]=np.where(rx>0)[0]-1; nb[rx<HALF-1,1]=np.where(rx<HALF-1)[0]+1
        nb[y>0,2]=np.where(y>0)[0]-HALF; nb[y<H-1,3]=np.where(y<H-1)[0]+HALF
        self.nb=nb; self.nbc=(nb>=0).sum(1).astype(float)

    def _nbmean(self):
        ns=np.maximum(self.nb,0); g=self.vals[ns]; g[self.nb<0]=0.0
        return np.where(self.nbc>0,g.sum(1)/self.nbc,.5)

    def _gru(self,x):
        h=self.hid
        def sig(a): return 1/(1+np.exp(-np.clip(a,-8,8)))
        def tanh(a): e2=np.exp(2*np.clip(a,-8,8)); return (e2-1)/(e2+1)
        x3,h3=x[:,:,None],h[:,:,None]
        z=sig((self.Wz@x3).squeeze(-1)+(self.Uz@h3).squeeze(-1)+self.bz)
        r=sig((self.Wr@x3).squeeze(-1)+(self.Ur@h3).squeeze(-1)+self.br)
        rh=(r*h)[:,:,None]
        g=tanh((self.Wh@x3).squeeze(-1)+(self.Uh@rh).squeeze(-1)+self.bh)
        hn=(1-z)*h+z*g; out=np.tanh(np.einsum('ni,ni->n',self.Wo,hn)+self.bo)
        return hn,out

    def _diffuse(self):
        ns=np.maximum(self.nb,0); nf=self.fieldM[ns]
        nf*=(self.nb>=0)[:,:,None]
        nm=np.where(self.nbc[:,None]>0,nf.sum(1)/self.nbc[:,None],self.fieldM)
        self.fieldM+=DIFFUSE*(nm-self.fieldM)

    def step(self,wa,wc):
        nb=self._nbmean(); an=np.minimum(1.,self.age/300.)
        x=np.stack([nb,np.minimum(1.,wa),an],1)
        hn,out=self._gru(x); self.hid=hn
        self.vals=np.clip(VALS_DECAY*self.vals+VALS_NAV*nb+ADJ_SCALE*out,0,1)
        dev=hn-self.base_h; self.base_h+=BASE_BETA*dev
        self.streak=np.where(np.sum(dev**2,1)<.0025,self.streak+1,0)
        self.mid=(self.mid+FA*dev)*MID_DECAY

        # Gate condition: quiescence-gated OR random
        if self.rand_gate:
            gate = self.rng.random(self.N) < self.p_write
        elif self.SS == 0:
            gate = np.ones(self.N, bool)
        else:
            gate = self.streak >= self.SS

        self.fieldM[gate] += FA*(self.mid[gate]-self.fieldM[gate])
        self.n_writes += int(gate.sum())
        self.fieldM *= FIELD_DECAY
        if self.t % DIFFUSE_EVERY == 0: self._diffuse()
        self.age += 1
        self._collapse()
        self.t += 1

    def _collapse(self):
        rng=self.rng
        bad=(self.vals<BOUND_LO)|(self.vals>BOUND_HI)
        inst=(np.sum(np.abs(self.hid-self.base_h),1)>INST_THRESH)&(rng.random(self.N)<INST_PROB)
        ci=np.where(bad|inst)[0]
        if not len(ci): return
        prev=self.hid.copy()
        for ai in ci:
            self.cc[ai]+=1; fm=self.fieldM[ai]; mag=np.sqrt(np.dot(fm,fm))
            nh=((1-SEED_BETA)*prev[ai]+SEED_BETA*fm if mag>1e-6 else prev[ai].copy())
            nh+=rng.randn(HS)*FRAG_NOISE
            self.hid[ai]=nh; self.vals[ai]=.5; self.age[ai]=0
            self.streak[ai]=0; self.mid[ai]=np.zeros(HS)


# ── Wave environment with parametrised r_wave ─────────────────────────────────
class WaveEnv:
    def __init__(self, vcml, WR, rng_seed=0, r_wave=2):
        self.vcml=vcml; self.WR=WR; self.r_wave=r_wave
        self.rng=random.Random(rng_seed)
        self.waves=[]
        N=vcml.N
        self._wa=np.empty(N); self._wc=np.empty(N,int)
        self.ax=vcml.ai_x; self.ay=vcml.ai_y

    def _launch(self):
        cls = self.rng.randint(0, N_ZONES-1)
        zlo = cls*ZW; zhi = (cls+1)*ZW-1
        sx = self.rng.randint(zlo, zhi)
        cx = HALF + sx; cy = self.rng.randint(0, H-1)
        self.waves.append([cx, cy, WAVE_DUR, cls])

    def _apply(self):
        wa=self._wa; wc=self._wc
        wa.fill(0.); wc.fill(-1)
        ax=self.ax; ay=self.ay; rw=self.r_wave; surv=[]
        for wave in self.waves:
            cx,cy,rem,cls=wave
            if rem<=0: continue
            dist=np.abs(ax-cx)+np.abs(ay-cy)
            act=np.maximum(0.,1.-dist/rw); act[dist>rw]=0.
            better=act>wa; wa[better]=act[better]; wc[better]=cls
            wave[2]-=1
            if wave[2]>0: surv.append(wave)
        self.waves=surv
        for even in [True, False]:
            amp = SUPP_AMP if even else EXC_AMP
            suppress = even
            par = (wc%2==0) if even else (wc%2==1)
            idx = np.where((wc>=0)&par&(wa>.05))[0]
            if not len(idx): continue
            sc = wa[idx]*amp
            if suppress:
                self.vcml.vals[idx]=np.maximum(0., self.vcml.vals[idx]*(1.-sc*.5))
            else:
                self.vcml.vals[idx]=np.minimum(1., self.vcml.vals[idx]+sc*.5)
        return wa.copy(), wc.copy()

    def step(self):
        exp=self.WR/WAVE_DUR
        nl=int(exp)+(1 if self.rng.random()<exp-int(exp) else 0)
        for _ in range(nl): self._launch()
        return self._apply()


# ── Metrics ───────────────────────────────────────────────────────────────────
def sg4_fn(vcml):
    """Mean pairwise L2 between zone-mean fieldM vectors."""
    zw = HALF // N_ZONES
    zmeans = []
    for z in range(N_ZONES):
        xlo = HALF + z*zw; xhi = HALF + (z+1)*zw
        mask = (vcml.ai_x >= xlo) & (vcml.ai_x < xhi)
        fm = vcml.fieldM[mask]
        zmeans.append(fm.mean(0) if len(fm) else np.zeros(HS))
    dists = []
    for i in range(N_ZONES):
        for j in range(i+1, N_ZONES):
            d = np.linalg.norm(zmeans[i]-zmeans[j])
            dists.append(d)
    return float(np.mean(dists))


def nonadj_adj_fn(vcml):
    """Ratio of non-adjacent to adjacent zone-pair mean distances."""
    zw = HALF // N_ZONES
    zmeans = []
    for z in range(N_ZONES):
        xlo = HALF + z*zw; xhi = HALF + (z+1)*zw
        mask = (vcml.ai_x >= xlo) & (vcml.ai_x < xhi)
        fm = vcml.fieldM[mask]
        zmeans.append(fm.mean(0) if len(fm) else np.zeros(HS))
    adj_dists=[]; nonadj_dists=[]
    for i in range(N_ZONES):
        for j in range(i+1, N_ZONES):
            d = float(np.linalg.norm(zmeans[i]-zmeans[j]))
            if j-i==1: adj_dists.append(d)
            else:       nonadj_dists.append(d)
    if not adj_dists or not nonadj_dists: return float('nan')
    return float(np.mean(nonadj_dists)/np.mean(adj_dists))


# ── Run function ──────────────────────────────────────────────────────────────
def run(seed, ss=10, rand_gate=False, p_write=None, wr=WR_STD, r_wave=R_WAVE_STD):
    vcml = FastVCML(seed, ss=ss, rand_gate=rand_gate, p_write=p_write)
    env  = WaveEnv(vcml, WR=wr, rng_seed=seed+1000, r_wave=r_wave)
    sg4_trace = []
    for t in range(STEPS):
        wa, wc = env.step()
        vcml.step(wa, wc)
        if t >= WARMUP and t % SAMPLE_EVERY == 0:
            sg4_trace.append(sg4_fn(vcml))
    sg4_tail = float(np.mean(sg4_trace[-TAIL_STEPS:])) if sg4_trace else float('nan')
    write_rate = vcml.n_writes / (vcml.N * STEPS)
    coll_per_site = float(vcml.cc.sum()) / vcml.N
    na_ratio = nonadj_adj_fn(vcml)
    return {
        'sg4':        sg4_tail,
        'write_rate': write_rate,
        'coll_site':  coll_per_site,
        'na_ratio':   na_ratio,
    }


# ── Worker ────────────────────────────────────────────────────────────────────
def _worker(args):
    tag, seed, params = args
    result = run(seed, **params)
    return tag, seed, result


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.freeze_support()

    # Load existing results
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    # Build task list
    todo = []

    # Exp A: temporal gate sweep
    for cond_name, cparams in A_CONDITIONS.items():
        for seed in SEEDS:
            key = f"A,{cond_name},{seed}"
            if key not in results:
                params = {
                    'ss':        cparams['ss'] if not cparams['rand_gate'] else 10,
                    'rand_gate': cparams['rand_gate'],
                    'p_write':   cparams['p_write'],
                    'wr':        WR_STD,
                    'r_wave':    R_WAVE_STD,
                }
                todo.append(('A', seed, params, key))

    # Exp B: phi_w (r_wave x WR) sweep
    for r in R_WAVE_SWEEP:
        for cond in ['omega_const', 'wr_const']:
            wr = wr_omega_const(r) if cond == 'omega_const' else WR_STD
            for seed in SEEDS:
                key = f"B,r{r},{cond},{seed}"
                if key not in results:
                    params = {
                        'ss': 10, 'rand_gate': False, 'p_write': None,
                        'wr': wr, 'r_wave': r,
                    }
                    todo.append(('B', seed, params, key))

    print(f"Total tasks: {len(todo)} (skipping {len(results)} cached)")

    if todo:
        worker_args = [(t[0], t[1], t[2]) for t in todo]
        with mp.Pool(processes=min(len(todo), mp.cpu_count())) as pool:
            for (exp_tag, seed, r), (tag, s, result) in zip(
                    [(t[0],t[1],t[3]) for t in todo],
                    pool.map(_worker, worker_args)):
                results[s] = result   # temp; properly keyed below
        # Re-run properly with keys
        results_new = {}
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE) as f:
                results_new = json.load(f)
        with mp.Pool(processes=min(len(todo), mp.cpu_count())) as pool:
            raw = pool.map(_worker, worker_args)
        for (_, _, _, key), (_, _, result) in zip(todo, raw):
            results_new[key] = result
        results = results_new
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} results -> {RESULTS_FILE}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    import numpy as _np
    from scipy import stats as _stats

    print("\n=== Exp A: Temporal Gate Test ===")
    ss10_vals = []
    rand_vals  = []
    print(f"{'Condition':<12} {'sg4 mean':>10} {'sg4 SE':>8} {'write_rate':>12} "
          f"{'sg4/ss10':>10}")
    a_summary = {}
    for cond_name in ['ss0','ss5','ss10','ss15','ss20','rand_p60']:
        sg4s = [results[f"A,{cond_name},{s}"]['sg4']
                for s in SEEDS if f"A,{cond_name},{s}" in results]
        wrs  = [results[f"A,{cond_name},{s}"]['write_rate']
                for s in SEEDS if f"A,{cond_name},{s}" in results]
        if not sg4s: continue
        m = _np.mean(sg4s); se = _np.std(sg4s)/_np.sqrt(len(sg4s))
        wr_m = _np.mean(wrs)
        a_summary[cond_name] = {'sg4': sg4s, 'wr': wrs}
        if cond_name == 'ss10': ss10_vals = sg4s
        if cond_name == 'rand_p60': rand_vals = sg4s
        print(f"{cond_name:<12} {m:>10.4f} {se:>8.4f} {wr_m:>12.4f}      --")

    # Re-print with ratio once ss10_vals is known
    ss10_mean = _np.mean(ss10_vals) if ss10_vals else 1.0
    print()
    for cond_name in ['ss0','ss5','ss10','ss15','ss20','rand_p60']:
        if cond_name not in a_summary: continue
        sg4s = a_summary[cond_name]['sg4']
        wrs  = a_summary[cond_name]['wr']
        m = _np.mean(sg4s); se = _np.std(sg4s)/_np.sqrt(len(sg4s))
        wr_m = _np.mean(wrs)
        ratio = m / ss10_mean
        print(f"{cond_name:<12} {m:>10.4f} {se:>8.4f} {wr_m:>12.4f} {ratio:>10.3f}")

    if ss10_vals and rand_vals:
        t, p = _stats.ttest_ind(ss10_vals, rand_vals)
        print(f"\nss10 vs rand_p60: t={t:.3f}, p={p:.4g}")
        if _np.mean(ss10_vals) > _np.mean(rand_vals) and p < 0.05:
            print("-> TIMING IS OPERATIVE: quiescence gate outperforms random gate at matched quantity")
        elif p >= 0.05:
            print("-> NO SIGNIFICANT DIFFERENCE: timing may not be the key variable")
        else:
            print("-> rand_p60 > ss10: unexpected result")

    print("\n=== Exp B: phi_w (Omega-constant) Validation ===")
    print(f"Omega_std = {OMEGA_STD:.1f} (WR={WR_STD} x phi_w={phi_w(R_WAVE_STD)})")
    print(f"\n{'r_wave':>7} {'phi_w':>6} {'WR_omega':>10} "
          f"{'sg4_omega':>10} {'sg4_wr4.8':>10} {'ratio':>8}")
    sg4_omega_all = []
    for r in R_WAVE_SWEEP:
        pw = phi_w(r)
        wr_o = wr_omega_const(r)
        sg4_o = [results[f"B,r{r},omega_const,{s}"]['sg4']
                 for s in SEEDS if f"B,r{r},omega_const,{s}" in results]
        sg4_w = [results[f"B,r{r},wr_const,{s}"]['sg4']
                 for s in SEEDS if f"B,r{r},wr_const,{s}" in results]
        if not sg4_o or not sg4_w: continue
        mo = _np.mean(sg4_o); mw = _np.mean(sg4_w)
        sg4_omega_all.append(mo)
        print(f"{r:>7} {pw:>6} {wr_o:>10.2f} {mo:>10.4f} {mw:>10.4f} "
              f"{mo/mw:>8.3f}")

    if len(sg4_omega_all) == len(R_WAVE_SWEEP):
        cv = _np.std(sg4_omega_all)/_np.mean(sg4_omega_all)
        print(f"\nomega_const CV = {cv:.3f} "
              f"({'FLAT - Omega is operative' if cv < 0.15 else 'NOT FLAT - Omega insufficient'})")

    # Save analysis summary to JSON
    analysis = {'exp_a': {}, 'exp_b': {}}
    for cond_name in A_CONDITIONS:
        sg4s = [results[f"A,{cond_name},{s}"]['sg4']
                for s in SEEDS if f"A,{cond_name},{s}" in results]
        wrs  = [results[f"A,{cond_name},{s}"]['write_rate']
                for s in SEEDS if f"A,{cond_name},{s}" in results]
        if sg4s:
            analysis['exp_a'][cond_name] = {
                'sg4_mean': float(_np.mean(sg4s)),
                'sg4_se':   float(_np.std(sg4s)/_np.sqrt(len(sg4s))),
                'wr_mean':  float(_np.mean(wrs)),
                'sg4_vals': [float(v) for v in sg4s],
            }
    for r in R_WAVE_SWEEP:
        for cond in ['omega_const', 'wr_const']:
            sg4s = [results[f"B,r{r},{cond},{s}"]['sg4']
                    for s in SEEDS if f"B,r{r},{cond},{s}" in results]
            if sg4s:
                k = f"r{r}_{cond}"
                analysis['exp_b'][k] = {
                    'r_wave': r, 'phi_w': phi_w(r),
                    'wr': float(wr_omega_const(r) if cond=='omega_const' else WR_STD),
                    'sg4_mean': float(_np.mean(sg4s)),
                    'sg4_se':   float(_np.std(sg4s)/_np.sqrt(len(sg4s))),
                    'sg4_vals': [float(v) for v in sg4s],
                }
    if ss10_vals and rand_vals:
        t, p = _stats.ttest_ind(ss10_vals, rand_vals)
        analysis['exp_a']['ss10_vs_rand_t'] = float(t)
        analysis['exp_a']['ss10_vs_rand_p'] = float(p)

    analysis_file = os.path.join(os.path.dirname(__file__),
                                 "results", "paper46_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis -> {analysis_file}")
