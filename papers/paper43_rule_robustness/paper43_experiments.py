"""
paper43_experiments.py -- Rule-Family Robustness of VCML

Addresses hostile-reviewer concern: is the copy-forward/viability-gating
phenomenon real, or is it fragile rule engineering specific to one implementation?

Strategy: 4 ablation conditions (remove each load-bearing primitive) + 6 nearby
rule variants (perturb parameters +-50%). Prediction:
  - Ablations (no_birth_seed, no_gating, no_both): sg4n collapses AND nonadj/adj ~= 1
  - Variants (ss_5, ss_20, mid_095, mid_0999, fsb_005, fsb_050): sg4n persists

If sg4n persists across all 6 variants but collapses in both ablations, this
demonstrates: (1) copy-forward and viability-gating are necessary, and (2) the
mechanism is robust to rule perturbation -- i.e., not fragile engineering.

10 conditions x 5 seeds = 50 runs. T=3000, CPS every 300.

Measures per condition:
  sg4n        -- normalized zone separation (primary metric, Papers 0-42)
  nonadj/adj  -- spatial specificity ratio (independent metric, no fieldM scale)
  cc_final    -- total collapse count (confirms mechanism activity)
"""
import numpy as np, json, os, math, random, multiprocessing as mp

# ── Simulation constants (identical to paper42) ───────────────────────────────
W=80; H=40; HALF=40
HS=2; IS=3
WAVE_DUR=15
SUPP_AMP=0.25; EXC_AMP=0.50
ZONE_K=320
FIELD_DECAY=0.9997; BASE_BETA=0.005
FA_STD=0.16
VALS_DECAY=0.92; VALS_NAV=0.08; ADJ_SCALE=0.03
BOUND_LO=0.05; BOUND_HI=0.95; FRAG_NOISE=0.01
INST_THRESH=0.45; INST_PROB=0.03; DIFFUSE=0.02; DIFFUSE_EVERY=2

WR_FIXED=4.8; N_ZONES=4; ZW=HALF//N_ZONES  # zw=10

SEEDS = list(range(5))
T_END = 3000
CPS   = list(range(300, T_END+1, 300))
CPS_SET = set(CPS)

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper43_results.json")

# ── Experiment B: persistence (waves stop at T=1500) ─────────────────────────
# Tests 4 conditions: ref, no_birth_seed, no_gating, no_both
# Prediction: ref maintains structure; ablations decay toward field-decay-only
T_WAVES   = 1500   # waves run until here, then WR -> 0
T_END_B   = 3000
CPS_B     = list(range(300, T_END_B+1, 300))
CPS_B_SET = set(CPS_B)
PERSIST_CONDS = ['ref', 'no_birth_seed', 'no_gating', 'no_both']
FIELD_DECAY_FACTOR = 0.9997  # for reference line

# ── Condition table ───────────────────────────────────────────────────────────
# (ss, mid_decay, seed_beta, no_gating)
CONDITIONS = {
    # Reference
    'ref':           (10,   0.99,    0.25,  False),
    # Core-mechanism ablations (expected: sg4n collapses)
    'no_birth_seed': (10,   0.99,    0.00,  False),  # copy-forward removed
    'no_gating':     (10,   0.99,    0.25,  True),   # viability gate removed
    'no_both':       (10,   0.99,    0.00,  True),   # both removed
    # Rule variants (expected: sg4n persists near ref)
    'ss_5':          (5,    0.99,    0.25,  False),  # faster consolidation
    'ss_20':         (20,   0.99,    0.25,  False),  # slower consolidation
    'mid_095':       (10,   0.95,    0.25,  False),  # faster gate decay
    'mid_0999':      (10,   0.9999,  0.25,  False),  # slower gate decay
    'fsb_005':       (10,   0.99,    0.05,  False),  # weak inheritance
    'fsb_050':       (10,   0.99,    0.50,  False),  # strong inheritance
}

ABLATION_CONDS = {'no_birth_seed', 'no_gating', 'no_both'}
VARIANT_CONDS  = {'ss_5', 'ss_20', 'mid_095', 'mid_0999', 'fsb_005', 'fsb_050'}


# ── Configurable FastVCML ─────────────────────────────────────────────────────
class ConfigFastVCML:
    """FastVCML with configurable SS, MID_DECAY, SEED_BETA, and optional
    no_gating (all cells always consolidate, ignoring calm-streak requirement)."""

    def __init__(self, seed, fa=FA_STD,
                 ss=10, mid_decay=0.99, seed_beta=0.25, no_gating=False):
        self.rng       = np.random.RandomState(seed)
        self.fa        = fa
        self.SS        = ss
        self.MID_DECAY = mid_decay
        self.SEED_BETA = seed_beta
        self.no_gating = no_gating
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

    def _gru(self, x):
        h=self.hid
        def sig(a): return 1/(1+np.exp(-np.clip(a,-8,8)))
        def tanh(a): e2=np.exp(2*np.clip(a,-8,8)); return (e2-1)/(e2+1)
        x3,h3=x[:,:,None],h[:,:,None]
        z=sig((self.Wz@x3).squeeze(-1)+(self.Uz@h3).squeeze(-1)+self.bz)
        r=sig((self.Wr@x3).squeeze(-1)+(self.Ur@h3).squeeze(-1)+self.br)
        rh=(r*h)[:,:,None]
        g=tanh((self.Wh@x3).squeeze(-1)+(self.Uh@rh).squeeze(-1)+self.bh)
        hn=(1-z)*h+z*g; out=np.tanh(np.einsum('ni,ni->n',self.Wo,hn)+self.bo)
        return hn, out

    def _diffuse(self):
        ns=np.maximum(self.nb,0); nf=self.fieldM[ns]
        nf*=(self.nb>=0)[:,:,None]
        nm=np.where(self.nbc[:,None]>0,nf.sum(1)/self.nbc[:,None],self.fieldM)
        self.fieldM+=DIFFUSE*(nm-self.fieldM)

    def step(self, wa, wc):
        nb=self._nbmean(); an=np.minimum(1.,self.age/300.)
        x=np.stack([nb,np.minimum(1.,wa),an],1)
        hn,out=self._gru(x); self.hid=hn
        self.vals=np.clip(VALS_DECAY*self.vals+VALS_NAV*nb+ADJ_SCALE*out,0,1)
        dev=hn-self.base_h; self.base_h+=BASE_BETA*dev
        self.streak=np.where(np.sum(dev**2,1)<.0025,self.streak+1,0)
        fa=self.fa
        self.mid=(self.mid+fa*dev)*self.MID_DECAY
        gate = np.ones(self.N, bool) if self.no_gating else (self.streak >= self.SS)
        self.fieldM[gate]+=fa*(self.mid[gate]-self.fieldM[gate])
        self.fieldM*=FIELD_DECAY
        if self.t%DIFFUSE_EVERY==0: self._diffuse()
        self.age+=1; self._collapse(); self.t+=1

    def _collapse(self):
        rng=self.rng
        bad=(self.vals<BOUND_LO)|(self.vals>BOUND_HI)
        inst=(np.sum(np.abs(self.hid-self.base_h),1)>INST_THRESH)&(rng.random(self.N)<INST_PROB)
        ci=np.where(bad|inst)[0]
        if not len(ci): return
        prev=self.hid.copy()
        sb=self.SEED_BETA
        for ai in ci:
            self.cc[ai]+=1; fm=self.fieldM[ai]; mag=np.sqrt(np.dot(fm,fm))
            if sb > 0 and mag > 1e-6:
                nh=(1-sb)*prev[ai]+sb*fm
            else:
                nh=prev[ai].copy()
            nh+=rng.randn(HS)*FRAG_NOISE
            self.hid[ai]=nh; self.vals[ai]=.5; self.age[ai]=0
            self.streak[ai]=0; self.mid[ai]=np.zeros(HS)

    def stable_zone(self, k): return np.argsort(self.cc)[:min(k,self.N//5)]


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(vcml):
    """Return (sg4n, nonadj_adj_ratio) from current fieldM state."""
    si = vcml.stable_zone(ZONE_K)
    xp = vcml.ai_x[si]
    za = np.minimum(N_ZONES-1, (xp - HALF) // ZW)
    means = []; wvars = []
    for z in range(N_ZONES):
        zs = si[za==z]
        if len(zs):
            fm=vcml.fieldM[zs]; mz=fm.mean(0); means.append(mz)
            wvars.append(float(np.mean(np.sum((fm-mz)**2,1))))
        else:
            means.append(np.zeros(HS)); wvars.append(float('nan'))

    dists = {}
    for a in range(N_ZONES):
        for b in range(a+1, N_ZONES):
            dists[(a,b)] = float(np.linalg.norm(means[a]-means[b]))

    sg4 = float(np.mean(list(dists.values()))) if dists else 0.
    vv  = [v for v in wvars if not math.isnan(v)]
    sw  = float(np.sqrt(np.mean(vv))) if vv else float('nan')
    sg4n = float(sg4/sw) if sw>1e-8 else float('nan')

    # nonadj/adj: for 4 zones, adj=(0,1),(1,2),(2,3); nonadj=(0,2),(0,3),(1,3)
    adj_pairs    = [(0,1),(1,2),(2,3)]
    nonadj_pairs = [(0,2),(0,3),(1,3)]
    adj_d    = float(np.mean([dists[p] for p in adj_pairs]))
    nonadj_d = float(np.mean([dists[p] for p in nonadj_pairs]))
    ratio = float(nonadj_d / adj_d) if adj_d > 1e-8 else float('nan')

    return sg4n, ratio


# ── Wave environment (same as paper42) ───────────────────────────────────────
class WaveEnv:
    def __init__(self, vcml, WR, rng_seed=0):
        self.vcml=vcml; self.WR=WR
        self.rng=random.Random(rng_seed)
        self.waves=[]
        N=vcml.N
        self._wa=np.empty(N); self._wc=np.empty(N,int)
        self.ax=vcml.ai_x; self.ay=vcml.ai_y

    def _launch(self):
        cls = self.rng.randint(0, N_ZONES-1)
        zlo = cls*ZW; zhi = (cls+1)*ZW-1
        shifted_x = self.rng.randint(zlo, zhi)
        cx = HALF + shifted_x
        cy = self.rng.randint(0, H-1)
        self.waves.append([cx, cy, WAVE_DUR, cls])

    def _apply(self):
        wa=self._wa; wc=self._wc
        wa.fill(0.); wc.fill(-1)
        ax=self.ax; ay=self.ay; surv=[]
        for wave in self.waves:
            cx,cy,rem,cls=wave
            if rem<=0: continue
            dist=np.abs(ax-cx)+np.abs(ay-cy)
            act=np.maximum(0.,1.-dist*0.4); act[dist>2]=0.
            better=act>wa; wa[better]=act[better]; wc[better]=cls
            wave[2]-=1
            if wave[2]>0: surv.append(wave)
        self.waves=surv
        for even in [True, False]:
            amp=SUPP_AMP if even else EXC_AMP
            par=(wc%2==0) if even else (wc%2==1)
            idx=np.where((wc>=0)&par&(wa>.05))[0]
            if not len(idx): continue
            sc=wa[idx]*amp
            if even: self.vcml.vals[idx]=np.maximum(0., self.vcml.vals[idx]*(1.-sc*.5))
            else:    self.vcml.vals[idx]=np.minimum(1., self.vcml.vals[idx]+sc*.5)
        return wa.copy(), wc.copy()

    def step(self):
        exp=self.WR/WAVE_DUR
        nl=int(exp)+(1 if self.rng.random()<exp-int(exp) else 0)
        for _ in range(nl): self._launch()
        return self._apply()


# ── Experiment A: single run (continuous waves) ───────────────────────────────
def run_expA(seed, cond):
    ss, mid_decay, seed_beta, no_gating = CONDITIONS[cond]
    vcml = ConfigFastVCML(seed=seed*7+11,
                          ss=ss, mid_decay=mid_decay,
                          seed_beta=seed_beta, no_gating=no_gating)
    env  = WaveEnv(vcml, WR_FIXED, rng_seed=seed*200+37)
    result = {'exp': 'A', 'cond': cond, 'seed': seed,
              'sg4ns': [], 'ratios': [], 'ts': CPS}
    for t in range(T_END):
        wa, wc = env.step()
        vcml.step(wa, wc)
        if (t+1) in CPS_SET:
            sg4n, ratio = compute_metrics(vcml)
            result['sg4ns'].append(sg4n)
            result['ratios'].append(ratio)
    result['cc_final'] = int(np.sum(vcml.cc))
    return result


# ── Experiment B: persistence (waves stop at T=1500) ─────────────────────────
def run_expB(seed, cond):
    """Waves run T=0..1500, then stop. Measures sg4n decay under each rule variant.
    Copy-forward predicts ref maintains structure; ablations decay."""
    ss, mid_decay, seed_beta, no_gating = CONDITIONS[cond]
    vcml = ConfigFastVCML(seed=seed*7+99,
                          ss=ss, mid_decay=mid_decay,
                          seed_beta=seed_beta, no_gating=no_gating)
    env  = WaveEnv(vcml, WR_FIXED, rng_seed=seed*200+113)
    result = {'exp': 'B', 'cond': cond, 'seed': seed,
              'sg4ns': [], 'ratios': [], 'ts': CPS_B,
              'sg4n_at_stop': None}   # sg4n right when waves stop
    for t in range(T_END_B):
        if t == T_WAVES:
            env.WR = 0.0
            env.waves = []
            # Record sg4n exactly at the wave-stop moment
            sg4n_stop, _ = compute_metrics(vcml)
            result['sg4n_at_stop'] = float(sg4n_stop)
        wa, wc = env.step()
        vcml.step(wa, wc)
        if (t+1) in CPS_B_SET:
            sg4n, ratio = compute_metrics(vcml)
            result['sg4ns'].append(sg4n)
            result['ratios'].append(ratio)
    result['cc_final'] = int(np.sum(vcml.cc))
    return result


# ── Multiprocessing ───────────────────────────────────────────────────────────
def _worker(args):
    tag, seed, cond = args
    if tag == 'A': return run_expA(seed, cond)
    else:          return run_expB(seed, cond)


def key(exp, seed, cond): return f"{exp},{cond},{seed}"


if __name__ == '__main__':
    mp.freeze_support()

    # Load existing results; tag old unlabelled entries as exp='A'
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f: data = json.load(f)
        for r in data:
            if 'exp' not in r: r['exp'] = 'A'
    else:
        data = []

    done = {key(r['exp'], r['seed'], r['cond']) for r in data}

    # Exp A: all 10 conditions x 5 seeds
    todo_A = [('A', seed, cond)
              for cond in CONDITIONS
              for seed in SEEDS
              if key('A', seed, cond) not in done]

    # Exp B: persistence, 4 conditions x 5 seeds
    todo_B = [('B', seed, cond)
              for cond in PERSIST_CONDS
              for seed in SEEDS
              if key('B', seed, cond) not in done]

    todo = todo_A + todo_B
    print(f"Paper 43: {len(done)} done, {len(todo)} remaining "
          f"({len(todo_A)} Exp-A + {len(todo_B)} Exp-B)")

    if todo:
        with mp.Pool(processes=min(len(todo), mp.cpu_count())) as pool:
            results = pool.map(_worker, todo)
        data.extend(results)
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, 'w') as f: json.dump(data, f)
        print(f"Saved {len(data)} total results to {RESULTS_FILE}")
    else:
        print("All done.")

    # ── Summary printout ──────────────────────────────────────────────────────
    import numpy as _np

    def mn(lst):
        v=[x for x in lst if x is not None and not math.isnan(x)]
        return (_np.mean(v), _np.std(v)/_np.sqrt(len(v))) if v else (float('nan'), float('nan'))

    cond_order = ['ref',
                  'no_birth_seed', 'no_gating', 'no_both',
                  'ss_5', 'ss_20', 'mid_095', 'mid_0999', 'fsb_005', 'fsb_050']

    print("\n--- Exp A: sg4n and nonadj/adj at T=3000 (continuous waves) ---")
    print(f"{'Cond':<16} {'sg4n':>8} {'+-':>7} {'nadj/adj':>10} {'+-':>7}  Type")
    print("-"*60)
    for cond in cond_order:
        runs=[r for r in data if r['exp']=='A' and r['cond']==cond]
        if not runs: continue
        s,se   = mn([r['sg4ns'][-1] for r in runs])
        rat,rse= mn([r['ratios'][-1] for r in runs])
        t = 'ablation' if cond in ABLATION_CONDS else ('variant' if cond in VARIANT_CONDS else 'ref')
        print(f"{cond:<16} {s:8.4f} {se:7.4f} {rat:10.4f} {rse:7.4f}  {t}")

    print("\n--- Exp B: persistence after wave stop (T_waves=1500, measure at T=3000) ---")
    print(f"{'Cond':<16} {'sg4n@stop':>10} {'sg4n@3000':>10} {'+-':>7} {'ratio':>8}")
    print("-"*60)
    fd_pred = FIELD_DECAY_FACTOR ** (T_END_B - T_WAVES)
    print(f"  Field-decay only (ratio): {fd_pred:.4f}")
    for cond in PERSIST_CONDS:
        runs=[r for r in data if r['exp']=='B' and r['cond']==cond]
        if not runs: continue
        s3,se3 = mn([r['sg4ns'][-1] for r in runs])
        stops  = [r['sg4n_at_stop'] for r in runs if r.get('sg4n_at_stop') is not None]
        sstop,_= mn(stops)
        ratio  = s3/sstop if sstop>1e-6 else float('nan')
        print(f"{cond:<16} {sstop:10.4f} {s3:10.4f} {se3:7.4f} {ratio:8.4f}")
