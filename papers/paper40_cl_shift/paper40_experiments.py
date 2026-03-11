"""
paper40_experiments.py -- Real Continual Learning: Zone Boundary Shift

Unlike Paper 38 (polarity flip, invisible to sg4), these experiments change
the SPATIAL POSITIONS of zones mid-run -- a hard perturbation that genuinely
disrupts the committed fieldM structure.

Exp A: Zone boundary shift at T_SHIFT=1500 (before commitment epoch ~2000)
  Conditions (5 seeds each):
    C_ref    : no shift at all (control)
    C_half   : delta=5  (half zone width -- 50% of cells cross boundary)
    C_full   : delta=10 (full cyclic permutation -- 100% of cells in new zones)
  Metric: sg4n_new(t) and sg4n_old(t) at each checkpoint
  Prediction: sg4n_new drops to near-zero at T_SHIFT, then recovers;
              sg4n_old stays high then decays; recovery speed varies with delta.

Exp B: Interleaved switching from T=0 (no prior commitment)
  5 seeds. Switch between delta=0 (task A) and delta=5 (task B) every 500 steps.
  Metric: sg4n_A(t), sg4n_B(t) tracked separately.
  Prediction: neither task is strongly committed; modest but stable sg4 for both.

Exp C: Post-commitment switching (switch starts at T=2000 after commitment)
  5 seeds. Run task A alone to T=2000, then switch every 500 steps.
  Metric: sg4n_A(t), sg4n_B(t).
  Prediction: task A persists longer due to committed attractor; task B
              learns slower; eventual interference or stable dual encoding.
"""
import numpy as np, json, os, math, random, multiprocessing as mp

# ── Constants ───────────────────────────────────────────────────────────────
W=80; H=40; HALF=40
HS=2; IS=3
WAVE_DUR=15
SUPP_AMP=0.25; EXC_AMP=0.50
ZONE_K=320
MID_DECAY=0.99; FIELD_DECAY=0.9997; BASE_BETA=0.005; SS=10
FA_STD=0.16
VALS_DECAY=0.92; VALS_NAV=0.08; ADJ_SCALE=0.03
BOUND_LO=0.05; BOUND_HI=0.95; FRAG_NOISE=0.01; SEED_BETA=0.25
INST_THRESH=0.45; INST_PROB=0.03; DIFFUSE=0.02; DIFFUSE_EVERY=2

WR_FIXED=4.8; N_ZONES=4; ZW=HALF//N_ZONES  # zw=10

SEEDS = list(range(5))

# Exp A
T_SHIFT_A = 1500       # shift BEFORE commitment (t*~2000)
T_END_A   = 4000
CPS_A = list(range(200, T_END_A+1, 200))   # 20 checkpoints

# Exp B
T_SWITCH_B = 500       # interleave interval
T_END_B    = 4000
CPS_B = list(range(200, T_END_B+1, 200))

# Exp C
T_COMMIT_C = 2000      # let task A commit first
T_SWITCH_C = 500
T_END_C    = 5000
CPS_C = list(range(200, T_END_C+1, 200))

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper40_results.json")


# ── FastVCML ─────────────────────────────────────────────────────────────────
class FastVCML:
    def __init__(self, seed, fa=FA_STD):
        self.rng = np.random.RandomState(seed)
        self.fa = fa
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

    def step(self, wa, wc):
        nb=self._nbmean(); an=np.minimum(1.,self.age/300.)
        x=np.stack([nb,np.minimum(1.,wa),an],1)
        hn,out=self._gru(x); self.hid=hn
        self.vals=np.clip(VALS_DECAY*self.vals+VALS_NAV*nb+ADJ_SCALE*out,0,1)
        dev=hn-self.base_h; self.base_h+=BASE_BETA*dev
        self.streak=np.where(np.sum(dev**2,1)<.0025,self.streak+1,0)
        fa=self.fa
        self.mid=(self.mid+fa*dev)*MID_DECAY
        gate=self.streak>=SS; self.fieldM[gate]+=fa*(self.mid[gate]-self.fieldM[gate])
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
        for ai in ci:
            self.cc[ai]+=1; fm=self.fieldM[ai]; mag=np.sqrt(np.dot(fm,fm))
            nh=((1-SEED_BETA)*prev[ai]+SEED_BETA*fm if mag>1e-6 else prev[ai].copy())
            nh+=rng.randn(HS)*FRAG_NOISE
            self.hid[ai]=nh; self.vals[ai]=.5; self.age[ai]=0
            self.streak[ai]=0; self.mid[ai]=np.zeros(HS)

    def stable_zone(self,k): return np.argsort(self.cc)[:min(k,self.N//5)]


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_sg4_delta(vcml, n_zones, delta=0):
    """Compute sg4n using zone boundaries offset by delta (modular wrap)."""
    si = vcml.stable_zone(ZONE_K)
    xp = vcml.ai_x[si]
    # Apply modular delta shift to zone assignment
    za = np.minimum(n_zones-1, ((xp - HALF + delta) % HALF) // ZW)
    means=[]; wvars=[]
    for z in range(n_zones):
        zs = si[za==z]
        if len(zs):
            fm=vcml.fieldM[zs]; mz=fm.mean(0); means.append(mz)
            wvars.append(float(np.mean(np.sum((fm-mz)**2,1))))
        else:
            means.append(np.zeros(HS)); wvars.append(float('nan'))
    dists=[np.linalg.norm(means[a]-means[b])
           for a in range(n_zones) for b in range(a+1,n_zones)]
    sg4=float(np.mean(dists)) if dists else 0.
    vv=[v for v in wvars if not math.isnan(v)]
    sw=float(np.sqrt(np.mean(vv))) if vv else float('nan')
    sg4n=float(sg4/sw) if sw>1e-8 else float('nan')
    return sg4, sg4n


# ── ShiftableEnv: waves follow the current delta ─────────────────────────────
class ShiftableEnv:
    """
    Structured wave environment. The zone-to-position mapping is
    controlled by self.delta (can be changed mid-run).

    With delta=0: zone 0 -> positions [HALF+0, HALF+ZW)
    With delta=d: zone 0 -> positions whose SHIFTED coordinate falls in [0, ZW)
                  i.e. cx such that (cx - HALF + d) % HALF < ZW
    """
    def __init__(self, vcml, WR, n_zones, delta=0, rng_seed=0):
        self.vcml=vcml; self.WR=WR; self.n_zones=n_zones
        self.delta=delta        # changeable mid-run
        self.zw=HALF//n_zones
        self.rng=random.Random(rng_seed)
        self.waves=[]
        N=vcml.N
        self._wa=np.empty(N); self._wc=np.empty(N,int)
        self.ax=vcml.ai_x; self.ay=vcml.ai_y

    def _cx_for_zone(self, cls):
        """Sample a launch x within the shifted zone."""
        # Zone cls in shifted space covers [(cls*zw - delta) % HALF, ...] in original space
        # Easiest: pick a position in shifted space then map back
        shifted_lo = cls * self.zw
        shifted_hi = (cls + 1) * self.zw - 1
        shifted_x = self.rng.randint(shifted_lo, shifted_hi)
        cx = HALF + (shifted_x - self.delta) % HALF
        return cx

    def _launch(self):
        cls = self.rng.randint(0, self.n_zones - 1)
        cx = self._cx_for_zone(cls)
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
            act=np.maximum(0., 1.-dist*0.4); act[dist>2]=0.
            better=act>wa; wa[better]=act[better]; wc[better]=cls
            wave[2]-=1
            if wave[2]>0: surv.append(wave)
        self.waves=surv
        # Apply supp/exc
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


# ── Run functions ─────────────────────────────────────────────────────────────
def run_A(seed, delta_after):
    """Exp A: zone boundary shift at T_SHIFT_A."""
    vcml = FastVCML(seed=seed*3)
    env  = ShiftableEnv(vcml, WR_FIXED, N_ZONES, delta=0, rng_seed=seed*100)
    result = {'exp':'A','seed':seed,'delta_after':delta_after,
              'sg4ns_new':[], 'sg4ns_old':[], 'ts':CPS_A}
    delta_cur = 0
    for t in range(T_END_A):
        if t == T_SHIFT_A:
            delta_cur = delta_after
            env.delta = delta_after
        wa,wc = env.step(); vcml.step(wa, wc)
        if (t+1) in CPS_A:
            # sg4 with current (new) zone boundaries
            _, sg4n_new = compute_sg4_delta(vcml, N_ZONES, delta=delta_cur)
            # sg4 with original (old) zone boundaries (always delta=0)
            _, sg4n_old = compute_sg4_delta(vcml, N_ZONES, delta=0)
            result['sg4ns_new'].append(sg4n_new)
            result['sg4ns_old'].append(sg4n_old)
    return result


def run_B(seed):
    """Exp B: interleaved task switching from t=0 (no prior commitment)."""
    vcml = FastVCML(seed=seed*5+1)
    env  = ShiftableEnv(vcml, WR_FIXED, N_ZONES, delta=0, rng_seed=seed*77)
    result = {'exp':'B','seed':seed,
              'sg4ns_A':[], 'sg4ns_B':[], 'ts':CPS_B, 'active_task':[]}
    DELTA_A = 0; DELTA_B = 5
    for t in range(T_END_B):
        # switch every T_SWITCH_B steps
        period = (t // T_SWITCH_B) % 2
        cur_delta = DELTA_A if period==0 else DELTA_B
        env.delta = cur_delta
        wa,wc = env.step(); vcml.step(wa, wc)
        if (t+1) in CPS_B:
            _, sg4n_A = compute_sg4_delta(vcml, N_ZONES, delta=DELTA_A)
            _, sg4n_B = compute_sg4_delta(vcml, N_ZONES, delta=DELTA_B)
            result['sg4ns_A'].append(sg4n_A)
            result['sg4ns_B'].append(sg4n_B)
            result['active_task'].append(int(period))
    return result


def run_C(seed):
    """Exp C: task A committed for T_COMMIT_C steps, then interleaved switching."""
    vcml = FastVCML(seed=seed*7+3)
    env  = ShiftableEnv(vcml, WR_FIXED, N_ZONES, delta=0, rng_seed=seed*33+9)
    result = {'exp':'C','seed':seed,
              'sg4ns_A':[], 'sg4ns_B':[], 'ts':CPS_C, 'active_task':[]}
    DELTA_A = 0; DELTA_B = 5
    for t in range(T_END_C):
        if t < T_COMMIT_C:
            cur_delta = DELTA_A   # task A only during commitment phase
        else:
            # interleave from T_COMMIT_C onward
            period = ((t - T_COMMIT_C) // T_SWITCH_C) % 2
            cur_delta = DELTA_A if period==0 else DELTA_B
        env.delta = cur_delta
        wa,wc = env.step(); vcml.step(wa, wc)
        if (t+1) in CPS_C:
            _, sg4n_A = compute_sg4_delta(vcml, N_ZONES, delta=DELTA_A)
            _, sg4n_B = compute_sg4_delta(vcml, N_ZONES, delta=DELTA_B)
            result['sg4ns_A'].append(sg4n_A)
            result['sg4ns_B'].append(sg4n_B)
            result['active_task'].append(0 if t < T_COMMIT_C else
                                         int(((t-T_COMMIT_C)//T_SWITCH_C)%2))
    return result


# ── Worker + main ─────────────────────────────────────────────────────────────
def _worker(args):
    tag = args[0]
    if tag == 'A': return run_A(*args[1:])
    if tag == 'B': return run_B(*args[1:])
    if tag == 'C': return run_C(*args[1:])


def make_key(r):
    e = r['exp']
    if e == 'A': return f"A,{r['seed']},{r['delta_after']}"
    if e == 'B': return f"B,{r['seed']}"
    if e == 'C': return f"C,{r['seed']}"
    return str(r)


if __name__ == '__main__':
    mp.freeze_support()
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f: all_results = json.load(f)
    else:
        all_results = []

    done = set(make_key(r) for r in all_results)

    all_conds = []
    # Exp A: 3 conditions x 5 seeds = 15 runs
    for seed in SEEDS:
        for delta_after in [0, 5, 10]:     # C_ref, C_half, C_full
            k = f"A,{seed},{delta_after}"
            if k not in done:
                all_conds.append(('A', seed, delta_after))
    # Exp B: 5 seeds
    for seed in SEEDS:
        if f"B,{seed}" not in done:
            all_conds.append(('B', seed))
    # Exp C: 5 seeds
    for seed in SEEDS:
        if f"C,{seed}" not in done:
            all_conds.append(('C', seed))

    total = 15 + 5 + 5
    print(f"Total: {total} runs, todo: {len(all_conds)}, done: {len(done)}")

    if all_conds:
        n_proc = min(8, len(all_conds))
        print(f"Running {len(all_conds)} jobs on {n_proc} cores ...")
        with mp.Pool(processes=n_proc) as pool:
            new_results = pool.map(_worker, all_conds)
        all_results.extend(new_results)
        with open(RESULTS_FILE, 'w') as f: json.dump(all_results, f)
        print("Saved.")

    # ── Analysis ──────────────────────────────────────────────────────────────
    def mn(lst):
        v = [x for x in lst if x is not None and not math.isnan(x)]
        return float(np.mean(v)) if v else float('nan')

    from collections import defaultdict

    # ── Exp A ─────────────────────────────────────────────────────────────────
    print("\n=== Exp A: Zone boundary shift ===")
    print(f"  T_SHIFT={T_SHIFT_A}, ZW={ZW}")
    for delta in [0, 5, 10]:
        runs = [r for r in all_results if r['exp']=='A' and r['delta_after']==delta]
        if not runs: continue
        label = {0:'C_ref  (no shift)', 5:'C_half (delta=5,  50% cells move)',
                 10:'C_full (delta=10, 100% cells move)'}[delta]
        n  = len(runs[0]['ts'])
        ts = runs[0]['ts']
        mean_new = [mn([r['sg4ns_new'][i] for r in runs]) for i in range(n)]
        mean_old = [mn([r['sg4ns_old'][i] for r in runs]) for i in range(n)]
        # Find post-shift peak of sg4n_new
        post = [(ts[i], mean_new[i]) for i in range(n) if ts[i] > T_SHIFT_A]
        peak_t, peak_v = max(post, key=lambda x: x[1]) if post else (None, float('nan'))
        # sg4n_old at T_END
        old_end = mean_old[-1]
        print(f"\n  {label}  ({len(runs)} seeds)")
        print(f"  sg4n_new at T_SHIFT (before): {mn([r['sg4ns_new'][CPS_A.index(T_SHIFT_A)] for r in runs if T_SHIFT_A in CPS_A]):.4f}")
        print(f"  sg4n_new post-shift peak:      {peak_v:.4f} at t={peak_t}")
        print(f"  sg4n_old at T_END:             {old_end:.4f}")

    # ── Exp B ─────────────────────────────────────────────────────────────────
    print("\n=== Exp B: Interleaved switching (no prior commitment) ===")
    runs_B = [r for r in all_results if r['exp']=='B']
    if runs_B:
        n = len(runs_B[0]['ts']); ts = runs_B[0]['ts']
        mean_A = [mn([r['sg4ns_A'][i] for r in runs_B]) for i in range(n)]
        mean_B = [mn([r['sg4ns_B'][i] for r in runs_B]) for i in range(n)]
        print(f"  mean sg4n_A (final 3 cps): {np.mean(mean_A[-3:]):.4f}")
        print(f"  mean sg4n_B (final 3 cps): {np.mean(mean_B[-3:]):.4f}")
        print(f"  ratio B/A: {np.mean(mean_B[-3:])/np.mean(mean_A[-3:]):.3f}")

    # ── Exp C ─────────────────────────────────────────────────────────────────
    print("\n=== Exp C: Post-commitment interleaved switching ===")
    runs_C = [r for r in all_results if r['exp']=='C']
    if runs_C:
        n = len(runs_C[0]['ts']); ts = runs_C[0]['ts']
        mean_A = [mn([r['sg4ns_A'][i] for r in runs_C]) for i in range(n)]
        mean_B = [mn([r['sg4ns_B'][i] for r in runs_C]) for i in range(n)]
        # Find sg4n_A just before switching and at end
        pre  = [(ts[i], mean_A[i]) for i in range(n) if ts[i] <= T_COMMIT_C]
        post = [(ts[i], mean_A[i]) for i in range(n) if ts[i] >  T_COMMIT_C]
        print(f"  sg4n_A pre-switch peak:  {max(v for _,v in pre):.4f}")
        print(f"  sg4n_A final:            {mean_A[-1]:.4f}")
        print(f"  sg4n_B final:            {mean_B[-1]:.4f}")
        print(f"  B/A ratio at end:        {mean_B[-1]/mean_A[-1]:.3f}" if mean_A[-1]>1e-6 else "  (A near zero)")
