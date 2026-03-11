"""
paper41_experiments.py -- Interleaving Period Threshold

Paper 38: metastability half-life ~800 steps.
Paper 40: T_switch=500 erodes both tasks to equal steady-state.

Prediction: T_crit ~ 800 steps (= metastability half-life).
  - T_switch << T_crit : neither task commits; both near zero
  - T_switch ~  T_crit : transition; first-mover advantage starts to emerge
  - T_switch >> T_crit : first task commits fully before switch; A >> B

Experiment: sweep T_switch in {50, 100, 200, 500, 800, 1000, 1500, 2000, 3000}
  Two tasks: A (delta=0) and B (delta=5), alternating from t=0.
  Measure sg4n_A and sg4n_B at T=6000 (well past transient for all T_switch).
  Also measure: sg4n_A/sg4n_B ratio (first-mover advantage as function of T_switch).

Reference condition:
  C_ref: single task A only (no switching), T=6000. Gives the no-switch ceiling.

5 seeds per condition. Total = 9 sweep values * 5 + 5 ref = 50 runs.
"""
import numpy as np, json, os, math, random, multiprocessing as mp

# ── Constants ──────────────────────────────────────────────────────────────
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
T_END  = 6000
# Checkpoint every 300 steps (20 checkpoints total)
CPS = list(range(300, T_END+1, 300))

DELTA_A = 0
DELTA_B = 5

T_SWITCH_SWEEP = [50, 100, 200, 500, 800, 1000, 1500, 2000, 3000]

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper41_results.json")


# ── FastVCML (copied from paper40) ─────────────────────────────────────────
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


def compute_sg4_delta(vcml, n_zones, delta=0):
    si = vcml.stable_zone(ZONE_K)
    xp = vcml.ai_x[si]
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


class ShiftableEnv:
    def __init__(self, vcml, WR, n_zones, delta=0, rng_seed=0):
        self.vcml=vcml; self.WR=WR; self.n_zones=n_zones
        self.delta=delta
        self.zw=HALF//n_zones
        self.rng=random.Random(rng_seed)
        self.waves=[]
        N=vcml.N
        self._wa=np.empty(N); self._wc=np.empty(N,int)
        self.ax=vcml.ai_x; self.ay=vcml.ai_y

    def _cx_for_zone(self, cls):
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


# ── Run functions ────────────────────────────────────────────────────────────
def run_sweep(seed, t_switch):
    """Interleaved switching: alternate A/B every t_switch steps from t=0."""
    vcml = FastVCML(seed=seed*3+1)
    env  = ShiftableEnv(vcml, WR_FIXED, N_ZONES, delta=DELTA_A, rng_seed=seed*100+7)
    result = {'exp':'sweep', 'seed':seed, 't_switch':t_switch,
              'sg4ns_A':[], 'sg4ns_B':[], 'ts':CPS}
    for t in range(T_END):
        period = (t // t_switch) % 2
        env.delta = DELTA_A if period == 0 else DELTA_B
        wa, wc = env.step(); vcml.step(wa, wc)
        if (t+1) in CPS:
            _, sg4n_A = compute_sg4_delta(vcml, N_ZONES, delta=DELTA_A)
            _, sg4n_B = compute_sg4_delta(vcml, N_ZONES, delta=DELTA_B)
            result['sg4ns_A'].append(sg4n_A)
            result['sg4ns_B'].append(sg4n_B)
    return result


def run_ref(seed):
    """Single task A only (no switching) for T_END steps."""
    vcml = FastVCML(seed=seed*7+3)
    env  = ShiftableEnv(vcml, WR_FIXED, N_ZONES, delta=DELTA_A, rng_seed=seed*55+2)
    result = {'exp':'ref', 'seed':seed, 'sg4ns_A':[], 'ts':CPS}
    for t in range(T_END):
        wa, wc = env.step(); vcml.step(wa, wc)
        if (t+1) in CPS:
            _, sg4n_A = compute_sg4_delta(vcml, N_ZONES, delta=DELTA_A)
            result['sg4ns_A'].append(sg4n_A)
    return result


# ── Worker + main ─────────────────────────────────────────────────────────────
def _worker(args):
    tag = args[0]
    if tag == 'sweep': return run_sweep(*args[1:])
    if tag == 'ref':   return run_ref(*args[1:])


def make_key(r):
    if r['exp'] == 'sweep': return f"sweep,{r['seed']},{r['t_switch']}"
    if r['exp'] == 'ref':   return f"ref,{r['seed']}"
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

    for seed in SEEDS:
        for ts in T_SWITCH_SWEEP:
            if f"sweep,{seed},{ts}" not in done:
                all_conds.append(('sweep', seed, ts))
        if f"ref,{seed}" not in done:
            all_conds.append(('ref', seed))

    total = len(T_SWITCH_SWEEP)*len(SEEDS) + len(SEEDS)
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
        return (float(np.mean(v)), float(np.std(v)/np.sqrt(len(v)))) if v else (float('nan'), float('nan'))

    # Reference ceiling
    refs = [r for r in all_results if r['exp']=='ref']
    if refs:
        final_ref = mn([r['sg4ns_A'][-1] for r in refs])
        print(f"\nC_ref (single task, T={T_END}): sg4n_A = {final_ref[0]:.4f} +- {final_ref[1]:.4f}")

    # Sweep results
    print(f"\n=== Interleaving sweep at T={T_END} ===")
    print(f"  T_half ~ 800 steps  |  T_commit ~ 2000 steps")
    print(f"  {'T_switch':>8}  {'sg4n_A':>8}  {'sg4n_B':>8}  {'ratio A/B':>9}  {'A/ref':>6}")
    ref_val = final_ref[0] if refs else float('nan')
    for ts in T_SWITCH_SWEEP:
        runs = [r for r in all_results if r['exp']=='sweep' and r['t_switch']==ts]
        if not runs: continue
        n = len(runs[0]['sg4ns_A'])
        # Use mean of final 3 checkpoints for stability
        vA = mn([np.mean(r['sg4ns_A'][-3:]) for r in runs])
        vB = mn([np.mean(r['sg4ns_B'][-3:]) for r in runs])
        ratio = vA[0]/vB[0] if vB[0]>1e-6 else float('nan')
        frac  = vA[0]/ref_val if ref_val>1e-6 else float('nan')
        print(f"  {ts:>8}  {vA[0]:>8.4f}  {vB[0]:>8.4f}  {ratio:>9.3f}  {frac:>6.3f}")
