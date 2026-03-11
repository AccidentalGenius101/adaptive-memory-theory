"""
paper42_experiments.py -- Three Anomalies Closed

Paper 42 closes three loose ends remaining from single-region VCML theory:

Anomaly 1 (C_loc ratio > 1, V89):
  Can VCML zone structure persist or grow without active wave input?
  Exp A: wave ablation.
    C_ref:      waves throughout (WR=4.8), T=4000
    C_wavestop: normal waves until T=2000, then WR=0 (waves cleared)
    C_nowaves:  WR=0 from t=0 (no waves ever)
  Measure sg4n and cumulative collapse count at each checkpoint.

Anomaly 2 (C_half late surge, Paper 40):
  At T=4000, C_half (delta=5 boundary shift at T=1500) reaches sg4n=0.80,
  far exceeding C_ref peak (0.38). Is the surge sustained? What drives it?
  Exp B: extend to T=6000, track collapse rate per checkpoint.
    C_ref:       delta=0 throughout
    C_half_shift: delta=0 until T_SHIFT=1500, then delta=5
  Prediction: C_half maintains elevated collapse rate after T_shift,
  sustaining copy-forward refresh (same mechanism as Paper 41).

Anomaly 3 (T_sw=50 vs 100 reversal, Paper 41):
  At T_sw=50: sg4n_B > sg4n_A (Table 1). At T_sw=100: sg4n_A > sg4n_B.
  Prediction: this is a CHECKPOINT PHASE ALIASING ARTIFACT.
  300-step CPS always hits task B epochs for T_sw=50 (300/50=6, even,
  so every checkpoint lands in the same task B parity). At T_sw=100
  (300/100=3, odd), checkpoints alternate A/B.
  Exp C: demonstrate with epoch-end measurements (unbiased).
    T_sw in {50, 100}, 10 seeds each.
    Two measurement modes:
      (a) CPS every 300 steps (reproduces Paper 41 bias)
      (b) Epoch-end: measure sg4n_A at end of each A epoch, sg4n_B at end of B epoch

Total: Exp A = 15, Exp B = 10, Exp C = 20. Total = 45 runs.
"""
import numpy as np, json, os, math, random, multiprocessing as mp

# ── Constants (identical to paper41) ──────────────────────────────────────────
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

SEEDS_A = list(range(5))
SEEDS_B = list(range(5))
SEEDS_C = list(range(10))

T_END_A = 4000;  CPS_A = list(range(200, T_END_A+1, 200));  CPS_A_SET = set(CPS_A)
T_END_B = 6000;  CPS_B = list(range(300, T_END_B+1, 300));  CPS_B_SET = set(CPS_B)
T_END_C = 6000;  CPS_C = list(range(300, T_END_C+1, 300));  CPS_C_SET = set(CPS_C)

T_COMMIT = 2000    # wave stop time for C_wavestop (Exp A)
T_SHIFT_B = 1500   # zone boundary shift time (Exp B)
T_TRANSIENT_C = 1500  # skip epoch-end measurements before this

DELTA_A = 0; DELTA_B = 5
TSW_C = [50, 100]

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper42_results.json")


# ── FastVCML ──────────────────────────────────────────────────────────────────
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


# ── Experiment A: Wave ablation ────────────────────────────────────────────────
def run_expA(seed, cond):
    """Wave ablation: C_ref, C_wavestop (stop at T=2000), C_nowaves (never)."""
    vcml = FastVCML(seed=seed*5+2)
    env  = ShiftableEnv(vcml, WR_FIXED, N_ZONES, delta=0, rng_seed=seed*200+13)
    # Set WR=0 immediately for no-wave condition
    if cond == 'nowaves':
        env.WR = 0
    result = {'exp':'A', 'cond':cond, 'seed':seed,
              'sg4ns':[], 'cc_totals':[], 'ts':CPS_A}
    for t in range(T_END_A):
        if cond == 'wavestop' and t == T_COMMIT:
            env.WR = 0; env.waves = []
        wa, wc = env.step(); vcml.step(wa, wc)
        if (t+1) in CPS_A_SET:
            _, sg4n = compute_sg4_delta(vcml, N_ZONES, delta=0)
            result['sg4ns'].append(sg4n)
            result['cc_totals'].append(int(np.sum(vcml.cc)))
    return result


# ── Experiment B: C_half surge anatomy ────────────────────────────────────────
def run_expB(seed, cond):
    """C_half surge: run to T=6000 tracking collapse rate."""
    vcml = FastVCML(seed=seed*7+3)
    env  = ShiftableEnv(vcml, WR_FIXED, N_ZONES, delta=0, rng_seed=seed*300+17)
    result = {'exp':'B', 'cond':cond, 'seed':seed,
              'sg4ns_new':[], 'sg4ns_old':[], 'cc_totals':[], 'ts':CPS_B}
    for t in range(T_END_B):
        if cond == 'half_shift' and t == T_SHIFT_B:
            env.delta = DELTA_B
        wa, wc = env.step(); vcml.step(wa, wc)
        if (t+1) in CPS_B_SET:
            cur_delta = env.delta
            _, sg4n_new = compute_sg4_delta(vcml, N_ZONES, delta=cur_delta)
            _, sg4n_old = compute_sg4_delta(vcml, N_ZONES, delta=0)
            result['sg4ns_new'].append(sg4n_new)
            result['sg4ns_old'].append(sg4n_old)
            result['cc_totals'].append(int(np.sum(vcml.cc)))
    return result


# ── Experiment C: Phase aliasing ──────────────────────────────────────────────
def run_expC(seed, t_switch):
    """Phase aliasing test: both biased CPS and corrected epoch-end measurements."""
    vcml = FastVCML(seed=seed*3+1)
    env  = ShiftableEnv(vcml, WR_FIXED, N_ZONES, delta=DELTA_A, rng_seed=seed*100+7)
    result = {'exp':'C', 'seed':seed, 't_switch':t_switch,
              # Checkpoint-based (reproduces Paper 41 bias)
              'sg4ns_A_cp':[], 'sg4ns_B_cp':[], 'ts_cp':CPS_C,
              # Epoch-end (unbiased: measured at transition from each epoch)
              'epoch_end_A':[], 'epoch_end_B':[],
              'epoch_ts_A':[], 'epoch_ts_B':[]}
    for t in range(T_END_C):
        period = (t // t_switch) % 2
        env.delta = DELTA_A if period == 0 else DELTA_B
        wa, wc = env.step(); vcml.step(wa, wc)
        # Standard checkpoints (every 300 steps)
        if (t+1) in CPS_C_SET:
            _, sg4n_A = compute_sg4_delta(vcml, N_ZONES, delta=DELTA_A)
            _, sg4n_B = compute_sg4_delta(vcml, N_ZONES, delta=DELTA_B)
            result['sg4ns_A_cp'].append(float(sg4n_A))
            result['sg4ns_B_cp'].append(float(sg4n_B))
        # Epoch-end: measure at last step before each task switch (after transient)
        if t >= T_TRANSIENT_C:
            next_period = ((t+1) // t_switch) % 2
            if next_period != period:  # last step of current epoch
                _, sg4n_A = compute_sg4_delta(vcml, N_ZONES, delta=DELTA_A)
                _, sg4n_B = compute_sg4_delta(vcml, N_ZONES, delta=DELTA_B)
                if period == 0:  # end of task A epoch -> record sg4n_A
                    result['epoch_end_A'].append(float(sg4n_A))
                    result['epoch_ts_A'].append(t+1)
                else:            # end of task B epoch -> record sg4n_B
                    result['epoch_end_B'].append(float(sg4n_B))
                    result['epoch_ts_B'].append(t+1)
    return result


# ── Worker + main ──────────────────────────────────────────────────────────────
def _worker(args):
    tag = args[0]
    if tag == 'A': return run_expA(*args[1:])
    if tag == 'B': return run_expB(*args[1:])
    if tag == 'C': return run_expC(*args[1:])


def make_key(r):
    if r['exp'] == 'A': return f"A,{r['seed']},{r['cond']}"
    if r['exp'] == 'B': return f"B,{r['seed']},{r['cond']}"
    if r['exp'] == 'C': return f"C,{r['seed']},{r['t_switch']}"
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

    # Exp A: 3 conditions x 5 seeds = 15
    for seed in SEEDS_A:
        for cond in ['ref', 'wavestop', 'nowaves']:
            if f"A,{seed},{cond}" not in done:
                all_conds.append(('A', seed, cond))

    # Exp B: 2 conditions x 5 seeds = 10
    for seed in SEEDS_B:
        for cond in ['ref', 'half_shift']:
            if f"B,{seed},{cond}" not in done:
                all_conds.append(('B', seed, cond))

    # Exp C: 2 T_sw x 10 seeds = 20
    for seed in SEEDS_C:
        for tsw in TSW_C:
            if f"C,{seed},{tsw}" not in done:
                all_conds.append(('C', seed, tsw))

    total = 3*len(SEEDS_A) + 2*len(SEEDS_B) + len(TSW_C)*len(SEEDS_C)
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

    # ── Exp A ──────────────────────────────────────────────────────────────────
    print("\n=== Exp A: Wave Ablation ===")
    print(f"{'Cond':>12}  {'sg4n@T=2000':>12}  {'sg4n@T=4000':>12}  "
          f"{'ratio 4k/2k':>11}  {'cc/step':>8}")
    for cond in ['ref', 'wavestop', 'nowaves']:
        runs = [r for r in all_results if r['exp']=='A' and r['cond']==cond]
        if not runs: continue
        ts = CPS_A
        def sg4_at(t_target):
            if t_target in ts:
                idx = ts.index(t_target)
                return mn([r['sg4ns'][idx] for r in runs])
            return (float('nan'), float('nan'))
        v2k = sg4_at(2000); v4k = sg4_at(4000)
        ratio = v4k[0]/v2k[0] if v2k[0]>1e-6 else float('nan')
        cc_rate = mn([r['cc_totals'][-1] / T_END_A for r in runs])
        print(f"  {cond:>10}  {v2k[0]:>12.4f}  {v4k[0]:>12.4f}  {ratio:>11.3f}  {cc_rate[0]:>8.5f}")

    print("\n  Field-decay-only prediction: sg4n(4000)/sg4n(2000) = "
          f"{FIELD_DECAY**2000:.4f}  (FIELD_DECAY^2000)")

    # ── Exp B ──────────────────────────────────────────────────────────────────
    print("\n=== Exp B: C_half Surge Anatomy (T=6000) ===")
    print(f"{'Cond':>14}  {'sg4n@T=1500':>12}  {'sg4n@T=4000':>12}  "
          f"{'sg4n@T=6000':>12}  {'cc/step_early':>14}  {'cc/step_late':>13}")
    for cond in ['ref', 'half_shift']:
        runs = [r for r in all_results if r['exp']=='B' and r['cond']==cond]
        if not runs: continue
        ts = CPS_B
        def sg4n_at_b(t_target):
            if t_target in ts:
                idx = ts.index(t_target)
                return mn([r['sg4ns_new'][idx] for r in runs])
            return (float('nan'), float('nan'))
        v15 = sg4n_at_b(1500); v40 = sg4n_at_b(4000); v60 = sg4n_at_b(6000)
        # Collapse rate: early (T=0-2000) vs late (T=4000-6000)
        # cc_totals is cumulative; index for T=2000 in 300-step CPS
        idx_2000 = ts.index(2100) if 2100 in ts else ts.index(1800) if 1800 in ts else 6  # ~T=2100
        idx_4000 = ts.index(4200) if 4200 in ts else ts.index(3900) if 3900 in ts else 13
        # Use simpler: first 7 checkpoints (T=300..2100) vs last 7 (T=4200..6000)
        # With 300-step CPS, index 0..6 = T=300..2100, index 13..19 = T=4200..6000
        cc_early = mn([r['cc_totals'][6] / 2100 for r in runs])
        cc_late  = mn([(r['cc_totals'][-1] - r['cc_totals'][13]) / (6000 - 4200) for r in runs])
        print(f"  {cond:>12}  {v15[0]:>12.4f}  {v40[0]:>12.4f}  {v60[0]:>12.4f}  "
              f"{cc_early[0]:>14.5f}  {cc_late[0]:>13.5f}")

    # ── Exp C ──────────────────────────────────────────────────────────────────
    print("\n=== Exp C: Phase Aliasing ===")
    print("\nCheckpoint parity analysis (which task is active at each 300-step CPS):")
    for tsw in [50, 100]:
        # Checkpoint at t_cp is measured at t = t_cp - 1
        phases = ['B' if ((tcp-1)//tsw)%2==1 else 'A' for tcp in CPS_C]
        n_B = sum(1 for p in phases if p=='B')
        n_A = len(phases) - n_B
        print(f"  T_sw={tsw:3d}: {' '.join(phases[:20])}  |  A={n_A}/20, B={n_B}/20")
        if tsw == 50:
            print(f"          Note: 300/50=6 (even) -> all checkpoints in same task parity -> always B")
        else:
            print(f"          Note: 300/100=3 (odd)  -> checkpoints alternate A/B")

    print("\nBiased CPS measurement (last 3 checkpoints, Paper 41 method):")
    for tsw in TSW_C:
        runs = [r for r in all_results if r['exp']=='C' and r['t_switch']==tsw]
        if not runs: continue
        vA = mn([np.mean(r['sg4ns_A_cp'][-3:]) for r in runs])
        vB = mn([np.mean(r['sg4ns_B_cp'][-3:]) for r in runs])
        winner = 'B>A (artifact!)' if vB[0]>vA[0] else 'A>B'
        print(f"  T_sw={tsw:3d}: sg4n_A={vA[0]:.4f}+-{vA[1]:.4f}  "
              f"sg4n_B={vB[0]:.4f}+-{vB[1]:.4f}  -> {winner}")

    print("\nEpoch-end measurement (unbiased: sg4n_A at end of A epochs, sg4n_B at end of B epochs):")
    for tsw in TSW_C:
        runs = [r for r in all_results if r['exp']=='C' and r['t_switch']==tsw]
        if not runs: continue
        # Use last 20 epoch-end measurements (steady state)
        def epoch_tail_mean(lst, k=20):
            tail = lst[-k:] if len(lst) >= k else lst
            return [float(x) for x in tail if not math.isnan(x)]
        vA = mn([np.mean(epoch_tail_mean(r['epoch_end_A'])) for r in runs
                 if r['epoch_end_A']])
        vB = mn([np.mean(epoch_tail_mean(r['epoch_end_B'])) for r in runs
                 if r['epoch_end_B']])
        winner = 'B>A' if vB[0]>vA[0] else 'A>B'
        print(f"  T_sw={tsw:3d}: sg4n_A={vA[0]:.4f}+-{vA[1]:.4f}  "
              f"sg4n_B={vB[0]:.4f}+-{vB[1]:.4f}  -> {winner}")
