"""
paper38_experiments.py -- Continual Learning, Basin Fraction, Long-Run Saturation

Exp A: Continual learning stability -- zone pattern shift at T_SHIFT=2000
  Three conditions run to T_END=4000:
    C_ref   : FA=0.16 throughout, shift at T_SHIFT (supp<->exc inverted in all zones)
    C_cryst : FA=0.16 phase-1, FA=0.01 phase-2, shift at T_SHIFT (low plasticity)
    C_null  : FA=0.16 throughout, NO shift (control)
  Prediction: C_ref sg4 dips then recovers; C_cryst sg4 dips and stays low.

Exp B: Basin fraction -- geographic structure biases the high-structure attractor basin
  30 seeds, T=3000, 2 conditions: geo (structured waves) vs ctrl (random waves)
  Measure: sg4n distribution, P_high(geo) vs P_high(ctrl)
  Prediction: P_high(geo) >> P_high(ctrl) explains relay gain G.

Exp C: Long-run saturation -- does sg4 plateau within T=3000?
  5 seeds, geo env, T=8000, checkpoints every 400 steps
  Also 5 ctrl seeds.
  Prediction: T_sat ~ kappa/(P_c*FA*nu*beta_s^2), expected 3000-5000 steps.
"""
import numpy as np, json, os, math, random, multiprocessing as mp

# ── Constants ───────────────────────────────────────────────────────────────
W=80; H=40; HALF=40
HS=2; IS=3
WAVE_DUR=15
SUPP_AMP=0.25; EXC_AMP=0.50
ZONE_K=320
MID_DECAY=0.99; FIELD_DECAY=0.9997; BASE_BETA=0.005; SS=10
FA_STD=0.16; FA_CRYST=0.01
VALS_DECAY=0.92; VALS_NAV=0.08; ADJ_SCALE=0.03
BOUND_LO=0.05; BOUND_HI=0.95; FRAG_NOISE=0.01; SEED_BETA=0.25
INST_THRESH=0.45; INST_PROB=0.03; DIFFUSE=0.02; DIFFUSE_EVERY=2

WR_FIXED=4.8; N_FIXED=4

# Exp A
SEEDS_A = list(range(5))
T_SHIFT = 2000; T_END_A = 4000
CPS_A = list(range(200, T_END_A+1, 200))  # 20 checkpoints

# Exp B
N_SEEDS_B = 30
STEPS_B = 3000

# Exp C
SEEDS_C = list(range(5))
T_END_C = 8000
CPS_C = list(range(400, T_END_C+1, 400))  # 20 checkpoints

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper38_results.json")


# ── FastVCML (instance-level FA for mid-run changes) ────────────────────────
class FastVCML:
    def __init__(self, seed, fa=FA_STD):
        self.rng = np.random.RandomState(seed)
        self.fa = fa           # instance variable -- can change at T_SHIFT
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


def compute_sg4(vcml, n_zones):
    zw=HALF//n_zones; si=vcml.stable_zone(ZONE_K)
    xp=vcml.ai_x[si]; za=np.minimum(n_zones-1,(xp-HALF)//zw)
    means=[]; wvars=[]
    for z in range(n_zones):
        zs=si[za==z]
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


def compute_zone_polarity(vcml, n_zones):
    """Measure zone-mean fieldM. Returns list of mean vectors, one per zone."""
    zw=HALF//n_zones; si=vcml.stable_zone(ZONE_K)
    xp=vcml.ai_x[si]; za=np.minimum(n_zones-1,(xp-HALF)//zw)
    means=[]
    for z in range(n_zones):
        zs=si[za==z]
        mz=vcml.fieldM[zs].mean(0).tolist() if len(zs) else [0.]*HS
        means.append(mz)
    return means


# ── StructuredEnv: single-lattice wave environment ───────────────────────────
class StructuredEnv:
    """
    Single-lattice wave environment.
    coupling='geo': waves always land in correct zone position (consistent cls).
    coupling='ctrl': waves land at random positions (random cls).
    cls_offset: shift applied before supp/exc assignment.
                At T_SHIFT, set to 1 to invert supp/exc in all zones.
    """
    def __init__(self, vcml, WR, n_zones, coupling='geo', rng_seed=0):
        self.vcml=vcml; self.WR=WR; self.n_zones=n_zones
        self.coupling=coupling
        self.cls_offset=0       # 0 = original pattern; 1 = inverted
        self.zw=HALF//n_zones
        self.rng=random.Random(rng_seed)
        N=vcml.N; self.ax=vcml.ai_x; self.ay=vcml.ai_y
        self.waves=[]
        self._wa=np.empty(N); self._wc=np.empty(N,int)

    def _launch(self):
        rng=self.rng
        cls=rng.randint(0, self.n_zones-1)
        if self.coupling=='geo':
            zs=HALF+cls*self.zw; ze=HALF+(cls+1)*self.zw-1
            cx=rng.randint(zs,ze)
        else:
            cx=rng.randint(HALF, HALF+HALF-1)
            cls=rng.randint(0, self.n_zones-1)
        cy=rng.randint(0, H-1)
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
        # Apply supp/exc with cls_offset (shifts zone-type assignment)
        eff=(wc+self.cls_offset)%self.n_zones
        for even in [True, False]:
            amp=SUPP_AMP if even else EXC_AMP
            par=(eff%2==0) if even else (eff%2==1)
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
def run_A(seed, fa_phase2, do_shift):
    """Exp A: continual learning stability."""
    vcml=FastVCML(seed=seed*3, fa=FA_STD)
    env=StructuredEnv(vcml, WR_FIXED, N_FIXED, coupling='geo', rng_seed=seed*100)
    result={'exp':'A','seed':seed,'fa_phase2':fa_phase2,'do_shift':do_shift,
            'sg4s':[],'sg4ns':[],'polarity':[],'ts':[]}
    for t in range(T_END_A):
        if t==T_SHIFT:
            vcml.fa=fa_phase2
            if do_shift: env.cls_offset=1
        wa,wc=env.step(); vcml.step(wa,wc)
        if (t+1) in CPS_A:
            sg4,sg4n=compute_sg4(vcml,N_FIXED)
            pol=compute_zone_polarity(vcml,N_FIXED)
            result['sg4s'].append(sg4); result['sg4ns'].append(sg4n)
            result['polarity'].append(pol); result['ts'].append(t+1)
    return result


def run_B(seed, coupling):
    """Exp B: basin fraction — large seed study."""
    vcml=FastVCML(seed=seed*7+1, fa=FA_STD)
    env=StructuredEnv(vcml, WR_FIXED, N_FIXED, coupling=coupling, rng_seed=seed*50+3)
    for t in range(STEPS_B):
        wa,wc=env.step(); vcml.step(wa,wc)
    sg4,sg4n=compute_sg4(vcml,N_FIXED)
    return {'exp':'B','seed':seed,'coupling':coupling,'sg4':sg4,'sg4n':sg4n}


def run_C(seed):
    """Exp C: long-run saturation."""
    vcml=FastVCML(seed=seed*11+5, fa=FA_STD)
    env=StructuredEnv(vcml, WR_FIXED, N_FIXED, coupling='geo', rng_seed=seed*200+7)
    result={'exp':'C','seed':seed,'sg4s':[],'sg4ns':[],'ts':[]}
    for t in range(T_END_C):
        wa,wc=env.step(); vcml.step(wa,wc)
        if (t+1) in CPS_C:
            sg4,sg4n=compute_sg4(vcml,N_FIXED)
            result['sg4s'].append(sg4); result['sg4ns'].append(sg4n); result['ts'].append(t+1)
    return result


# ── Worker + main ────────────────────────────────────────────────────────────
def _worker(args):
    tag=args[0]
    if tag=='A': return run_A(*args[1:])
    if tag=='B': return run_B(*args[1:])
    if tag=='C': return run_C(*args[1:])


def make_key(r):
    e=r['exp']
    if e=='A': return f"A,{r['seed']},{r['fa_phase2']},{r['do_shift']}"
    if e=='B': return f"B,{r['seed']},{r['coupling']}"
    if e=='C': return f"C,{r['seed']}"
    return str(r)


if __name__=='__main__':
    mp.freeze_support()

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f: all_results=json.load(f)
    else: all_results=[]

    done=set(make_key(r) for r in all_results)

    all_conds=[]
    # Exp A: 3 conditions x 5 seeds
    for seed in SEEDS_A:
        for fa2,do_shift in [(FA_STD,True),(FA_CRYST,True),(FA_STD,False)]:
            if f"A,{seed},{fa2},{do_shift}" not in done:
                all_conds.append(('A', seed, fa2, do_shift))
    # Exp B: 30 seeds x 2 conditions
    for seed in range(N_SEEDS_B):
        for coupling in ['geo','ctrl']:
            if f"B,{seed},{coupling}" not in done:
                all_conds.append(('B', seed, coupling))
    # Exp C: 5 seeds
    for seed in SEEDS_C:
        if f"C,{seed}" not in done:
            all_conds.append(('C', seed))

    total=15+60+5
    print(f"Total: {total}, todo: {len(all_conds)}, done: {len(done)}")
    if all_conds:
        n_proc=min(8, len(all_conds))
        print(f"Running {len(all_conds)} jobs on {n_proc} cores...")
        with mp.Pool(processes=n_proc) as pool:
            new_results=pool.map(_worker, all_conds)
        all_results.extend(new_results)
        with open(RESULTS_FILE,'w') as f: json.dump(all_results,f)
        print("Saved.")

    # ── Analysis ────────────────────────────────────────────────────────────
    from collections import defaultdict
    def mn(lst):
        v=[x for x in lst if x is not None and not math.isnan(x)]
        return float(np.mean(v)) if v else float('nan')

    # Exp A analysis
    A_data = defaultdict(list)   # key (fa2, do_shift) -> list of runs
    for r in all_results:
        if r['exp']=='A':
            A_data[(r['fa_phase2'], r['do_shift'])].append(r)

    print("\n=== Exp A: sg4n(t) per condition ===")
    cond_labels={
        (FA_STD, True):   'C_ref   (FA=0.16, shift)',
        (FA_CRYST, True): 'C_cryst (FA=0.01, shift)',
        (FA_STD, False):  'C_null  (FA=0.16, no shift)',
    }
    for key, label in cond_labels.items():
        runs=A_data[key]
        if not runs: continue
        n=len(runs[0]['ts'])
        ts=runs[0]['ts']
        mean_sg4n=[mn([r['sg4ns'][i] for r in runs]) for i in range(n)]
        print(f"\n  {label}  ({len(runs)} seeds)")
        print(f"  {'t':>6} {'sg4n':>8}  (shift at t={T_SHIFT})")
        for t,s in zip(ts, mean_sg4n):
            marker=' <-- SHIFT' if t==T_SHIFT else ''
            print(f"  {t:6d} {s:8.4f}{marker}")

    # Exp B analysis
    B_geo  = [r for r in all_results if r['exp']=='B' and r['coupling']=='geo']
    B_ctrl = [r for r in all_results if r['exp']=='B' and r['coupling']=='ctrl']
    if B_geo and B_ctrl:
        geo_vals  = [r['sg4n'] for r in B_geo  if not math.isnan(r['sg4n'])]
        ctrl_vals = [r['sg4n'] for r in B_ctrl if not math.isnan(r['sg4n'])]
        threshold=np.percentile(ctrl_vals, 75)  # top quartile of ctrl as "high-structure"
        P_hi_geo  = float(np.mean([v>threshold for v in geo_vals]))
        P_hi_ctrl = float(np.mean([v>threshold for v in ctrl_vals]))
        print(f"\n=== Exp B: Basin fraction (N={len(B_geo)} seeds per condition) ===")
        print(f"  Threshold (75th pct of ctrl): {threshold:.4f}")
        print(f"  P_high(geo) = {P_hi_geo:.3f}  ({int(P_hi_geo*len(geo_vals))}/{len(geo_vals)})")
        print(f"  P_high(ctrl) = {P_hi_ctrl:.3f}  ({int(P_hi_ctrl*len(ctrl_vals))}/{len(ctrl_vals)})")
        print(f"  G_basin = P_high(geo)/P_high(ctrl) = {P_hi_geo/P_hi_ctrl:.2f}" if P_hi_ctrl>0 else "  G_basin: ctrl all low")
        print(f"  Mean sg4n: geo={mn(geo_vals):.4f}, ctrl={mn(ctrl_vals):.4f}, ratio={mn(geo_vals)/mn(ctrl_vals):.2f}")
        # Distribution quartiles
        print(f"  Geo  sg4n quartiles: {np.percentile(geo_vals,[25,50,75])}")
        print(f"  Ctrl sg4n quartiles: {np.percentile(ctrl_vals,[25,50,75])}")

    # Exp C analysis
    C_runs=[r for r in all_results if r['exp']=='C']
    if C_runs:
        ts_C=C_runs[0]['ts']
        n_C=len(ts_C)
        mean_sg4n_C=[mn([r['sg4ns'][i] for r in C_runs]) for i in range(n_C)]
        print(f"\n=== Exp C: Long-run saturation (N={len(C_runs)} seeds) ===")
        print(f"  {'t':>6} {'sg4n':>8}")
        for t,s in zip(ts_C, mean_sg4n_C):
            print(f"  {t:6d} {s:8.4f}")
        # Estimate T_sat: first t where |sg4n(t+1)-sg4n(t)| < 1% of sg4n(t)
        for i in range(len(mean_sg4n_C)-1):
            if mean_sg4n_C[i]>1e-3:
                rel_change=abs(mean_sg4n_C[i+1]-mean_sg4n_C[i])/mean_sg4n_C[i]
                if rel_change<0.01:
                    print(f"  T_sat estimate: ~{ts_C[i]} (rel_change={rel_change:.3f}<1%)")
                    break
        else:
            print(f"  T_sat: not reached by T={T_END_C}")
