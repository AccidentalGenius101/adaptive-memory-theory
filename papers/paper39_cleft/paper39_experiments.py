"""
paper39_experiments.py -- Synaptic Cleft Geometry: Deterministic Shift vs Stochastic Noise

Tests whether a deterministic spatial offset (D_cleft) between two VCML lattices
is more damaging to relay gain than stochastic positional noise (sigma) of the
same magnitude.

Exp A: D_cleft sweep -- deterministic horizontal shift applied to wave positions
  L2 receives waves at (cx + D_cleft) mod HALF, with cls updated from new position.
  D_cleft in {0, 2, 5, 8, 10, 12, 15, 20} -- spans from perfect relay (0) to
  two-zone shift (20=2*zw for N=4, zw=10).
  Prediction: G drops to 1 at D_cleft=zw=10 (all waves land in wrong zone).

Exp B: sigma sweep (extended) -- stochastic Gaussian noise per wave position
  sigma in {2, 5, 8, 10, 12, 15, 20} -- matches D_cleft values.
  D_cleft=0 / sigma=0 is the shared baseline (no offset).
  Prediction: G drops slower than Exp A; stochastic noise averages out across waves.

Both use the same ctrl baseline (random waves) for G computation.
G = sg4n(L2_shifted) / sg4n(ctrl)

Key prediction: G(D_cleft=m) < G(sigma=m) for m > 0 because deterministic shift
causes systematic zone misclassification while stochastic noise partially averages.
Critical cleft width: D_cleft* where G=1 expected at D_cleft=zw=10.
Critical noise: sigma* where G=1 expected at sigma > zw (larger than deterministic).
"""
import numpy as np, json, os, math, random, multiprocessing as mp

# ── Constants ───────────────────────────────────────────────────────────────
W=80; H=40; HALF=40
HS=2; IS=3
WAVE_DUR=15
SUPP_AMP=0.25; EXC_AMP=0.50
ZONE_K=320
MID_DECAY=0.99; FIELD_DECAY=0.9997; BASE_BETA=0.005; SS=10
FA=0.16; VALS_DECAY=0.92; VALS_NAV=0.08; ADJ_SCALE=0.03
BOUND_LO=0.05; BOUND_HI=0.95; FRAG_NOISE=0.01; SEED_BETA=0.25
INST_THRESH=0.45; INST_PROB=0.03; DIFFUSE=0.02; DIFFUSE_EVERY=2

WR_FIXED=4.8; N_FIXED=4; ZW=HALF//N_FIXED  # =10

SEEDS = list(range(5))
STEPS = 3000
SAMPLE_EVERY=20; WARMUP=300; TAIL=30

# Exp A: deterministic cleft shift (modular in zone space)
D_CLEFT_SWEEP = [0, 2, 5, 8, 10, 12, 15, 20]
# Exp B: stochastic noise (extended from Paper 37)
SIGMA_SWEEP = [2, 5, 8, 10, 12, 15, 20]  # 0 covered by D_cleft=0

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper39_results.json")


# ── FastVCML ────────────────────────────────────────────────────────────────
class FastVCML:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
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

    def step(self,wa,wc):
        nb=self._nbmean(); an=np.minimum(1.,self.age/300.)
        x=np.stack([nb,np.minimum(1.,wa),an],1)
        hn,out=self._gru(x); self.hid=hn
        self.vals=np.clip(VALS_DECAY*self.vals+VALS_NAV*nb+ADJ_SCALE*out,0,1)
        dev=hn-self.base_h; self.base_h+=BASE_BETA*dev
        self.streak=np.where(np.sum(dev**2,1)<.0025,self.streak+1,0)
        self.mid=(self.mid+FA*dev)*MID_DECAY
        gate=self.streak>=SS; self.fieldM[gate]+=FA*(self.mid[gate]-self.fieldM[gate])
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


# ── Cleft Environments ────────────────────────────────────────────────────
class CleftEnv:
    """
    Two-lattice relay with configurable cleft model.
    mode='det':  L2 receives L1 waves shifted by d_cleft sites (modular wrap in zone space).
    mode='stoch': L2 receives L1 waves displaced by N(0, sigma^2) per axis.
    mode='ctrl':  L2 receives independent random waves.
    L1 always receives perfect geo waves (consistent zone positions).
    """
    def __init__(self, l1, l2, mode, WR, n_zones, shift_param, rng_seed=0):
        self.l1=l1; self.l2=l2
        self.mode=mode  # 'det', 'stoch', 'ctrl'
        self.WR=WR; self.n_zones=n_zones
        self.shift_param=shift_param  # d_cleft or sigma
        self.zw=HALF//n_zones
        self.rng=random.Random(rng_seed)
        self.np_rng=np.random.RandomState(rng_seed+777)
        N=l1.N; self.ax=l1.ai_x; self.ay=l1.ai_y
        self.wv1=[]; self.wv2=[]
        self._wa1=np.empty(N); self._wc1=np.empty(N,int)
        self._wa2=np.empty(N); self._wc2=np.empty(N,int)

    def _launch(self):
        rng=self.rng; m=self.mode; p=self.shift_param; zw=self.zw; nz=self.n_zones
        # L1: perfect geo wave
        cls=rng.randint(0, nz-1)
        zs=HALF+cls*zw; ze=HALF+(cls+1)*zw-1
        cx=rng.randint(zs, ze); cy=rng.randint(0, H-1)
        self.wv1.append([cx, cy, WAVE_DUR, cls])
        # L2: shifted wave
        if m=='det':
            # Deterministic modular shift: (cx-HALF+d_cleft) % HALF + HALF
            cx2=HALF+(cx-HALF+p)%HALF
            cls2=min(nz-1, (cx2-HALF)//zw)
            cy2=cy
        elif m=='stoch':
            dx=int(round(self.np_rng.normal(0, p)))
            dy=int(round(self.np_rng.normal(0, p)))
            cx2=max(HALF, min(HALF+HALF-1, cx+dx))
            cy2=max(0, min(H-1, cy+dy))
            cls2=min(nz-1, max(0, (cx2-HALF)//zw))
        else:  # ctrl
            cx2=rng.randint(HALF, HALF+HALF-1)
            cy2=rng.randint(0, H-1)
            cls2=rng.randint(0, nz-1)
        self.wv2.append([cx2, cy2, WAVE_DUR, cls2])

    def _apply(self, waves, vcml, wa, wc):
        wa.fill(0.); wc.fill(-1); ax=self.ax; ay=self.ay; surv=[]
        for wave in waves:
            cx,cy,rem,cls=wave
            if rem<=0: continue
            dist=np.abs(ax-cx)+np.abs(ay-cy)
            act=np.maximum(0., 1.-dist*0.4); act[dist>2]=0.
            better=act>wa; wa[better]=act[better]; wc[better]=cls
            wave[2]-=1
            if wave[2]>0: surv.append(wave)
        for even in [True, False]:
            amp=SUPP_AMP if even else EXC_AMP
            par=(wc%2==0) if even else (wc%2==1)
            idx=np.where((wc>=0)&par&(wa>.05))[0]
            if not len(idx): continue
            sc=wa[idx]*amp
            if even: vcml.vals[idx]=np.maximum(0., vcml.vals[idx]*(1.-sc*.5))
            else:    vcml.vals[idx]=np.minimum(1., vcml.vals[idx]+sc*.5)
        return wa.copy(), wc.copy(), surv

    def step(self):
        exp=self.WR/WAVE_DUR
        nl=int(exp)+(1 if self.rng.random()<exp-int(exp) else 0)
        for _ in range(nl): self._launch()
        wa1,wc1,self.wv1=self._apply(self.wv1,self.l1,self._wa1,self._wc1)
        wa2,wc2,self.wv2=self._apply(self.wv2,self.l2,self._wa2,self._wc2)
        return wa1,wc1,wa2,wc2


def _sg4n_tail(vcml, n_zones, sample_every, warmup, tail):
    """sg4n averaged over the last `tail` samples after warmup (at STEPS end)."""
    return compute_sg4(vcml, n_zones)[1]


def run(seed, mode, shift_param):
    """Run one relay condition and return sg4n for L1 and L2."""
    l1=FastVCML(seed=seed*3)
    l2=FastVCML(seed=seed+11)
    env=CleftEnv(l1, l2, mode=mode, WR=WR_FIXED, n_zones=N_FIXED,
                 shift_param=shift_param, rng_seed=seed*100)
    sg4n_l1s=[]; sg4n_l2s=[]
    for t in range(STEPS):
        wa1,wc1,wa2,wc2=env.step()
        l1.step(wa1,wc1); l2.step(wa2,wc2)
        if t>=STEPS-TAIL*SAMPLE_EVERY and t%SAMPLE_EVERY==0:
            sg4n_l1s.append(compute_sg4(l1,N_FIXED)[1])
            sg4n_l2s.append(compute_sg4(l2,N_FIXED)[1])
    return {
        'mode': mode, 'shift_param': shift_param, 'seed': seed,
        'l1_sg4n': float(np.nanmean(sg4n_l1s)),
        'l2_sg4n': float(np.nanmean(sg4n_l2s)),
    }


def _worker(args):
    return run(*args)


def make_key(r):
    return f"{r['mode']},{r['shift_param']},{r['seed']}"


if __name__=='__main__':
    mp.freeze_support()

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f: all_results=json.load(f)
    else: all_results=[]

    done=set(make_key(r) for r in all_results)

    all_conds=[]
    # Exp A: deterministic cleft (includes D=0 as shared baseline)
    for d in D_CLEFT_SWEEP:
        for seed in SEEDS:
            if f"det,{d},{seed}" not in done:
                all_conds.append((seed, 'det', d))
    # Exp B: stochastic noise (sigma>0; sigma=0 equiv to det,0)
    for sigma in SIGMA_SWEEP:
        for seed in SEEDS:
            if f"stoch,{sigma},{seed}" not in done:
                all_conds.append((seed, 'stoch', sigma))
    # ctrl baseline (shift_param=0, mode=ctrl)
    for seed in SEEDS:
        if f"ctrl,0,{seed}" not in done:
            all_conds.append((seed, 'ctrl', 0))

    total = len(D_CLEFT_SWEEP)*5 + len(SIGMA_SWEEP)*5 + 5
    print(f"Total: {total}, todo: {len(all_conds)}, done: {len(done)}")
    if all_conds:
        n_proc=min(8, len(all_conds))
        print(f"Running {len(all_conds)} jobs on {n_proc} cores...")
        with mp.Pool(processes=n_proc) as pool:
            new_results=pool.map(_worker, all_conds)
        all_results.extend(new_results)
        with open(RESULTS_FILE,'w') as f: json.dump(all_results,f)
        print("Saved.")

    # ── Analysis ───────────────────────────────────────────────────────────
    from collections import defaultdict
    def mn(lst):
        v=[x for x in lst if x is not None and not math.isnan(x)]
        return float(np.mean(v)) if v else float('nan')
    def se(lst):
        v=[x for x in lst if x is not None and not math.isnan(x)]
        return float(np.std(v,ddof=1)/math.sqrt(len(v))) if len(v)>1 else 0.

    by=defaultdict(list)
    for r in all_results: by[(r['mode'],r['shift_param'])].append(r)

    ctrl_sg4n=mn([r['l2_sg4n'] for r in by[('ctrl',0)]])
    print(f"\nCtrl baseline: sg4n={ctrl_sg4n:.4f}")

    print(f"\n=== Exp A: G vs D_cleft (deterministic shift, zw={ZW}) ===")
    print(f"  {'D_cleft':>8} {'G':>8} {'SE':>6}  pred_misclass")
    for d in D_CLEFT_SWEEP:
        runs=by[('det',d)]
        sg4n_l2=mn([r['l2_sg4n'] for r in runs])
        G=sg4n_l2/ctrl_sg4n if ctrl_sg4n>1e-6 else float('nan')
        G_se=se([r['l2_sg4n']/ctrl_sg4n for r in runs if ctrl_sg4n>1e-6])
        misclass=min(1.0, d/ZW)
        print(f"  {d:8d} {G:8.3f} {G_se:6.3f}  (misclass~{misclass:.0%})")

    print(f"\n=== Exp B: G vs sigma (stochastic noise) ===")
    # Include det,0 as sigma=0
    det0_runs=by[('det',0)]
    sg4n_det0=mn([r['l2_sg4n'] for r in det0_runs])
    G_det0=sg4n_det0/ctrl_sg4n if ctrl_sg4n>1e-6 else float('nan')
    print(f"  {'sigma':>7} {'G':>8} {'SE':>6}")
    print(f"  {0:7d} {G_det0:8.3f}  (det,0 baseline)")
    for sigma in SIGMA_SWEEP:
        runs=by[('stoch',sigma)]
        sg4n_l2=mn([r['l2_sg4n'] for r in runs])
        G=sg4n_l2/ctrl_sg4n if ctrl_sg4n>1e-6 else float('nan')
        G_se=se([r['l2_sg4n']/ctrl_sg4n for r in runs if ctrl_sg4n>1e-6])
        print(f"  {sigma:7d} {G:8.3f} {G_se:6.3f}")

    print(f"\n=== Direct comparison at matched magnitudes ===")
    print(f"  {'shift':>6} {'G_det':>8} {'G_stoch':>9} {'det/stoch':>10}")
    for m in [2, 5, 8, 10, 12, 15, 20]:
        runs_d=by[('det',m)]; runs_s=by[('stoch',m)]
        G_d=mn([r['l2_sg4n'] for r in runs_d])/ctrl_sg4n if ctrl_sg4n>1e-6 else float('nan')
        G_s=mn([r['l2_sg4n'] for r in runs_s])/ctrl_sg4n if ctrl_sg4n>1e-6 else float('nan')
        ratio=G_d/G_s if G_s>1e-4 else float('nan')
        print(f"  {m:6d} {G_d:8.3f} {G_s:9.3f} {ratio:10.3f}")
