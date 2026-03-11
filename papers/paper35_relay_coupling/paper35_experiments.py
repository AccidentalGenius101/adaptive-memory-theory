"""
paper35_experiments.py -- Relay Coupling Quantitative Theory (Q6)

V100 established: geographic coupling (L2 receives waves at SAME SPATIAL POSITIONS
as L1) gives 3.48x sg4_norm gain over random-position control at WR=4.8, N_zones=4.

This paper quantifies G(WR, N_zones) across parameter space and fits:
  Theory: G(WR, N) = (K*N + WR) / (K + WR)
  Derivation: effective WR for ctrl = WR/N (random zone has ~1/N chance of
  matching any given site's zone) -> sg4_ctrl ~ sg4_geo(WR/N)
  -> G = sg4(WR) / sg4(WR/N) ~ (K+WR/N)/(K+WR) * N ... simplified to above.

Validation anchor from V100: G(WR=4.8, N=4) = 3.48x -> K ~ 23.

Framework: FastVCML (numpy vectorized, same as V100), GeoHierEnv with
zone-organized waves. Spatial waves: L2_geo gets same (cx, cy) as L1;
L2_ctrl gets random (cx2, cy2) within active region. Both at same rate WR.

Exp A: WR sweep (N_zones=4 fixed)
  WR in {0.6, 1.2, 2.4, 4.8, 9.6, 19.2} x 5 seeds x {geo, ctrl} = 60 runs

Exp B: N_zones sweep (WR=4.8 fixed)
  N in {2, 4, 5, 8} x 5 seeds x {geo, ctrl} = 40 runs
  (N values chosen to divide HALF=40 evenly: zone_width = 20,10,8,5)

Total: 100 runs (N=4, WR=4.8 overlap counted once = 90 unique)
"""
import numpy as np, json, os, math, random, multiprocessing as mp
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ── Constants (V100/V46 optimized) ────────────────────────────────────────────
W  = 80; H = 40; HALF = 40
HS = 2;  IS = 3
WAVE_DUR   = 15
SUPP_AMP   = 0.25; EXC_AMP = 0.50
ZONE_K     = 320
STEPS      = 3000
SEEDS      = list(range(5))
SAMPLE_EVERY = 20; WARMUP = 300; TAIL = 30

MID_DECAY=0.99; FIELD_DECAY=0.9997; BASE_BETA=0.005; SS=10
FA=0.16; VALS_DECAY=0.92; VALS_NAV=0.08; ADJ_SCALE=0.03
BOUND_LO=0.05; BOUND_HI=0.95; FRAG_NOISE=0.01; SEED_BETA=0.25
INST_THRESH=0.45; INST_PROB=0.03; DIFFUSE=0.02; DIFFUSE_EVERY=2

WR_SWEEP  = [0.6, 1.2, 2.4, 4.8, 9.6, 19.2]  # Exp A
N_FIXED   = 4
WR_FIXED  = 4.8
# N values that evenly divide HALF=40: zone_width = 20,10,8,5
N_SWEEP   = [2, 4, 5, 8]                        # Exp B


# ── FastVCML (V100 numpy-vectorized, each instance uses its OWN private RNG) ──
class FastVCML:
    """
    Private RNG (self.rng) prevents L1/L2 from polluting each other's
    random stream when they run in the same Python process.
    """
    def __init__(self, seed, static=False):
        self.rng = np.random.RandomState(seed)   # private — never shared
        N = HALF * H
        self.N=N; self.static=static; self.t=0
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
        hn=(1-z)*h+z*g; out=tanh(np.einsum('ni,ni->n',self.Wo,hn)+self.bo)
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
        self.age+=1
        if not self.static: self._collapse()
        self.t+=1

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
    zw = HALF // n_zones
    si = vcml.stable_zone(ZONE_K)
    xp = vcml.ai_x[si]
    za = np.minimum(n_zones-1,(xp-HALF)//zw)
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


# ── GeoHierEnv parameterized over WR and N_zones ─────────────────────────────
class GeoHierEnv:
    """
    L1 zone-organized waves + L2 coupling.
    geo: L2 gets same (cx,cy,cls) as L1  [geographic]
    ctrl: L2 gets random (cx2,cy2,cls2)  [position-scrambled]
    """
    def __init__(self,l1,l2,coupling,WR,n_zones,rng_seed=0):
        self.l1=l1; self.l2=l2; self.coupling=coupling
        self.WR=WR; self.n_zones=n_zones
        self.zw=HALF//n_zones
        # Private Python Random instance — no global state pollution
        import random as _random
        self.rng = _random.Random(rng_seed)
        N=l1.N; self.ax=l1.ai_x; self.ay=l1.ai_y
        self.wv1=[]; self.wv2=[]
        self._d=np.empty(N); self._w=np.empty(N); self._b=np.empty(N,bool)
        self._wa1=np.empty(N); self._wc1=np.empty(N,int)
        self._wa2=np.empty(N); self._wc2=np.empty(N,int)

    def _launch(self):
        rng=self.rng
        cls=rng.randint(0,self.n_zones-1)
        zs=HALF+cls*self.zw; ze=HALF+(cls+1)*self.zw-1
        cx=rng.randint(zs,ze); cy=rng.randint(0,H-1)
        self.wv1.append([cx,cy,WAVE_DUR,cls])
        if self.l2 is not None:
            if self.coupling=='geo':
                self.wv2.append([cx,cy,WAVE_DUR,cls])
            else:
                cx2=rng.randint(HALF,HALF+HALF-1); cy2=rng.randint(0,H-1)
                cls2=rng.randint(0,self.n_zones-1)
                self.wv2.append([cx2,cy2,WAVE_DUR,cls2])

    def _apply(self,waves,vcml,wa,wc):
        wa.fill(0.); wc.fill(-1); ax=self.ax; ay=self.ay; surv=[]
        for wave in waves:
            cx,cy,rem,cls=wave
            if rem<=0: continue
            dist=np.abs(ax-cx)+np.abs(ay-cy)
            act=np.maximum(0.,1.-dist*.4); act[dist>2]=0.
            better=act>wa; wa[better]=act[better]; wc[better]=cls
            wave[2]-=1
            if wave[2]>0: surv.append(wave)
        for even in [True,False]:
            amp=SUPP_AMP if even else EXC_AMP
            par=(wc%2==0) if even else (wc%2==1)
            idx=np.where((wc>=0)&par&(wa>.05))[0]
            if not len(idx): continue
            sc=wa[idx]*amp
            if even: vcml.vals[idx]=np.maximum(0.,vcml.vals[idx]*(1.-sc*.5))
            else:    vcml.vals[idx]=np.minimum(1.,vcml.vals[idx]+sc*.5)
        return wa.copy(),wc.copy(),surv

    def step(self):
        exp=self.WR/WAVE_DUR; nl=int(exp)+(1 if self.rng.random()<exp-int(exp) else 0)
        for _ in range(nl): self._launch()
        wa1,wc1,self.wv1=self._apply(self.wv1,self.l1,self._wa1,self._wc1)
        wa2,wc2=None,None
        if self.l2:
            wa2,wc2,self.wv2=self._apply(self.wv2,self.l2,self._wa2,self._wc2)
        return wa1,wc1,wa2,wc2


# ── Single run ────────────────────────────────────────────────────────────────
def run(seed, WR, n_zones, coupling):
    # Match V100's seed scheme: L1=seed*3, L2=seed+11, env=seed*100
    l1=FastVCML(seed=seed*3)
    l2=FastVCML(seed=seed+11) if coupling else None
    env=GeoHierEnv(l1,l2,coupling,WR,n_zones,rng_seed=seed*100)
    log1=[]; log2=[]
    for t in range(STEPS):
        wa1,wc1,wa2,wc2=env.step()
        l1.step(wa1,wc1)
        if l2: l2.step(wa2,wc2)
        if t>=WARMUP and t%SAMPLE_EVERY==0:
            _,sn1=compute_sg4(l1,n_zones); log1.append(sn1)
            if l2: _,sn2=compute_sg4(l2,n_zones); log2.append(sn2)
    def tm(lst):
        lst=[x for x in lst if not math.isnan(x)]
        return float(np.mean(lst[-TAIL:])) if lst else float('nan')
    _,sg1=compute_sg4(l1,n_zones)
    r={'seed':seed,'WR':WR,'n_zones':n_zones,'coupling':coupling,
       'l1_sg4n':float(sg1),'l1_tm':tm(log1)}
    if l2:
        _,sg2=compute_sg4(l2,n_zones)
        r.update({'l2_sg4n':float(sg2),'l2_tm':tm(log2)})
    return r

def _worker(args): return run(*args)


# ── Key factories ─────────────────────────────────────────────────────────────
def make_key(seed,WR,n_zones,coupling): return f"{seed},{WR:.8g},{n_zones},{coupling}"

RESULTS_FILE=os.path.join(os.path.dirname(__file__),"results","paper35_results.json")


if __name__=="__main__":
    mp.freeze_support()

    # Build all runs
    all_args=[]
    for WR in WR_SWEEP:
        for c in ['geo','ctrl']:
            for s in SEEDS: all_args.append((s,WR,N_FIXED,c))
        for s in SEEDS: all_args.append((s,WR,N_FIXED,None))  # L1 ref
    for N in N_SWEEP:
        if N==N_FIXED: continue  # already covered
        for c in ['geo','ctrl']:
            for s in SEEDS: all_args.append((s,WR_FIXED,N,c))
        for s in SEEDS: all_args.append((s,WR_FIXED,N,None))

    # Deduplicate
    seen=set(); dedup=[]
    for a in all_args:
        k=make_key(*a);
        if k not in seen: seen.add(k); dedup.append(a)
    all_args=dedup

    # Load cache
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f: existing=json.load(f)
        done={make_key(r['seed'],r['WR'],r['n_zones'],r['coupling']) for r in existing}
    else:
        existing=[]; done=set()

    todo=[a for a in all_args if make_key(*a) not in done]
    print(f"Total runs: {len(all_args)}, todo: {len(todo)}")

    if todo:
        n_proc=min(mp.cpu_count(),len(todo),8)
        print(f"Running {len(todo)} jobs on {n_proc} cores...")
        with mp.Pool(n_proc) as pool: new=pool.map(_worker,todo)
        existing.extend(new)
        os.makedirs(os.path.dirname(RESULTS_FILE),exist_ok=True)
        with open(RESULTS_FILE,'w') as f: json.dump(existing,f,indent=2)
        print("Saved.")

    # ── Analysis ──────────────────────────────────────────────────────────────
    from collections import defaultdict
    by=defaultdict(list)
    for r in existing: by[(r['WR'],r['n_zones'],r['coupling'])].append(r)

    def mn(lst): v=[x for x in lst if not math.isnan(x)]; return float(np.mean(v)) if v else float('nan')

    print("\n=== Exp A: WR sweep (N_zones=4) ===")
    print(f"{'WR':>6} {'geo_sg4n':>10} {'ctrl_sg4n':>10} {'G':>7} {'l1_sg4n':>9}")
    for WR in WR_SWEEP:
        geo=by[(WR,N_FIXED,'geo')]; ctrl=by[(WR,N_FIXED,'ctrl')]; ref=by[(WR,N_FIXED,None)]
        sg=mn([r.get('l2_sg4n',float('nan')) for r in geo])
        sc=mn([r.get('l2_sg4n',float('nan')) for r in ctrl])
        s1=mn([r.get('l1_sg4n',float('nan')) for r in (geo or ref)])
        G=sg/sc if sc>1e-6 else float('nan')
        print(f"{WR:6.1f} {sg:10.4f} {sc:10.4f} {G:7.3f} {s1:9.4f}")

    print("\n=== Exp B: N_zones sweep (WR=4.8) ===")
    print(f"{'N':>4} {'geo_sg4n':>10} {'ctrl_sg4n':>10} {'G':>7} {'l1_sg4n':>9}")
    for N in N_SWEEP:
        geo=by[(WR_FIXED,N,'geo')]; ctrl=by[(WR_FIXED,N,'ctrl')]; ref=by[(WR_FIXED,N,None)]
        sg=mn([r.get('l2_sg4n',float('nan')) for r in geo])
        sc=mn([r.get('l2_sg4n',float('nan')) for r in ctrl])
        s1=mn([r.get('l1_sg4n',float('nan')) for r in (geo or ref)])
        G=sg/sc if sc>1e-6 else float('nan')
        print(f"{N:4d} {sg:10.4f} {sc:10.4f} {G:7.3f} {s1:9.4f}")

    # Theory fit G(WR, N) = (K*N + WR) / (K + WR)
    pts=[]
    for (WR,N,coup),rows in by.items():
        if coup!='geo': continue
        ctrl_rows=by[(WR,N,'ctrl')]
        if not ctrl_rows: continue
        sg=mn([r.get('l2_sg4n',float('nan')) for r in rows])
        sc=mn([r.get('l2_sg4n',float('nan')) for r in ctrl_rows])
        if sc>1e-6 and not math.isnan(sg): pts.append((WR,N,sg/sc))
    if pts:
        WRs=np.array([p[0] for p in pts]); Ns=np.array([p[1] for p in pts])
        Gs=np.array([p[2] for p in pts])
        try:
            def gm(X,K): WR,N=X; return (K*N+WR)/(K+WR)
            popt,_=curve_fit(gm,(WRs,Ns),Gs,p0=[10.],bounds=([.01],[1e4]))
            K=float(popt[0])
            print(f"\n=== Theory fit: G(WR,N) = (K*N + WR)/(K + WR),  K={K:.2f} ===")
            print(f"{'WR':>6} {'N':>4} {'G_obs':>8} {'G_pred':>8} {'err':>7}")
            for WR,N,G in sorted(pts):
                pred=(K*N+WR)/(K+WR)
                print(f"{WR:6.1f} {N:4.0f} {G:8.3f} {pred:8.3f} {abs(G-pred):7.3f}")
            V100_pred=(K*4+4.8)/(K+4.8)
            print(f"\nV100 anchor: G(WR=4.8,N=4) predicted={V100_pred:.3f}x, observed=3.48x")
        except Exception as e:
            print(f"Theory fit failed: {e}")
    print("\nAll done.")
