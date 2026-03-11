"""
paper37_experiments.py -- Wave Radius Sweep + K-Stage Relay Chain

Exp A: r_wave sweep -- confirms N_crit = L/(4*r_wave) from Paper 36
  r_wave in {1, 2, 3, 4}: change wave activation footprint radius
  Wave activation: act = max(0, 1 - dist * 0.8/r_wave), dist>r_wave -> 0
  (coeff=0.8/r_wave ensures edge site at dist=r_wave always gets act=0.2)
  WR=4.8 fixed. N_SWEEP adapted per r_wave to straddle N_crit_pred.
  Predictions: N_crit = L/(4*r_wave) = {10, 5, 3.3, 2.5} for r_wave={1,2,3,4}

Exp B: k-stage relay chain -- tests G compounding vs saturation
  k stages L1->L2->...->Lk at WR=4.8, N=4, r_wave=2 (standard).
  Each stage receives previous stage's wave positions + Gaussian noise sigma.
  sigma=0: all stages receive identical positions -> G_k = G_1 (flat, baseline)
  sigma>0: positions degrade per hop -> G_k decreases with k.
  Measures: sg4n at each stage Lk, ratio sg4n(Lk)/sg4n(L1).
  Prediction: G does NOT compound (bimodal attractor saturates at G_1).
              G decays geometrically with k for sigma>0: G_k ~ G_1*(1-sigma/zw)^(k-1)
"""
import numpy as np, json, os, math, random, multiprocessing as mp
from collections import defaultdict

# ── Constants ──────────────────────────────────────────────────────────────────
W=80; H=40; HALF=40
HS=2; IS=3
WAVE_DUR=15
SUPP_AMP=0.25; EXC_AMP=0.50
ZONE_K=320
STEPS=3000
SEEDS=list(range(5))
SAMPLE_EVERY=20; WARMUP=300; TAIL=30

MID_DECAY=0.99; FIELD_DECAY=0.9997; BASE_BETA=0.005; SS=10
FA=0.16; VALS_DECAY=0.92; VALS_NAV=0.08; ADJ_SCALE=0.03
BOUND_LO=0.05; BOUND_HI=0.95; FRAG_NOISE=0.01; SEED_BETA=0.25
INST_THRESH=0.45; INST_PROB=0.03; DIFFUSE=0.02; DIFFUSE_EVERY=2

WR_FIXED=4.8; N_FIXED=4

# Exp A: r_wave sweep, N adapted per r_wave
R_WAVE_SWEEP=[1, 2, 3, 4]
# N values that straddle N_crit_pred = L/(4*r_wave) for each r_wave
# Valid N: divisors of HALF=40: 1,2,4,5,8,10,20,40
N_PER_RWAVE={
    1: [5, 8, 10, 20],  # N_crit_pred=10: test across {5,8,10,20}
    2: [2, 4, 5, 8],    # N_crit_pred=5:  standard P35/P36 sweep
    3: [2, 4, 5],       # N_crit_pred=3.3: N=2 (above), N=4,5 (below)
    4: [2, 4, 5],       # N_crit_pred=2.5: N=2 (above), N=4,5 (well below)
}

# Exp B: k-stage chain
K_STAGES=[1, 2, 3, 4]
SIGMA_SWEEP=[0, 2, 5]  # positional noise (sites) per hop
N_CHAIN=4              # N_zones for chain experiment

RESULTS_FILE=os.path.join(os.path.dirname(__file__),"results","paper37_results.json")


# ── FastVCML (private RNG, same as P35/P36) ───────────────────────────────────
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


# ── Exp A: GeoEnv with parameterised r_wave ───────────────────────────────────
class GeoEnvRwave:
    """Wave activation radius r_wave parameterised.
    coeff = 0.8/r_wave ensures edge site (dist=r_wave) gets act=0.2."""
    def __init__(self, l1, l2, coupling, WR, n_zones, r_wave, rng_seed=0):
        self.l1=l1; self.l2=l2; self.coupling=coupling
        self.WR=WR; self.n_zones=n_zones; self.r_wave=r_wave
        self.coeff=0.8/r_wave
        self.zw=HALF//n_zones
        import random as _r; self.rng=_r.Random(rng_seed)
        N=l1.N; self.ax=l1.ai_x; self.ay=l1.ai_y
        self.wv1=[]; self.wv2=[]
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
        wa.fill(0.); wc.fill(-1); ax=self.ax; ay=self.ay
        coeff=self.coeff; rw=self.r_wave; surv=[]
        for wave in waves:
            cx,cy,rem,cls=wave
            if rem<=0: continue
            dist=np.abs(ax-cx)+np.abs(ay-cy)
            act=np.maximum(0., 1.-dist*coeff); act[dist>rw]=0.
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


# ── Exp B: ChainEnv (k stages, positional noise sigma per hop) ────────────────
class ChainEnv:
    """
    k-stage relay chain. Each stage receives previous stage's wave positions
    + Gaussian noise sigma (sites). cls is position-derived (updated at each hop).
    sigma=0: all stages receive identical waves (no relay degradation).
    sigma>0: positions degrade per hop; cls updated to reflect new position.
    """
    def __init__(self, lattices, WR, n_zones, sigma, rng_seed=0):
        self.lattices=list(lattices); self.k=len(lattices)
        self.WR=WR; self.n_zones=n_zones; self.sigma=sigma
        self.zw=HALF//n_zones
        import random as _r; self.rng=_r.Random(rng_seed)
        self.np_rng=np.random.RandomState(rng_seed+999)
        N=lattices[0].N; self.ax=lattices[0].ai_x; self.ay=lattices[0].ai_y
        self.wave_queues=[[] for _ in lattices]
        self._was=[np.empty(N) for _ in lattices]
        self._wcs=[np.empty(N,int) for _ in lattices]

    def _launch(self):
        rng=self.rng; np_rng=self.np_rng
        cls=rng.randint(0,self.n_zones-1)
        zs=HALF+cls*self.zw; ze=HALF+(cls+1)*self.zw-1
        cx=rng.randint(zs,ze); cy=rng.randint(0,H-1)
        self.wave_queues[0].append([cx,cy,WAVE_DUR,cls])
        pcx,pcy=cx,cy
        for i in range(1,self.k):
            if self.sigma>0:
                nx=int(round(np_rng.normal(0,self.sigma)))
                ny=int(round(np_rng.normal(0,self.sigma)))
                ncx=max(HALF,min(HALF+HALF-1,pcx+nx))
                ncy=max(0,min(H-1,pcy+ny))
                # cls updated based on new position
                ncls=max(0,min(self.n_zones-1,(ncx-HALF)//self.zw))
            else:
                ncx,ncy,ncls=pcx,pcy,cls
            self.wave_queues[i].append([ncx,ncy,WAVE_DUR,ncls])
            pcx,pcy=ncx,ncy

    def _apply_one(self,waves,vcml,wa,wc):
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
        results=[]
        for i,lat in enumerate(self.lattices):
            wa,wc,self.wave_queues[i]=self._apply_one(self.wave_queues[i],lat,self._was[i],self._wcs[i])
            results.append((wa,wc))
        return results


# ── Run functions ──────────────────────────────────────────────────────────────
def run_A(seed, WR, n_zones, coupling, r_wave):
    """Exp A: single-stage relay gain at varied r_wave."""
    l1=FastVCML(seed=seed*3)
    l2=FastVCML(seed=seed+11) if coupling else None
    env=GeoEnvRwave(l1,l2,coupling,WR,n_zones,r_wave,rng_seed=seed*100)
    log1=[]; log2=[]
    for t in range(STEPS):
        wa1,wc1,wa2,wc2=env.step()
        l1.step(wa1,wc1)
        if l2: l2.step(wa2,wc2)
        if t>=WARMUP and t%SAMPLE_EVERY==0:
            _,sn1=compute_sg4(l1,n_zones); log1.append(sn1)
            if l2: _,sn2=compute_sg4(l2,n_zones); log2.append(sn2)
    def tm(lst): lst=[x for x in lst if not math.isnan(x)]; return float(np.mean(lst[-TAIL:])) if lst else float('nan')
    _,sg1=compute_sg4(l1,n_zones)
    r={'seed':seed,'WR':WR,'n_zones':n_zones,'coupling':coupling,'r_wave':r_wave,'exp':'A',
       'l1_sg4n':float(sg1),'l1_tm':tm(log1)}
    if l2:
        _,sg2=compute_sg4(l2,n_zones)
        r.update({'l2_sg4n':float(sg2),'l2_tm':tm(log2)})
    return r


def run_B(seed, sigma, k_stages):
    """Exp B: k-stage relay chain at varied positional noise sigma."""
    lattices=[FastVCML(seed=seed*3+i*17) for i in range(k_stages)]
    env=ChainEnv(lattices, WR_FIXED, N_CHAIN, sigma, rng_seed=seed*100)
    logs=[[] for _ in range(k_stages)]
    for t in range(STEPS):
        results=env.step()
        for i,(lat,(wa,wc)) in enumerate(zip(lattices,results)):
            lat.step(wa,wc)
        if t>=WARMUP and t%SAMPLE_EVERY==0:
            for i,lat in enumerate(lattices):
                _,sn=compute_sg4(lat,N_CHAIN); logs[i].append(sn)
    def tm(lst): lst=[x for x in lst if not math.isnan(x)]; return float(np.mean(lst[-TAIL:])) if lst else float('nan')
    sg4ns=[compute_sg4(lat,N_CHAIN)[1] for lat in lattices]
    tms=[tm(logs[i]) for i in range(k_stages)]
    r={'seed':seed,'sigma':sigma,'k_stages':k_stages,'exp':'B',
       'sg4ns':[float(x) for x in sg4ns],'tms':tms}
    return r


def run_B_ctrl(seed):
    """Exp B control: single stage with random waves (ctrl baseline for G calculation)."""
    l1=FastVCML(seed=seed*3)
    l_ctrl=FastVCML(seed=seed+11)
    env=GeoEnvRwave(l1,l_ctrl,'ctrl',WR_FIXED,N_CHAIN,2,rng_seed=seed*100)
    logs=[]; logc=[]
    for t in range(STEPS):
        wa1,wc1,wa2,wc2=env.step()
        l1.step(wa1,wc1); l_ctrl.step(wa2,wc2)
        if t>=WARMUP and t%SAMPLE_EVERY==0:
            _,sn1=compute_sg4(l1,N_CHAIN); logs.append(sn1)
            _,snc=compute_sg4(l_ctrl,N_CHAIN); logc.append(snc)
    def tm(lst): lst=[x for x in lst if not math.isnan(x)]; return float(np.mean(lst[-TAIL:])) if lst else float('nan')
    _,sg1=compute_sg4(l1,N_CHAIN); _,sgc=compute_sg4(l_ctrl,N_CHAIN)
    return {'seed':seed,'exp':'B_ctrl','l1_sg4n':float(sg1),'ctrl_sg4n':float(sgc),
            'l1_tm':tm(logs),'ctrl_tm':tm(logc)}


def _worker(args):
    exp=args[0]
    if exp=='A': return run_A(*args[1:])
    elif exp=='B': return run_B(*args[1:])
    elif exp=='B_ctrl': return run_B_ctrl(args[1])


# ── Key factory ───────────────────────────────────────────────────────────────
def make_key(r):
    exp=r.get('exp','?')
    if exp=='A': return f"A,{r['seed']},{r['WR']:.4g},{r['n_zones']},{r['coupling']},{r['r_wave']}"
    elif exp=='B': return f"B,{r['seed']},{r['sigma']},{r['k_stages']}"
    elif exp=='B_ctrl': return f"B_ctrl,{r['seed']}"


if __name__=="__main__":
    mp.freeze_support()

    all_args=[]

    # Exp A: r_wave sweep
    for rw in R_WAVE_SWEEP:
        for N in N_PER_RWAVE[rw]:
            for c in ['geo','ctrl']:
                for s in SEEDS: all_args.append(('A',s,WR_FIXED,N,c,rw))
            for s in SEEDS: all_args.append(('A',s,WR_FIXED,N,None,rw))

    # Exp B: k-stage chain
    for sigma in SIGMA_SWEEP:
        for k in K_STAGES:
            for s in SEEDS: all_args.append(('B',s,sigma,k))

    # Exp B ctrl (single stage, random waves baseline)
    for s in SEEDS: all_args.append(('B_ctrl',s))

    # Deduplicate
    seen=set(); dedup=[]
    for a in all_args:
        # generate a key from args
        if a[0]=='A': k=f"A,{a[1]},{a[2]:.4g},{a[3]},{a[4]},{a[5]}"
        elif a[0]=='B': k=f"B,{a[1]},{a[2]},{a[3]}"
        elif a[0]=='B_ctrl': k=f"B_ctrl,{a[1]}"
        if k not in seen: seen.add(k); dedup.append(a)
    all_args=dedup

    # Load cache
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f: existing=json.load(f)
        done={make_key(r) for r in existing}
    else:
        existing=[]; done=set()

    todo=[a for a in all_args if (
        (a[0]=='A' and f"A,{a[1]},{a[2]:.4g},{a[3]},{a[4]},{a[5]}" not in done) or
        (a[0]=='B' and f"B,{a[1]},{a[2]},{a[3]}" not in done) or
        (a[0]=='B_ctrl' and f"B_ctrl,{a[1]}" not in done)
    )]
    print(f"Total runs: {len(all_args)}, todo: {len(todo)}, done: {len(done)}")

    if todo:
        n_proc=min(mp.cpu_count(),len(todo),8)
        print(f"Running {len(todo)} jobs on {n_proc} cores...")
        with mp.Pool(n_proc) as pool: new=pool.map(_worker,todo)
        existing.extend(new)
        os.makedirs(os.path.dirname(RESULTS_FILE),exist_ok=True)
        with open(RESULTS_FILE,'w') as f: json.dump(existing,f,indent=2)
        print("Saved.")

    # ── Analysis ──────────────────────────────────────────────────────────────
    def mn(lst): v=[x for x in lst if not math.isnan(x)]; return float(np.mean(v)) if v else float('nan')

    by_A=defaultdict(list)
    by_B=defaultdict(list)
    by_Bc=[]
    for r in existing:
        if r.get('exp')=='A': by_A[(r['r_wave'],r['n_zones'],r['coupling'])].append(r)
        elif r.get('exp')=='B': by_B[(r['sigma'],r['k_stages'])].append(r)
        elif r.get('exp')=='B_ctrl': by_Bc.append(r)

    # ── Exp A ──────────────────────────────────────────────────────────────────
    print("\n=== Exp A: G(N) per r_wave ===")
    print(f"{'rw':>4} {'N':>4} {'zw':>5} {'zw/rw':>7} {'G':>7} {'N_crit_pred':>12}")
    for rw in R_WAVE_SWEEP:
        nc_pred=HALF/(4*rw)
        for N in N_PER_RWAVE[rw]:
            zw=HALF//N
            geo=by_A[(rw,N,'geo')]; ctrl=by_A[(rw,N,'ctrl')]
            sg=mn([r.get('l2_sg4n',float('nan')) for r in geo])
            sc=mn([r.get('l2_sg4n',float('nan')) for r in ctrl])
            G=sg/sc if sc>1e-6 else float('nan')
            print(f"{rw:4d} {N:4d} {zw:5d} {zw/rw:7.2f} {G:7.3f} {nc_pred:12.1f}")
        print()

    print("=== Exp A: N_crit summary ===")
    print(f"{'r_wave':>8} {'N_crit_pred':>12} {'N_crit_obs':>12}")
    for rw in R_WAVE_SWEEP:
        nc_pred=HALF/(4*rw)
        Gs=[]
        for N in N_PER_RWAVE[rw]:
            geo=by_A[(rw,N,'geo')]; ctrl=by_A[(rw,N,'ctrl')]
            sg=mn([r.get('l2_sg4n',float('nan')) for r in geo])
            sc=mn([r.get('l2_sg4n',float('nan')) for r in ctrl])
            G=sg/sc if sc>1e-6 else float('nan')
            if not math.isnan(G): Gs.append((N,G))
        nc_obs=max((N for N,G in Gs if G>1.5),default=float('nan'))
        print(f"{rw:8d} {nc_pred:12.1f} {nc_obs:12.1f}")

    # ── Exp B ──────────────────────────────────────────────────────────────────
    ctrl_sg4n=mn([r.get('ctrl_sg4n',float('nan')) for r in by_Bc])
    l1_sg4n=mn([r.get('l1_sg4n',float('nan')) for r in by_Bc])
    print(f"\n=== Exp B: k-stage relay chain ===")
    print(f"Ctrl baseline: ctrl_sg4n={ctrl_sg4n:.4f}, l1_sg4n={l1_sg4n:.4f}, G_baseline={l1_sg4n/ctrl_sg4n:.3f}")
    print(f"\n{'sigma':>8} {'k':>4} {'sg4n_L1':>10} {'sg4n_Lk':>10} {'ratio_Lk/L1':>13} {'G_k':>8}")
    for sigma in SIGMA_SWEEP:
        for k in K_STAGES:
            rows=by_B[(sigma,k)]
            if not rows: continue
            # sg4ns[0] = L1, sg4ns[k-1] = Lk
            s1=mn([r['sg4ns'][0] if len(r['sg4ns'])>0 else float('nan') for r in rows])
            sk=mn([r['sg4ns'][k-1] if len(r['sg4ns'])>=k else float('nan') for r in rows])
            ratio=sk/s1 if s1>1e-6 else float('nan')
            Gk=sk/ctrl_sg4n if ctrl_sg4n>1e-6 else float('nan')
            print(f"{sigma:8.1f} {k:4d} {s1:10.4f} {sk:10.4f} {ratio:13.3f} {Gk:8.3f}")
        print()
    print("All done.")
