"""
paper36_experiments.py -- xi as the Inter-Module Coupling Bandwidth (Q6 extension)

Paper 35 derived: N_crit = HALF / (4 * xi)
where xi is the correlation length from Paper 3.

Paper 3 law: xi ~ C * sqrt(kappa / nu)
  kappa = FIELD_ALPHA * DIFFUSE
  nu    = collapse rate (~constant at ~0.002/site)
  -> xi ∝ sqrt(DIFFUSE)

Paper 35 anchor: DIFFUSE=0.02 -> xi≈2.5 sites -> N_crit≈4

Predictions for this paper:
  DIFFUSE=0.005 -> xi≈1.25 sites -> N_crit≈8
  DIFFUSE=0.02  -> xi≈2.50 sites -> N_crit≈4  (anchor)
  DIFFUSE=0.08  -> xi≈5.00 sites -> N_crit≈2

Exp A: xi(DIFFUSE) -- measure spatial autocorrelation of fieldM along x-axis
  for each DIFFUSE: run single lattice (N_zones=2), extract C(r), fit xi
  DIFFUSE in {0.005, 0.01, 0.02, 0.04, 0.08}, 5 seeds each = 25 runs

Exp B: G(N, DIFFUSE) -- relay gain vs N_zones at each DIFFUSE
  DIFFUSE in {0.005, 0.02, 0.08}, N in {2,4,5,8,10} x {geo, ctrl} x 5 seeds
  = 3 x 5 x 2 x 5 = 150 runs (+ 25 ref runs)
  N=10 added for DIFFUSE=0.005 (N_crit predicted at 8, need N>8 to see full degradation)

Validation: N_crit(DIFFUSE) vs 1/xi(DIFFUSE) -- linear through origin, slope=HALF/4=10
"""
import numpy as np, json, os, math, random, multiprocessing as mp
from scipy.optimize import curve_fit
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
INST_THRESH=0.45; INST_PROB=0.03; DIFFUSE_EVERY=2

WR_FIXED=4.8

# Exp A: DIFFUSE sweep for xi measurement
DIFFUSE_SWEEP=[0.005, 0.01, 0.02, 0.04, 0.08]

# Exp B: DIFFUSE x N_zones for N_crit
DIFFUSE_TEST=[0.005, 0.02, 0.08]
# N values: must evenly divide HALF=40: zw = 40,20,10,8,5,4
N_SWEEP_FULL=[1,2,4,5,8,10]   # N=1 (whole region=one zone), N=10 (zw=4)
N_SWEEP_CORE=[2,4,5,8]        # core sweep for all DIFFUSE (matches P35)

RESULTS_FILE=os.path.join(os.path.dirname(__file__),"results","paper36_results.json")


# ── FastVCML with parameterised DIFFUSE ───────────────────────────────────────
class FastVCML:
    def __init__(self, seed, diffuse=0.02, static=False):
        self.rng    = np.random.RandomState(seed)
        self.diffuse= diffuse
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
        hn=(1-z)*h+z*g; out=np.tanh(np.einsum('ni,ni->n',self.Wo,hn)+self.bo)
        return hn,out

    def _diffuse(self):
        ns=np.maximum(self.nb,0); nf=self.fieldM[ns]
        nf*=(self.nb>=0)[:,:,None]
        nm=np.where(self.nbc[:,None]>0,nf.sum(1)/self.nbc[:,None],self.fieldM)
        self.fieldM+=self.diffuse*(nm-self.fieldM)

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

    def measure_xi(self, n_zones):
        """
        Measure spatial correlation length xi from fieldM x-autocorrelation.
        Uses stable sites within zone 0 (to avoid zone-alternation confound).
        Returns xi in sites, from exponential fit C(r) = A*exp(-r/xi).
        """
        zw = HALF // n_zones
        # All sites in zone 0
        mask = (self.ai_x >= HALF) & (self.ai_x < HALF + zw)
        fidx = np.where(mask)[0]
        if len(fidx) < 10:
            return float('nan')
        fm = self.fieldM[fidx]           # (n_sites, HS)
        xs = self.ai_x[fidx] - HALF     # relative x in [0, zw)

        # Column means: for each x-value, mean fieldM vector
        col_means = {}
        for xi in range(zw):
            sel = fm[xs == xi]
            if len(sel) > 0:
                col_means[xi] = sel.mean(0)

        if len(col_means) < 3:
            return float('nan')

        # Autocorrelation C(r) = mean_x dot(col_mean(x), col_mean(x+r)) / dot(col_mean(x), col_mean(x))
        xs_avail = sorted(col_means.keys())
        var0 = np.mean([np.dot(col_means[x], col_means[x]) for x in xs_avail])
        if var0 < 1e-12:
            return float('nan')

        rs, Cs = [], []
        for r in range(1, zw):
            pairs = [(x, x+r) for x in xs_avail if x+r in col_means]
            if len(pairs) < 2:
                continue
            corr = np.mean([np.dot(col_means[a], col_means[b]) for a,b in pairs])
            rs.append(r); Cs.append(corr / var0)

        if len(rs) < 3:
            return float('nan')

        rs_arr = np.array(rs, float); Cs_arr = np.array(Cs, float)
        # Fit C(r) = A * exp(-r/xi); only use positive correlations
        pos = Cs_arr > 0
        if pos.sum() < 2:
            return float('nan')
        try:
            def exp_model(r, A, xi): return A * np.exp(-r / xi)
            popt, _ = curve_fit(exp_model, rs_arr[pos], Cs_arr[pos],
                                p0=[1.0, 3.0], bounds=([0, 0.1], [5, 50]), maxfev=2000)
            return float(popt[1])
        except Exception:
            return float('nan')


def compute_sg4(vcml, n_zones):
    zw = HALF // n_zones
    si = vcml.stable_zone(ZONE_K)
    xp = vcml.ai_x[si]
    za = np.minimum(n_zones-1, (xp-HALF)//zw)
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


# ── GeoHierEnv ────────────────────────────────────────────────────────────────
class GeoHierEnv:
    def __init__(self,l1,l2,coupling,WR,n_zones,rng_seed=0):
        self.l1=l1; self.l2=l2; self.coupling=coupling
        self.WR=WR; self.n_zones=n_zones
        self.zw=HALF//n_zones
        import random as _random
        self.rng=_random.Random(rng_seed)
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


# ── Run function (Exp A + Exp B unified) ──────────────────────────────────────
def run(seed, WR, n_zones, coupling, diffuse, exp_type):
    """
    exp_type='A': single-lattice xi measurement (coupling=None, measure xi)
    exp_type='B': relay gain (coupling in {'geo','ctrl'})
    """
    l1 = FastVCML(seed=seed*3, diffuse=diffuse)
    l2 = FastVCML(seed=seed+11, diffuse=diffuse) if coupling else None
    n_z_env = n_zones if n_zones > 1 else 2  # env needs >=2 zones
    env = GeoHierEnv(l1, l2, coupling, WR, n_zones if n_zones>=2 else 2, rng_seed=seed*100)

    log1=[]; log2=[]
    for t in range(STEPS):
        wa1,wc1,wa2,wc2 = env.step()
        l1.step(wa1,wc1)
        if l2: l2.step(wa2,wc2)
        if t>=WARMUP and t%SAMPLE_EVERY==0:
            _,sn1=compute_sg4(l1,n_zones); log1.append(sn1)
            if l2: _,sn2=compute_sg4(l2,n_zones); log2.append(sn2)

    def tm(lst):
        lst=[x for x in lst if not math.isnan(x)]
        return float(np.mean(lst[-TAIL:])) if lst else float('nan')

    _,sg1=compute_sg4(l1,n_zones)
    xi1=l1.measure_xi(n_zones) if exp_type=='A' else float('nan')
    coll1=float(l1.cc.sum())/l1.N

    r={'seed':seed,'WR':WR,'n_zones':n_zones,'coupling':coupling,
       'diffuse':diffuse,'exp_type':exp_type,
       'l1_sg4n':float(sg1),'l1_tm':tm(log1),'l1_xi':xi1,'l1_coll':coll1}
    if l2:
        _,sg2=compute_sg4(l2,n_zones)
        xi2=l2.measure_xi(n_zones) if exp_type=='A' else float('nan')
        coll2=float(l2.cc.sum())/l2.N
        r.update({'l2_sg4n':float(sg2),'l2_tm':tm(log2),'l2_xi':xi2,'l2_coll':coll2})
    return r

def _worker(args): return run(*args)


# ── Key factory ───────────────────────────────────────────────────────────────
def make_key(seed,WR,n_zones,coupling,diffuse,exp_type):
    return f"{seed},{WR:.8g},{n_zones},{coupling},{diffuse:.6g},{exp_type}"


if __name__=="__main__":
    mp.freeze_support()

    all_args=[]

    # Exp A: xi measurement -- single lattice (coupling=None), varied DIFFUSE
    for D in DIFFUSE_SWEEP:
        for s in SEEDS:
            all_args.append((s, WR_FIXED, 2, None, D, 'A'))  # N=2: wide zones, strong xi signal

    # Exp B: relay gain -- varied DIFFUSE x N_zones x coupling
    for D in DIFFUSE_TEST:
        # Core N sweep
        for N in N_SWEEP_CORE:
            if HALF % N != 0: continue
            for c in ['geo','ctrl']:
                for s in SEEDS:
                    all_args.append((s, WR_FIXED, N, c, D, 'B'))
            # L1 reference
            for s in SEEDS:
                all_args.append((s, WR_FIXED, N, None, D, 'B'))
        # Extra N=10 for DIFFUSE=0.005 (need to see degradation past N_crit=8)
        if D == 0.005:
            for c in ['geo','ctrl']:
                for s in SEEDS:
                    all_args.append((s, WR_FIXED, 10, c, D, 'B'))
            for s in SEEDS:
                all_args.append((s, WR_FIXED, 10, None, D, 'B'))
        # N=1 for DIFFUSE=0.08 (N_crit=2, want to see strong G below threshold)
        # N=1: all one zone, G is undefined (no zone structure to relay)
        # Skip N=1

    # Deduplicate
    seen=set(); dedup=[]
    for a in all_args:
        k=make_key(*a)
        if k not in seen: seen.add(k); dedup.append(a)
    all_args=dedup

    # Load cache
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f: existing=json.load(f)
        done={make_key(r['seed'],r['WR'],r['n_zones'],r['coupling'],r['diffuse'],r['exp_type'])
              for r in existing}
    else:
        existing=[]; done=set()

    todo=[a for a in all_args if make_key(*a) not in done]
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
    from collections import defaultdict
    by=defaultdict(list)
    for r in existing: by[(r['diffuse'],r['n_zones'],r['coupling'],r['exp_type'])].append(r)

    def mn(lst): v=[x for x in lst if not math.isnan(x)]; return float(np.mean(v)) if v else float('nan')

    # ── Exp A: xi vs DIFFUSE ──────────────────────────────────────────────────
    print("\n=== Exp A: xi(DIFFUSE) ===")
    print(f"{'DIFFUSE':>10} {'xi_meas':>10} {'xi_pred':>10} {'ratio':>8}")
    xi_std = None; D_std = 0.02
    xi_data = []
    for D in DIFFUSE_SWEEP:
        rows = by[(D, 2, None, 'A')]
        xis  = [r.get('l1_xi', float('nan')) for r in rows]
        xi   = mn(xis)
        # Predicted: xi_pred = xi_std * sqrt(D/D_std)
        if D == D_std and not math.isnan(xi): xi_std = xi
        xi_data.append((D, xi))

    for D, xi in xi_data:
        xi_pred = (xi_std * math.sqrt(D/D_std)) if xi_std else float('nan')
        ratio   = xi/xi_pred if (xi_pred>0 and not math.isnan(xi)) else float('nan')
        print(f"{D:10.4g} {xi:10.3f} {xi_pred:10.3f} {ratio:8.3f}")

    # ── Exp B: G(N) at each DIFFUSE, find N_crit ──────────────────────────────
    print("\n=== Exp B: G(N) per DIFFUSE ===")
    N_SWEEP_ALL = sorted({a[2] for a in all_args if a[5]=='B'})
    ncrit_data = []
    for D in DIFFUSE_TEST:
        xi_D = mn([r.get('l1_xi',float('nan')) for r in by[(D,2,None,'A')]])
        xi_pred = (xi_std * math.sqrt(D/D_std)) if xi_std else float('nan')
        xi_use  = xi_D if not math.isnan(xi_D) else xi_pred
        print(f"\n  DIFFUSE={D}  (xi_meas={xi_D:.3f}, xi_pred={xi_pred:.3f})")
        print(f"  {'N':>4} {'zw':>6} {'zw/xi':>8} {'geo_sg4n':>10} {'ctrl_sg4n':>10} {'G':>7}")
        Gs = []
        for N in sorted(N_SWEEP_ALL):
            if HALF % N != 0: continue
            geo  = by[(D,N,'geo','B')]; ctrl=by[(D,N,'ctrl','B')]
            if not geo or not ctrl: continue
            sg=mn([r.get('l2_sg4n',float('nan')) for r in geo])
            sc=mn([r.get('l2_sg4n',float('nan')) for r in ctrl])
            G =sg/sc if sc>1e-6 else float('nan')
            zw=HALF//N
            zeta=zw/xi_use if not math.isnan(xi_use) else float('nan')
            print(f"  {N:4d} {zw:6d} {zeta:8.2f} {sg:10.4f} {sc:10.4f} {G:7.3f}")
            Gs.append((N,G))
        # Find N_crit: largest N where G > 1.5
        nc = max((N for N,G in Gs if not math.isnan(G) and G>1.5), default=float('nan'))
        nc_pred = HALF/(4*xi_use) if not math.isnan(xi_use) else float('nan')
        print(f"  N_crit (G>1.5 threshold) = {nc}  predicted={nc_pred:.1f}")
        ncrit_data.append((D, xi_use, nc, nc_pred))

    # ── Validation: N_crit vs 1/xi ────────────────────────────────────────────
    print("\n=== Validation: N_crit = HALF/(4*xi) ===")
    print(f"{'DIFFUSE':>10} {'xi':>8} {'N_crit_obs':>12} {'N_crit_pred':>13} {'err':>8}")
    for D, xi_use, nc, nc_pred in ncrit_data:
        err = abs(nc-nc_pred)/nc_pred if not math.isnan(nc_pred) and not math.isnan(nc) else float('nan')
        print(f"{D:10.4g} {xi_use:8.3f} {nc:12.1f} {nc_pred:13.1f} {err:8.3f}")

    # Power law fit: xi ~ C * DIFFUSE^alpha
    Ds_fit=[]; xis_fit=[]
    for D, xi in xi_data:
        if not math.isnan(xi) and xi>0: Ds_fit.append(D); xis_fit.append(xi)
    if len(Ds_fit)>=3:
        lD=np.log(Ds_fit); lX=np.log(xis_fit)
        from scipy.stats import linregress
        sl,ic,r,_,_=linregress(lD,lX)
        print(f"\nPower law fit: xi = {math.exp(ic):.3f} * DIFFUSE^{sl:.3f}  (r={r:.3f})")
        print(f"  Predicted exponent from diffusion theory: 0.5")

    print("\nAll done.")
