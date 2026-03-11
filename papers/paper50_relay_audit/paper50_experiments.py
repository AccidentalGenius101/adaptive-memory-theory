"""
Paper 50: Zone Identity vs Parity in Geographic Relay

Audit: L1 gets na_ratio=1.25 (identity encoding) from WaveEnvStd.
L2 in geo amplitude-copy relay gets na_ratio=0.83 (parity encoding),
even though it receives the same wave amplitude pattern as L1.
Why? And how do we fix it?

Three conditions x 5 seeds = 15 runs:

  ctrl_rnd:      L2 gets WaveEnvRandom (Paper 49 ctrl baseline).
                 Expect na_l2>1 (ctrl self-organises into identity via copy-forward drift).

  ctrl_std:      L2 gets its OWN WaveEnvStd (same zone structure as L1, independent seed).
                 Tests: does L2 develop identity encoding from structured waves independently?
                 Expect na_l2 > ctrl_rnd (structured waves should be better than random).
                 If na_l2 > 1 here: the amplitude-copy geo mechanism is the problem.

  geo_copy:      Paper 49 amplitude-copy geo relay (L1 waves applied to L2 directly).
                 Baseline — confirms na_l2 < 1 (parity encoding).

  geo_4class:    4-level amplitude geo relay.
                 cls=0 -> suppress amp=0.40, cls=1 -> excite amp=0.15,
                 cls=2 -> suppress amp=0.10, cls=3 -> excite amp=0.40.
                 Creates 4 distinct viability effects (not binary parity).
                 Tests: does graded amplitude restore identity encoding?

  geo_own_relay: L2 gets OWN WaveEnvStd (same zone structure) AND birth seeding from
                 L1 zone means (mean_relay). Combines structured ongoing events with
                 structured seeding. Does this achieve identity encoding AND exceed ctrl?

5 conditions x 5 seeds = 25 runs.
"""
import numpy as np, json, os, math, random, multiprocessing as mp

W=80; H=40; HALF=40
HS=2; IS=3
WAVE_DUR=15
SUPP_AMP=0.25; EXC_AMP=0.50
MID_DECAY=0.99; FIELD_DECAY=0.9997; BASE_BETA=0.005
FA=0.16
VALS_DECAY=0.92; VALS_NAV=0.08; ADJ_SCALE=0.03
BOUND_LO=0.05; BOUND_HI=0.95; FRAG_NOISE=0.01
INST_THRESH=0.45; INST_PROB=0.03; DIFFUSE=0.02; DIFFUSE_EVERY=2
SEEDS      = list(range(5))
STEPS_STD  = 3000
N_ZONES    = 4
WR         = 4.8
SEED_BETA  = 0.25

CONDITIONS = ['ctrl_rnd','ctrl_std','geo_copy','geo_4class','geo_own_relay']

RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper50_results.json")

# 4-class amplitude table: indexed by wave class (0-3)
AMP_4CLASS = {0: ('supp', 0.40), 1: ('exc', 0.15),
              2: ('supp', 0.10), 3: ('exc', 0.40)}


# ── FastVCML ──────────────────────────────────────────────────────────────────
class FastVCML:
    def __init__(self, seed, ss=10, seed_beta=0.25):
        self.rng       = np.random.RandomState(seed)
        self.SS        = ss
        self.seed_beta = seed_beta
        self._zm_source = None
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

    def set_zone_mean_source(self, l1, n_zones):
        self._zm_source = (l1, n_zones)

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
        gate=self.streak>=self.SS
        self.fieldM[gate]+=FA*(self.mid[gate]-self.fieldM[gate])
        self.fieldM*=FIELD_DECAY
        if self.t%DIFFUSE_EVERY==0: self._diffuse()
        self.age+=1
        self._collapse()
        self.t+=1

    def _collapse(self):
        rng=self.rng
        bad=(self.vals<BOUND_LO)|(self.vals>BOUND_HI)
        inst=(np.sum(np.abs(self.hid-self.base_h),1)>INST_THRESH)&(rng.random(self.N)<INST_PROB)
        ci=np.where(bad|inst)[0]
        if not len(ci): return
        prev=self.hid.copy()
        zm=None
        if self._zm_source is not None:
            l1, nz = self._zm_source
            zw = HALF // nz
            zm = self._get_zone_means(l1, nz)
        for ai in ci:
            self.cc[ai]+=1
            if zm is not None:
                zw = HALF // self._zm_source[1]
                zone_idx = min(self._zm_source[1]-1, int((self.ai_x[ai]-HALF)//zw))
                fm = zm[zone_idx]
            else:
                fm = self.fieldM[ai]
            mag=np.sqrt(np.dot(fm,fm))
            sb=self.seed_beta
            nh=((1-sb)*prev[ai]+sb*fm if mag>1e-6 else prev[ai].copy())
            nh+=rng.randn(HS)*FRAG_NOISE
            self.hid[ai]=nh; self.vals[ai]=.5; self.age[ai]=0
            self.streak[ai]=0; self.mid[ai]=np.zeros(HS)

    def _get_zone_means(self, l1, n_zones):
        zw = HALF // n_zones; means=[]
        for z in range(n_zones):
            xlo=HALF+z*zw; xhi=HALF+(z+1)*zw
            mask=(l1.ai_x>=xlo)&(l1.ai_x<xhi)
            fm=l1.fieldM[mask]
            means.append(fm.mean(0) if len(fm) else np.zeros(HS))
        return means


# ── Metrics ───────────────────────────────────────────────────────────────────
def sg4_fn(vcml, n_zones=4):
    zw=HALF//n_zones; means=[]; spreads=[]
    for z in range(n_zones):
        xlo=HALF+z*zw; xhi=HALF+(z+1)*zw
        mask=(vcml.ai_x>=xlo)&(vcml.ai_x<xhi); fm=vcml.fieldM[mask]
        if len(fm)>1: means.append(fm.mean(0)); spreads.append(float(np.std(fm)))
        else: means.append(np.zeros(HS)); spreads.append(float('nan'))
    dists=[np.linalg.norm(means[a]-means[b])
           for a in range(n_zones) for b in range(a+1,n_zones)]
    sg4=float(np.mean(dists)) if dists else 0.
    sw_vals=[s for s in spreads if not math.isnan(s)]
    sw=float(np.mean(sw_vals)) if sw_vals else float('nan')
    snr=float(sg4/sw) if sw and sw>1e-8 else float('nan')
    return sg4,sw,snr

def na_ratio_fn(vcml, n_zones=4):
    zw=HALF//n_zones; means=[]
    for z in range(n_zones):
        xlo=HALF+z*zw; xhi=HALF+(z+1)*zw
        mask=(vcml.ai_x>=xlo)&(vcml.ai_x<xhi); fm=vcml.fieldM[mask]
        means.append(fm.mean(0) if len(fm) else np.zeros(HS))
    adj=[]; nadj=[]
    for a in range(n_zones):
        for b in range(a+1,n_zones):
            d=float(np.linalg.norm(means[a]-means[b]))
            if abs(a-b)==1: adj.append(d)
            else:           nadj.append(d)
    if not adj or not nadj: return float('nan')
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(np.nanmean(nadj)/np.nanmean(adj)) if np.nanmean(adj)>1e-8 else float('nan')

def decode_acc_fn(vcml, n_zones=4):
    zw=HALF//n_zones
    true_zones=np.minimum(n_zones-1,(vcml.ai_x-HALF)//zw).astype(int)
    centroids=[]
    for z in range(n_zones):
        mask=true_zones==z; fm=vcml.fieldM[mask]
        centroids.append(fm.mean(0) if len(fm) else np.zeros(HS))
    centroids=np.array(centroids)
    dists=np.linalg.norm(vcml.fieldM[:,None,:]-centroids[None,:,:],axis=2)
    pred_zones=np.argmin(dists,axis=1)
    acc=float((pred_zones==true_zones).mean())
    chance=1.0/n_zones
    return acc, float((acc-chance)/(1.0-chance))

def mm_lda_fn(vcml, n_zones=4):
    zw=HALF//n_zones
    true_zones=np.minimum(n_zones-1,(vcml.ai_x-HALF)//zw).astype(int)
    grand_mean=vcml.mid.mean(0); between=0.0; within=0.0
    for z in range(n_zones):
        mask=true_zones==z; zm=vcml.mid[mask]
        if len(zm)<2: continue
        zm_mean=zm.mean(0)
        between+=len(zm)*float(np.sum((zm_mean-grand_mean)**2))
        within+=float(np.sum((zm-zm_mean)**2))
    if within<1e-12: return float('nan')
    return float(between/within)

def all_metrics(vcml, n_zones=4):
    sg4,sw,snr=sg4_fn(vcml,n_zones)
    na=na_ratio_fn(vcml,n_zones)
    acc,dnorm=decode_acc_fn(vcml,n_zones)
    lda=mm_lda_fn(vcml,n_zones)
    cps=float(vcml.cc.sum())/vcml.N
    return dict(sg4=sg4,sw=sw,snr=snr,na_ratio=na,
                decode_acc=acc,decode_norm=dnorm,mm_lda=lda,coll_per_site=cps)


# ── Wave environments ─────────────────────────────────────────────────────────
class WaveEnvStd:
    def __init__(self, vcml, WR, rng_seed=0, r_wave=2, n_zones=4):
        self.vcml=vcml; self.WR=WR; self.rwave=r_wave; self.nz=n_zones
        self.zw=HALF//n_zones; self.rng=random.Random(rng_seed); self.waves=[]
        N=vcml.N; self._wa=np.empty(N); self._wc=np.empty(N,int)

    def _launch(self):
        cls=self.rng.randint(0,self.nz-1)
        sx=self.rng.randint(cls*self.zw,(cls+1)*self.zw-1)
        cx=HALF+sx; cy=self.rng.randint(0,H-1)
        self.waves.append([cx,cy,WAVE_DUR,cls])

    def _apply(self):
        wa=self._wa; wc=self._wc; wa.fill(0.); wc.fill(-1)
        ax=self.vcml.ai_x; ay=self.vcml.ai_y; rw=self.rwave; surv=[]
        for wave in self.waves:
            cx,cy,rem,cls=wave
            if rem<=0: continue
            dist=np.abs(ax-cx)+np.abs(ay-cy)
            act=np.maximum(0.,1.-dist/rw); act[dist>rw]=0.
            better=act>wa; wa[better]=act[better]; wc[better]=cls
            wave[2]-=1
            if wave[2]>0: surv.append(wave)
        self.waves=surv
        for even in [True,False]:
            amp=SUPP_AMP if even else EXC_AMP
            par=(wc%2==0) if even else (wc%2==1)
            idx=np.where((wc>=0)&par&(wa>.05))[0]
            if not len(idx): continue
            sc=wa[idx]*amp
            if even: self.vcml.vals[idx]=np.maximum(0.,self.vcml.vals[idx]*(1.-sc*.5))
            else:    self.vcml.vals[idx]=np.minimum(1.,self.vcml.vals[idx]+sc*.5)
        return wa.copy(),wc.copy()

    def step(self):
        exp=self.WR/WAVE_DUR
        nl=int(exp)+(1 if self.rng.random()<exp-int(exp) else 0)
        for _ in range(nl): self._launch()
        return self._apply()


class WaveEnvRandom:
    def __init__(self, vcml, WR, rng_seed=0, r_wave=2):
        self.vcml=vcml; self.WR=WR; self.rwave=r_wave
        self.rng=random.Random(rng_seed); self.waves=[]
        N=vcml.N; self._wa=np.empty(N); self._wc=np.empty(N,int)

    def _launch(self):
        cx=self.rng.randint(HALF,HALF+HALF-1); cy=self.rng.randint(0,H-1)
        cls=self.rng.randint(0,1)
        self.waves.append([cx,cy,WAVE_DUR,cls])

    def _apply(self):
        wa=self._wa; wc=self._wc; wa.fill(0.); wc.fill(-1)
        ax=self.vcml.ai_x; ay=self.vcml.ai_y; rw=self.rwave; surv=[]
        for wave in self.waves:
            cx,cy,rem,cls=wave
            if rem<=0: continue
            dist=np.abs(ax-cx)+np.abs(ay-cy)
            act=np.maximum(0.,1.-dist/rw); act[dist>rw]=0.
            better=act>wa; wa[better]=act[better]; wc[better]=cls
            wave[2]-=1
            if wave[2]>0: surv.append(wave)
        self.waves=surv
        for even in [True,False]:
            amp=SUPP_AMP if even else EXC_AMP
            par=(wc%2==0) if even else (wc%2==1)
            idx=np.where((wc>=0)&par&(wa>.05))[0]
            if not len(idx): continue
            sc=wa[idx]*amp
            if even: self.vcml.vals[idx]=np.maximum(0.,self.vcml.vals[idx]*(1.-sc*.5))
            else:    self.vcml.vals[idx]=np.minimum(1.,self.vcml.vals[idx]+sc*.5)
        return wa.copy(),wc.copy()

    def step(self):
        exp=self.WR/WAVE_DUR
        nl=int(exp)+(1 if self.rng.random()<exp-int(exp) else 0)
        for _ in range(nl): self._launch()
        return self._apply()


# ── Geo apply helpers ─────────────────────────────────────────────────────────
def _apply_geo_copy(l2, wa, wc):
    """Standard amplitude-copy geo relay (Paper 49 style, binary supp/exc)."""
    idx_supp=np.where((wc>=0)&(wc%2==0)&(wa>.05))[0]
    idx_exc =np.where((wc>=0)&(wc%2==1)&(wa>.05))[0]
    if len(idx_supp):
        sc=wa[idx_supp]*SUPP_AMP
        l2.vals[idx_supp]=np.maximum(0.,l2.vals[idx_supp]*(1.-sc*.5))
    if len(idx_exc):
        sc=wa[idx_exc]*EXC_AMP
        l2.vals[idx_exc]=np.minimum(1.,l2.vals[idx_exc]+sc*.5)

def _apply_geo_4class(l2, wa, wc):
    """4-class amplitude geo relay: each zone class gets a unique amplitude."""
    for cls,(direction,amp) in AMP_4CLASS.items():
        idx=np.where((wc==cls)&(wa>.05))[0]
        if not len(idx): continue
        sc=wa[idx]*amp
        if direction=='supp':
            l2.vals[idx]=np.maximum(0.,l2.vals[idx]*(1.-sc*.5))
        else:
            l2.vals[idx]=np.minimum(1.,l2.vals[idx]+sc*.5)


# ── Run function ──────────────────────────────────────────────────────────────
def run(seed, condition):
    l1=FastVCML(seed=seed*3,   ss=10, seed_beta=SEED_BETA)
    l2=FastVCML(seed=seed+200, ss=10, seed_beta=SEED_BETA)

    env_l1=WaveEnvStd(l1, WR=WR, rng_seed=seed*100, r_wave=2, n_zones=N_ZONES)

    if condition=='ctrl_rnd':
        env_l2=WaveEnvRandom(l2, WR=WR, rng_seed=seed*300+11, r_wave=2)
    elif condition=='ctrl_std':
        env_l2=WaveEnvStd(l2, WR=WR, rng_seed=seed*400+7, r_wave=2, n_zones=N_ZONES)
    elif condition in ('geo_copy','geo_4class'):
        env_l2=None  # uses L1 wave events directly
    elif condition=='geo_own_relay':
        l2.set_zone_mean_source(l1, N_ZONES)
        env_l2=WaveEnvStd(l2, WR=WR, rng_seed=seed*500+13, r_wave=2, n_zones=N_ZONES)

    for t in range(STEPS_STD):
        wa1,wc1=env_l1.step()

        if condition=='ctrl_rnd' or condition=='ctrl_std' or condition=='geo_own_relay':
            wa2,wc2=env_l2.step()
            l2.step(wa2,wc2)
        elif condition=='geo_copy':
            wa2=wa1.copy(); wc2=wc1.copy()
            _apply_geo_copy(l2,wa2,wc2)
            l2.step(wa2,wc2)
        elif condition=='geo_4class':
            wa2=wa1.copy(); wc2=wc1.copy()
            _apply_geo_4class(l2,wa2,wc2)
            l2.step(wa2,wc2)

        l1.step(wa1,wc1)

    m_l1=all_metrics(l1, N_ZONES)
    m_l2=all_metrics(l2, N_ZONES)
    return {
        'exp':'main','seed':seed,'condition':condition,
        'sg4_l1':m_l1['sg4'],   'snr_l1':m_l1['snr'],
        'na_l1': m_l1['na_ratio'], 'dnorm_l1':m_l1['decode_norm'],
        'sg4_l2':m_l2['sg4'],   'snr_l2':m_l2['snr'],
        'na_l2': m_l2['na_ratio'], 'dnorm_l2':m_l2['decode_norm'],
        'mm_lda_l2':m_l2['mm_lda'], 'coll_l2':m_l2['coll_per_site'],
    }


# ── Parallel dispatch ─────────────────────────────────────────────────────────
def _worker(args):
    return run(*args)

def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f: results=json.load(f)
    else:
        results=[]

    done_keys=set((r['seed'],r['condition']) for r in results)
    all_args=[(seed,cond) for cond in CONDITIONS for seed in SEEDS
              if (seed,cond) not in done_keys]

    print(f"Tasks remaining: {len(all_args)} / {len(CONDITIONS)*len(SEEDS)}")
    if all_args:
        with mp.Pool(processes=min(len(all_args),mp.cpu_count())) as pool:
            new=pool.map(_worker,all_args)
        results.extend(new)
        with open(RESULTS_FILE,'w') as f: json.dump(results,f,indent=2)
        print(f"Saved {len(results)} total results.")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print(f"\n{'condition':>16} {'sg4_L1':>8} {'na_L1':>8} {'sg4_L2':>8} {'snr_L2':>8} "
          f"{'na_L2':>8} {'dnorm_L2':>10} {'gain_vs_ctrl':>12}")
    ctrl_rnd_sg4=float(np.nanmean([r['sg4_l2'] for r in results if r['condition']=='ctrl_rnd']))
    for cond in CONDITIONS:
        rows=[r for r in results if r['condition']==cond]
        if not rows: continue
        sg4_l1=np.nanmean([r['sg4_l1'] for r in rows])
        na_l1 =np.nanmean([r['na_l1']  for r in rows])
        sg4_l2=np.nanmean([r['sg4_l2'] for r in rows])
        snr_l2=np.nanmean([r['snr_l2'] for r in rows])
        na_l2 =np.nanmean([r['na_l2']  for r in rows])
        dn_l2 =np.nanmean([r['dnorm_l2']for r in rows])
        gain  =sg4_l2/ctrl_rnd_sg4 if ctrl_rnd_sg4>1e-6 else float('nan')
        enc   = 'IDENTITY' if na_l2>1 else 'parity'
        print(f"{cond:>16} {sg4_l1:8.4f} {na_l1:8.4f} {sg4_l2:8.4f} {snr_l2:8.4f} "
              f"{na_l2:8.4f} {dn_l2:10.4f} {gain:12.3f}  [{enc}]")

    summary={}
    for cond in CONDITIONS:
        rows=[r for r in results if r['condition']==cond]
        if rows:
            summary[cond]={
                'sg4_l1':  float(np.nanmean([r['sg4_l1']    for r in rows])),
                'na_l1':   float(np.nanmean([r['na_l1']     for r in rows])),
                'sg4_l2':  float(np.nanmean([r['sg4_l2']    for r in rows])),
                'snr_l2':  float(np.nanmean([r['snr_l2']    for r in rows])),
                'na_l2':   float(np.nanmean([r['na_l2']     for r in rows])),
                'dnorm_l2':float(np.nanmean([r['dnorm_l2']  for r in rows])),
                'mm_lda_l2':float(np.nanmean([r['mm_lda_l2']for r in rows])),
                'coll_l2': float(np.nanmean([r['coll_l2']   for r in rows])),
            }
    an_file=RESULTS_FILE.replace('results.json','analysis.json')
    with open(an_file,'w') as f: json.dump(summary,f,indent=2)
    print(f"\nAnalysis saved to {an_file}")

if __name__=='__main__':
    mp.freeze_support()
    main()
