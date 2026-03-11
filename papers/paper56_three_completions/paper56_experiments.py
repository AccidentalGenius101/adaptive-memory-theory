"""
Paper 56: Three Completions for the VCML Regime Diagram.

Exp A — SS threshold sweep in C_ref (noise-floor control)
  Fix: sub-threshold C_ref (supp=0.06, exc=0.12, r_wave=2, S1).
  Sweep SS: 10, 20, 30, 50, 100, 200 x 5 seeds = 30 runs.
  Question: does high SS (fewer writes) flip C_order from negative to positive?
  Mechanism test: high SS -> low write_ready_frac -> low sigma_w -> C_order > 0.

Exp B — r_wave x C_perturb at S2 (K_self measurement)
  Fix: C_perturb, sub-threshold amplitude (supp=0.06, exc=0.12, S2).
  Sweep r_wave: 1, 2, 4, 8 x 5 seeds = 20 runs.
  Question: does formation rate scale linearly with r_wave? (K_self ∝ r_wave)
  Also: does C_perturb push the effective correlation-length threshold from 5 to 10?

Exp C — Analysis only (no new experiments)
  Use Exp A data: compute write_ready_frac and scatter vs na_ratio, C_order.
  Show na_ratio is write-frequency-invariant but C_order is not.
  Conclusion: na_ratio - 1 is the robust unified order parameter.

Total: 50 runs x 3000 steps each, tracked every 25 steps (120 timepoints/run).
"""

import numpy as np, json, os, math, random, multiprocessing as mp

# ── Fixed constants ────────────────────────────────────────────────────────────
HS = 2; IS = 3
WAVE_DUR   = 15
MID_DECAY  = 0.99; FIELD_DECAY = 0.9997; BASE_BETA = 0.005
FA         = 0.16
VALS_DECAY = 0.92; VALS_NAV = 0.08; ADJ_SCALE = 0.03
BOUND_LO = 0.05; BOUND_HI = 0.95; FRAG_NOISE = 0.01
INST_THRESH = 0.45; INST_PROB = 0.03; DIFFUSE = 0.02; DIFFUSE_EVERY = 2

SEEDS       = list(range(5))
STEPS       = 3000
TRACK_EVERY = 25
N_ZONES     = 4

S1 = dict(W=80,  H=40,  HALF=40,  WR=4.8)
S2 = dict(W=160, H=80,  HALF=80,  WR=19.2)

# Exp A: SS sweep (C_ref, sub-threshold, S1)
SS_LEVELS = [10, 20, 30, 50, 100, 200]

# Exp B: r_wave sweep (C_perturb, sub-threshold, S2)
RWAVE_LEVELS_B = [1, 2, 4, 8]

RESULTS_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results', 'paper56_results.json')
ANALYSIS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results', 'paper56_analysis.json')


# ── FastVCML (C_ref) ───────────────────────────────────────────────────────────
class FastVCML:
    def __init__(self, seed, ss=10, seed_beta=0.25, half=40, h=40):
        self.rng       = np.random.RandomState(seed)
        self.SS        = ss
        self.seed_beta = seed_beta
        self.HALF      = half; self.H = h
        N = half * h
        self.N = N; self.t = 0
        self.ai_x = half + (np.arange(N) % half)
        self.ai_y = np.arange(N) // half
        rng = self.rng; sc = 0.3
        self.Wz=rng.randn(N,HS,IS)*sc; self.Uz=rng.randn(N,HS,HS)*sc; self.bz=np.full((N,HS),.5)
        self.Wr=rng.randn(N,HS,IS)*sc; self.Ur=rng.randn(N,HS,HS)*sc; self.br=np.full((N,HS),.5)
        self.Wh=rng.randn(N,HS,IS)*sc; self.Uh=rng.randn(N,HS,HS)*sc; self.bh=np.zeros((N,HS))
        self.Wo=rng.randn(N,HS)*0.1;   self.bo=np.zeros(N)
        self.vals  = rng.uniform(.3,.7,N)
        self.hid   = rng.randn(N,HS)*.1
        self.age   = rng.randint(0,50,N).astype(float)
        self.base_h = np.zeros((N,HS))
        self.mid    = np.zeros((N,HS))
        self.fieldM = np.zeros((N,HS))
        self.streak = np.zeros(N,int)
        self.cc     = np.zeros(N,int)
        self._build_nb()

    def _build_nb(self):
        N=self.N; HALF=self.HALF; H=self.H
        rx=np.arange(N)%HALF; y=np.arange(N)//HALF
        nb=np.full((N,4),-1,int)
        nb[rx>0,0]=np.where(rx>0)[0]-1; nb[rx<HALF-1,1]=np.where(rx<HALF-1)[0]+1
        nb[y>0,2]=np.where(y>0)[0]-HALF; nb[y<H-1,3]=np.where(y<H-1)[0]+HALF
        self.nb=nb; self.nbc=(nb>=0).sum(1).astype(float)

    def _nbmean(self):
        ns=np.maximum(self.nb,0); g=self.vals[ns]; g[self.nb<0]=0.
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
        hn=(1-z)*h+z*g
        out=np.tanh(np.einsum('ni,ni->n',self.Wo,hn)+self.bo)
        return hn,out

    def _diffuse(self):
        ns=np.maximum(self.nb,0); nf=self.fieldM[ns]; nf*=(self.nb>=0)[:,:,None]
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
        self.age+=1; self._collapse(); self.t+=1

    def _collapse(self):
        rng=self.rng
        bad=(self.vals<BOUND_LO)|(self.vals>BOUND_HI)
        inst=(np.sum(np.abs(self.hid-self.base_h),1)>INST_THRESH)&(rng.random(self.N)<INST_PROB)
        ci=np.where(bad|inst)[0]
        if not len(ci): return
        prev=self.hid.copy()
        for ai in ci:
            self.cc[ai]+=1
            fm=self.fieldM[ai]; mag=np.sqrt(np.dot(fm,fm)); sb=self.seed_beta
            nh=((1-sb)*prev[ai]+sb*fm if mag>1e-6 else prev[ai].copy())
            nh+=rng.randn(HS)*FRAG_NOISE
            self.hid[ai]=nh; self.vals[ai]=.5; self.age[ai]=0
            self.streak[ai]=0; self.mid[ai]=np.zeros(HS)


# ── FastVCML C_perturb ─────────────────────────────────────────────────────────
class FastVCMLCP(FastVCML):
    def step(self,wa,wc):
        nb=self._nbmean(); an=np.minimum(1.,self.age/300.)
        x=np.stack([nb,np.minimum(1.,wa),an],1)
        hn,out=self._gru(x); self.hid=hn
        self.vals=np.clip(VALS_DECAY*self.vals+VALS_NAV*nb+ADJ_SCALE*out,0,1)
        dev=hn-self.base_h; self.base_h+=BASE_BETA*dev
        self.streak=np.where(np.sum(dev**2,1)<.0025,self.streak+1,0)
        self.mid=(self.mid+FA*dev)*MID_DECAY
        wg=(self.streak>=self.SS)&(wa>0.05)
        if wg.any(): self.fieldM[wg]+=FA*(self.mid[wg]-self.fieldM[wg])
        self.fieldM*=FIELD_DECAY
        if self.t%DIFFUSE_EVERY==0: self._diffuse()
        self.age+=1; self._collapse(); self.t+=1


# ── Wave environment ────────────────────────────────────────────────────────────
class WaveEnv:
    def __init__(self,vcml,WR,supp_amp=0.25,exc_amp=0.50,r_wave=2,n_zones=4,rng_seed=0):
        self.vcml=vcml; self.WR=WR; self.supp_amp=supp_amp; self.exc_amp=exc_amp
        self.rwave=r_wave; self.nz=n_zones; self.zw=vcml.HALF//n_zones
        self.rng=random.Random(rng_seed); self.waves=[]
        N=vcml.N; self._wa=np.empty(N); self._wc=np.empty(N,int)

    def _launch(self):
        cls=self.rng.randint(0,self.nz-1)
        sx=self.rng.randint(cls*self.zw,(cls+1)*self.zw-1)
        cx=self.vcml.HALF+sx; cy=self.rng.randint(0,self.vcml.H-1)
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
            amp=self.supp_amp if even else self.exc_amp
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


# ── Snapshot metrics (includes write_ready_frac) ───────────────────────────────
def compute_snapshot(vcml):
    HALF=vcml.HALF; nz=N_ZONES; zw=HALF//nz
    means=[]; spreads=[]
    for z in range(nz):
        xlo=HALF+z*zw; xhi=HALF+(z+1)*zw
        mask=(vcml.ai_x>=xlo)&(vcml.ai_x<xhi)
        fm=vcml.fieldM[mask]
        means.append(fm.mean(0) if len(fm) else np.zeros(HS))
        spreads.append(float(np.std(fm)) if len(fm)>1 else 0.)
    adj=[]; nadj=[]
    for a in range(nz):
        for b in range(a+1,nz):
            d=float(np.linalg.norm(means[a]-means[b]))
            (adj if abs(a-b)==1 else nadj).append(d)
    D_adj   = float(np.mean(adj))   if adj  else 0.
    D_nonadj= float(np.mean(nadj))  if nadj else 0.
    sg4     = float(np.mean(adj+nadj)) if (adj or nadj) else 0.
    sigma_w = float(np.mean(spreads)) if spreads else 0.
    na      = float(D_nonadj/D_adj) if D_adj>1e-8 else float('nan')
    c_order = float((D_nonadj-D_adj)/sigma_w) if sigma_w>1e-5 else float('nan')
    # fraction of cells passing the write streak gate right now
    write_ready = float((vcml.streak>=vcml.SS).mean())
    coll_site   = float(vcml.cc.sum())/(vcml.N*vcml.t) if vcml.t>0 else 0.
    return dict(D_adj=D_adj,D_nonadj=D_nonadj,sg4=sg4,sigma_w=sigma_w,
                na=na,c_order=c_order,write_ready=write_ready,coll_site=coll_site)


# ── Run functions ──────────────────────────────────────────────────────────────
def run_exp_a(seed, ss):
    """Exp A: C_ref, sub-threshold, S1, variable SS."""
    vcml = FastVCML(seed, ss=ss, half=S1['HALF'], h=S1['H'])
    env  = WaveEnv(vcml, S1['WR'], supp_amp=0.06, exc_amp=0.12,
                   r_wave=2, rng_seed=seed*1000)
    traj = []
    for t in range(STEPS):
        wa,wc = env.step(); vcml.step(wa,wc)
        if (t+1) % TRACK_EVERY == 0:
            s = compute_snapshot(vcml); s['step'] = t+1
            traj.append(s)
    return traj


def run_exp_b(seed, r_wave):
    """Exp B: C_perturb, sub-threshold, S2, variable r_wave."""
    vcml = FastVCMLCP(seed, ss=10, half=S2['HALF'], h=S2['H'])
    env  = WaveEnv(vcml, S2['WR'], supp_amp=0.06, exc_amp=0.12,
                   r_wave=r_wave, rng_seed=seed*1000)
    traj = []
    for t in range(STEPS):
        wa,wc = env.step(); vcml.step(wa,wc)
        if (t+1) % TRACK_EVERY == 0:
            s = compute_snapshot(vcml); s['step'] = t+1
            traj.append(s)
    return traj


# ── Worker ─────────────────────────────────────────────────────────────────────
def _worker(args):
    exp, seed, param = args
    if exp == 'A':
        return ('A', seed, param, run_exp_a(seed, param))
    else:
        return ('B', seed, param, run_exp_b(seed, param))


# ── Aggregation ────────────────────────────────────────────────────────────────
def mean_safe(vals):
    v = [x for x in vals if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return float(np.mean(v)) if v else float('nan')

def se_safe(vals):
    v = [x for x in vals if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return float(np.std(v)/math.sqrt(len(v))) if len(v)>1 else 0.


def aggregate_group(trajs):
    """Aggregate a list of seed trajectories into mean/se time series + summary."""
    n_tp = min(len(t) for t in trajs)
    steps = [trajs[0][tp]['step'] for tp in range(n_tp)]
    keys_to_track = ['c_order','na','sg4','sigma_w','write_ready','coll_site']
    series = {}
    for k in keys_to_track:
        series[f'{k}_mean'] = [mean_safe([t[tp][k] for t in trajs]) for tp in range(n_tp)]
        series[f'{k}_se']   = [se_safe( [t[tp][k] for t in trajs]) for tp in range(n_tp)]

    # Drift in three windows
    co = series['c_order_mean']
    thirds = n_tp // 3
    drifts = []
    for lo,hi in [(0,thirds),(thirds,2*thirds),(2*thirds,n_tp)]:
        seg = [x for x in co[lo:hi] if not math.isnan(x)]
        drifts.append(float((seg[-1]-seg[0])/((hi-lo)*TRACK_EVERY)) if len(seg)>=2 else float('nan'))

    # First step where C_order > 0 and stays positive for 5 consecutive points
    cross_step = float('nan')
    for i in range(n_tp-5):
        if all(not math.isnan(co[i+j]) and co[i+j]>0 for j in range(5)):
            cross_step = float(steps[i]); break

    final_na  = series['na_mean'][-1]
    final_co  = series['c_order_mean'][-1]
    mean_wr   = mean_safe(series['write_ready_mean'])
    peak_co   = max((x for x in co if not math.isnan(x)), default=float('nan'))

    return dict(steps=steps, drift_windows=drifts,
                final_na=final_na, final_c_order=final_co,
                peak_c_order=peak_co, cross_step=cross_step,
                mean_write_ready=mean_wr,
                **series)


def aggregate(done):
    summary = {'A': {}, 'B': {}}

    for ss in SS_LEVELS:
        trajs = [done[f'A|{seed}|{ss}'] for seed in SEEDS if f'A|{seed}|{ss}' in done]
        if trajs:
            summary['A'][f'SS{ss}'] = aggregate_group(trajs)
            summary['A'][f'SS{ss}']['ss'] = ss

    for rw in RWAVE_LEVELS_B:
        trajs = [done[f'B|{seed}|{rw}'] for seed in SEEDS if f'B|{seed}|{rw}' in done]
        if trajs:
            summary['B'][f'rw{rw}'] = aggregate_group(trajs)
            summary['B'][f'rw{rw}']['r_wave'] = rw

    # Exp C: correlation between write_ready and identity metrics across all conditions
    rows_c = []
    for ss in SS_LEVELS:
        k = f'SS{ss}'
        if k in summary['A']:
            s = summary['A'][k]
            rows_c.append(dict(label=k, ss=ss, r_wave=None,
                               final_na=s['final_na'], final_c_order=s['final_c_order'],
                               mean_write_ready=s['mean_write_ready'],
                               peak_c_order=s['peak_c_order']))
    for rw in RWAVE_LEVELS_B:
        k = f'rw{rw}'
        if k in summary['B']:
            s = summary['B'][k]
            rows_c.append(dict(label=k, ss=None, r_wave=rw,
                               final_na=s['final_na'], final_c_order=s['final_c_order'],
                               mean_write_ready=s['mean_write_ready'],
                               peak_c_order=s['peak_c_order']))
    summary['C'] = rows_c
    return summary


def main():
    mp.freeze_support()

    done = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            done = json.load(f)
        print(f"Loaded {len(done)} existing results")

    all_tasks = []
    for ss in SS_LEVELS:
        for seed in SEEDS:
            k = f'A|{seed}|{ss}'
            if k not in done:
                all_tasks.append(('A', seed, ss))
    for rw in RWAVE_LEVELS_B:
        for seed in SEEDS:
            k = f'B|{seed}|{rw}'
            if k not in done:
                all_tasks.append(('B', seed, rw))

    print(f"Tasks to run: {len(all_tasks)}")
    if all_tasks:
        n_proc = min(len(all_tasks), mp.cpu_count())
        with mp.Pool(processes=n_proc) as pool:
            for res in pool.imap_unordered(_worker, all_tasks):
                exp, seed, param, traj = res
                if exp == 'A':
                    k = f'A|{seed}|{param}'
                else:
                    k = f'B|{seed}|{param}'
                done[k] = traj
                final = traj[-1] if traj else {}
                co = final.get('c_order', float('nan'))
                na = final.get('na',      float('nan'))
                wr = final.get('write_ready', float('nan'))
                print(f"  {k}: c_order={co:.3f} na={na:.3f} wr={wr:.3f}")
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(done, f, indent=2)

    summary = aggregate(done)
    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {ANALYSIS_FILE}")

    # ── Print summary tables ──────────────────────────────────────────────────
    print("\n=== Exp A: SS sweep (C_ref sub-threshold) ===")
    print(f"  {'SS':>6} {'final_co':>10} {'final_na':>10} {'peak_co':>10} "
          f"{'mean_wr':>10} {'cross_step':>12}")
    for ss in SS_LEVELS:
        k = f'SS{ss}'
        if k in summary['A']:
            s = summary['A'][k]
            def f(x): return f'{x:.4f}' if not math.isnan(x) else '  nan '
            cs = f"{s['cross_step']:.0f}" if not math.isnan(s['cross_step']) else 'never'
            print(f"  {ss:>6} {f(s['final_c_order']):>10} {f(s['final_na']):>10} "
                  f"{f(s['peak_c_order']):>10} {f(s['mean_write_ready']):>10} {cs:>12}")

    print("\n=== Exp B: r_wave sweep (C_perturb sub-threshold S2) ===")
    print(f"  {'r_wave':>8} {'delta':>8} {'final_co':>10} {'final_na':>10} "
          f"{'peak_co':>10} {'cross_step':>12}")
    for rw in RWAVE_LEVELS_B:
        k = f'rw{rw}'
        if k in summary['B']:
            s = summary['B'][k]
            def f(x): return f'{x:.4f}' if not math.isnan(x) else '  nan '
            delta = S2['HALF'] // N_ZONES / rw
            cs = f"{s['cross_step']:.0f}" if not math.isnan(s['cross_step']) else 'never'
            print(f"  {rw:>8} {delta:>8.1f} {f(s['final_c_order']):>10} "
                  f"{f(s['final_na']):>10} {f(s['peak_c_order']):>10} {cs:>12}")

    print("\n=== Exp C: write_ready vs identity (all conditions) ===")
    print(f"  {'label':>10} {'mean_wr':>10} {'final_na':>10} "
          f"{'final_co':>10} {'identity?':>12}")
    for row in summary['C']:
        def f(x): return f'{x:.4f}' if x is not None and not math.isnan(x) else '  nan '
        ident = 'YES' if row['final_na'] > 1.0 else 'no'
        print(f"  {row['label']:>10} {f(row['mean_write_ready']):>10} "
              f"{f(row['final_na']):>10} {f(row['final_c_order']):>10} {ident:>12}")


if __name__ == '__main__':
    main()
