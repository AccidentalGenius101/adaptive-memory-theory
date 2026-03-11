"""
Paper 57: Analytic Closure of the VCML Lyapunov Condition.

Experiment: Fine r_wave sweep near the predicted critical threshold delta_c = 2.

Theory predicts: Lyapunov condition holds iff delta = W_zone/r_wave > 2.
- Interior fraction f_int(delta) = (1 - 2/delta)^+ = fraction of cells carrying
  unambiguous single-zone signal.
- At delta=2: f_int=0, all cells are boundary-contaminated, sigma_w inflates.
- Critical threshold is geometric, not a parameter choice.

W_zone = 20 (S2 grid: 4 zones x 20 cols in HALF=80).
r_wave sweep: {2, 3, 4, 5, 6, 7, 8, 10} -> delta = {10, 6.7, 5, 4, 3.3, 2.9, 2.5, 2}

Predictions:
  delta > 2: sigma_w small, na_ratio > 1, C_order > 0
  delta = 2: sigma_w inflates, transition
  delta < 2: sigma_w ~ C_ref level, identity collapses

Also compute theoretical interior fraction f_int for each r_wave to verify the
noise floor follows the geometric prediction.

Total: 8 conditions x 5 seeds = 40 runs x 3000 steps.
"""
import numpy as np, json, os, math, random, multiprocessing as mp

# ── Constants (mirror paper56) ─────────────────────────────────────────────────
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
SS          = 10

S2 = dict(W=160, H=80, HALF=80, WR=19.2)
W_ZONE = S2['HALF'] // N_ZONES   # = 20

# r_wave values near the predicted critical delta_c = 2 (W_zone/r_wave = 2 at r_wave=10)
RWAVE_LEVELS = [2, 3, 4, 5, 6, 7, 8, 10]

RESULTS_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results', 'paper57_results.json')
ANALYSIS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results', 'paper57_analysis.json')


# ── Theoretical predictions ────────────────────────────────────────────────────
def f_interior(r_wave, w_zone=W_ZONE):
    """Interior fraction: (1 - 2*r_wave/w_zone)^+"""
    return max(0.0, 1.0 - 2*r_wave/w_zone)

def delta(r_wave, w_zone=W_ZONE):
    return w_zone / r_wave


# ── FastVCML C_perturb (copied from paper56) ───────────────────────────────────
class FastVCMLCP:
    def __init__(self, seed, ss=10, seed_beta=0.25, half=80, h=80):
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
        wg=(self.streak>=self.SS)&(wa>0.05)
        if wg.any(): self.fieldM[wg]+=FA*(self.mid[wg]-self.fieldM[wg])
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


# ── Metrics ────────────────────────────────────────────────────────────────────
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
    na      = float(D_nonadj/D_adj)         if D_adj>1e-8  else float('nan')
    c_order = float((D_nonadj-D_adj)/sigma_w) if sigma_w>1e-5 else float('nan')
    coll_site = float(vcml.cc.sum())/(vcml.N*vcml.t) if vcml.t>0 else 0.
    return dict(D_adj=D_adj, D_nonadj=D_nonadj, sg4=sg4, sigma_w=sigma_w,
                na=na, c_order=c_order, coll_site=coll_site)


def run_one(seed, r_wave):
    vcml = FastVCMLCP(seed, ss=SS, half=S2['HALF'], h=S2['H'])
    env  = WaveEnv(vcml, S2['WR'], supp_amp=0.06, exc_amp=0.12,
                   r_wave=r_wave, rng_seed=seed*1000)
    traj = []
    for t in range(STEPS):
        wa, wc = env.step(); vcml.step(wa, wc)
        if (t+1) % TRACK_EVERY == 0:
            s = compute_snapshot(vcml); s['step'] = t+1
            traj.append(s)
    return traj


def _worker(args):
    seed, r_wave = args
    return (seed, r_wave, run_one(seed, r_wave))


def mean_s(vals):
    v=[x for x in vals if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return float(np.mean(v)) if v else float('nan')

def se_s(vals):
    v=[x for x in vals if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return float(np.std(v)/math.sqrt(len(v))) if len(v)>1 else 0.


def aggregate(trajs):
    n_tp = min(len(t) for t in trajs)
    steps = [trajs[0][i]['step'] for i in range(n_tp)]
    keys = ['na','c_order','sigma_w','sg4','coll_site']
    out = {'steps': steps}
    for k in keys:
        out[f'{k}_mean'] = [mean_s([t[i][k] for t in trajs]) for i in range(n_tp)]
        out[f'{k}_se']   = [se_s( [t[i][k] for t in trajs]) for i in range(n_tp)]
    out['final_na']      = mean_s([t[-1]['na']      for t in trajs])
    out['final_c_order'] = mean_s([t[-1]['c_order'] for t in trajs])
    out['final_sigma_w'] = mean_s([t[-1]['sigma_w'] for t in trajs])
    # cross_step: first stable positive C_order (5 consecutive positives)
    co = out['c_order_mean']
    cross_step = float('nan')
    for i in range(n_tp-5):
        if all(not math.isnan(co[i+j]) and co[i+j]>0 for j in range(5)):
            cross_step = float(steps[i]); break
    out['cross_step'] = cross_step
    return out


if __name__ == '__main__':
    mp.freeze_support()

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    # Load or run
    if os.path.exists(RESULTS_FILE):
        print(f'Loading existing results from {RESULTS_FILE}')
        with open(RESULTS_FILE) as f:
            raw = json.load(f)
    else:
        all_args = [(seed, rw) for rw in RWAVE_LEVELS for seed in SEEDS]
        print(f'Running {len(all_args)} experiments...')
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            results = pool.map(_worker, all_args)
        raw = {}
        for seed, r_wave, traj in results:
            k = str(r_wave)
            if k not in raw: raw[k] = {}
            raw[k][str(seed)] = traj
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print(f'Saved to {RESULTS_FILE}')

    # Aggregate
    summary = {}
    print(f'\nW_zone={W_ZONE} (S2, 4 zones x {W_ZONE} cols)')
    print(f'\nTheoretical predictions (delta_c=2, f_int=(1-2/delta)^+):')
    print(f'{"r_wave":>8} {"delta":>6} {"f_int":>7} {"predicted":>12}')
    print('-'*40)
    for rw in RWAVE_LEVELS:
        d = delta(rw); fi = f_interior(rw)
        pred = 'identity' if d > 2 else ('threshold' if d == 2 else 'failure')
        print(f'{rw:>8} {d:>6.2f} {fi:>7.3f} {pred:>12}')

    print(f'\nObserved results at step {STEPS}:')
    print(f'{"r_wave":>8} {"delta":>6} {"f_int_pred":>11} {"final_na":>10} {"C_order":>9} {"sigma_w":>9} {"cross_step":>12}')
    print('-'*72)
    for rw in RWAVE_LEVELS:
        k = str(rw)
        if k not in raw: continue
        trajs = list(raw[k].values())
        agg = aggregate(trajs)
        summary[f'rw{rw}'] = agg
        d = delta(rw); fi = f_interior(rw)
        cs = agg['cross_step']
        cs_str = f'{int(cs)}' if not math.isnan(cs) else 'never'
        print(f'{rw:>8} {d:>6.2f} {fi:>11.3f} {agg["final_na"]:>10.3f} {agg["final_c_order"]:>9.3f} {agg["final_sigma_w"]:>9.4f} {cs_str:>12}')

    # Cross-condition summary for paper
    print('\nKey ratio test: sigma_w vs f_int prediction')
    print(f'{"r_wave":>8} {"f_int":>7} {"sigma_w":>9} {"sigma_w*delta/2":>16}')
    print('-'*44)
    for rw in RWAVE_LEVELS:
        k = f'rw{rw}'
        if k not in summary: continue
        sw = summary[k]['final_sigma_w']
        fi = f_interior(rw); d = delta(rw)
        norm = sw*(d/2) if d>0 else float('nan')
        print(f'{rw:>8} {fi:>7.3f} {sw:>9.4f} {norm:>16.4f}')

    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved analysis to {ANALYSIS_FILE}')
