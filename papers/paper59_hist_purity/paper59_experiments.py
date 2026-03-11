"""
Paper 59: Historical Causal Purity and the Three-Dimensional Identity Condition.

Paper 58 showed that spatial purity P_sp AND temporal purity P_temp are both required.
But C_perturb at r_wave=8 has P_sp=0.905 and P_temp=1, yet fails (na=0.943).
Neither purity dimension predicts this failure.

The hypothesis: a THIRD dimension is missing.

P_hist = historical purity of mid_mem at consolidation time.
        = fraction of the accumulated mid_mem content (at the moment of consolidation)
          that came from same-zone wave contacts.

Unlike P_sp (which measures the TRIGGERING wave only), P_hist integrates
ALL prior wave contributions exponentially weighted by MID_DECAY^(t-tau).

Even when C_perturb writes only during correct-zone waves (P_sp~1, P_temp=1),
mid_mem has been accumulating contamination from prior cross-zone waves.
At r_wave=8, 80% of zone cells are boundary-adjacent, receiving frequent
cross-zone waves that pollute mid_mem before the consolidation fires.

Measurement: for each cell, maintain:
  midpure[i] = EW-sum of (FA*|dev|) contributions during same-zone wave contact
  midtotal[i] = EW-sum of (FA*|dev|) contributions during ANY wave contact
  P_hist = mean(midpure/midtotal) at consolidation events

Expected:
  P_hist < P_sp at all r_wave (history integrates more contamination than a single event)
  P_hist(r=8) < p_c_hist -> explains C_perturb failure
  P_hist(r=10) > P_hist(r=8) due to launch asymmetry (same mechanism as Paper 57)
"""
import numpy as np, json, os, math, random, multiprocessing as mp

# ── Constants ──────────────────────────────────────────────────────────────────
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
ZONE_WIDTH  = S2['HALF'] // N_ZONES   # 20

RWAVE_LEVELS = [2, 4, 8, 10]

RESULTS_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results', 'paper59_results.json')
ANALYSIS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results', 'paper59_analysis.json')


# ── Base VCML with P_hist tracking ────────────────────────────────────────────
class FastVCML:
    """C_ref base, with P_sp and P_hist tracking."""
    def __init__(self, seed, ss=10, seed_beta=0.25, half=80, h=80):
        self.rng       = np.random.RandomState(seed)
        self.SS        = ss
        self.seed_beta = seed_beta
        self.HALF      = half; self.H = h
        N = half * h
        self.N = N; self.t = 0
        self.ai_x = half + (np.arange(N) % half)
        self.ai_y = np.arange(N) // half
        self.cell_zone = (self.ai_x - half) // ZONE_WIDTH
        rng = self.rng; sc = 0.3
        self.Wz=rng.randn(N,HS,IS)*sc; self.Uz=rng.randn(N,HS,HS)*sc; self.bz=np.full((N,HS),.5)
        self.Wr=rng.randn(N,HS,IS)*sc; self.Ur=rng.randn(N,HS,HS)*sc; self.br=np.full((N,HS),.5)
        self.Wh=rng.randn(N,HS,IS)*sc; self.Uh=rng.randn(N,HS,HS)*sc; self.bh=np.zeros((N,HS))
        self.Wo=rng.randn(N,HS)*0.1;   self.bo=np.zeros(N)
        self.vals   = rng.uniform(.3,.7,N)
        self.hid    = rng.randn(N,HS)*.1
        self.age    = rng.randint(0,50,N).astype(float)
        self.base_h = np.zeros((N,HS))
        self.mid    = np.zeros((N,HS))
        self.fieldM = np.zeros((N,HS))
        self.streak = np.zeros(N,int)
        self.cc     = np.zeros(N,int)
        # P_sp accumulators
        self.n_writes = 0; self.n_same = 0
        self.last_wc  = np.full(N, -1, int)
        # P_hist accumulators
        self.midpure  = np.zeros(N)   # EW same-zone contribution magnitude
        self.midtotal = np.zeros(N)   # EW total wave contribution magnitude
        self.n_phist_writes = 0
        self.n_phist_same   = 0.0     # sum of p_hist values (float)
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

    def _update_phist(self, wa, wc, dev):
        """Update historical purity accumulators."""
        active = wc >= 0
        new_contrib = FA * np.sqrt(np.sum(dev**2, 1))  # magnitude of mid_mem update
        wave_contrib = np.where(active, new_contrib, 0.)
        same_wave = active & (wc == self.cell_zone)
        # exponential decay
        self.midpure  *= MID_DECAY
        self.midtotal *= MID_DECAY
        # add wave-gated contributions
        self.midpure  += np.where(same_wave, wave_contrib, 0.)
        self.midtotal += wave_contrib

    def _record_phist(self, gate_indices):
        """Record P_hist for consolidating cells."""
        total = self.midtotal[gate_indices]
        valid = total > 1e-8
        if valid.any():
            ph = np.where(valid, self.midpure[gate_indices] / total, 0.)
            self.n_phist_writes += int(valid.sum())
            self.n_phist_same   += float(ph[valid].sum())

    def step(self, wa, wc):
        nb=self._nbmean(); an=np.minimum(1.,self.age/300.)
        x=np.stack([nb,np.minimum(1.,wa),an],1)
        hn,out=self._gru(x); self.hid=hn
        self.vals=np.clip(VALS_DECAY*self.vals+VALS_NAV*nb+ADJ_SCALE*out,0,1)
        dev=hn-self.base_h; self.base_h+=BASE_BETA*dev
        self.streak=np.where(np.sum(dev**2,1)<.0025,self.streak+1,0)
        # update P_hist before mid_mem changes this step
        self._update_phist(wa, wc, dev)
        self.mid=(self.mid+FA*dev)*MID_DECAY
        # update P_sp last-wc proxy
        active = wc >= 0
        self.last_wc[active] = wc[active]
        # C_ref write gate
        gate = self.streak >= self.SS
        if gate.any():
            wi = np.where(gate)[0]
            lw = self.last_wc[wi]; cz = self.cell_zone[wi]
            valid = lw >= 0
            self.n_writes += int(valid.sum())
            self.n_same   += int(((lw == cz) & valid).sum())
            self._record_phist(wi)
            self.fieldM[gate] += FA*(self.mid[gate]-self.fieldM[gate])
        self.fieldM *= FIELD_DECAY
        if self.t % DIFFUSE_EVERY == 0: self._diffuse()
        self.age += 1; self._collapse(); self.t += 1

    def p_causal(self):
        return float(self.n_same/self.n_writes) if self.n_writes > 0 else float('nan')

    def p_hist(self):
        return float(self.n_phist_same/self.n_phist_writes) if self.n_phist_writes > 0 else float('nan')

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


class FastVCMLCP(FastVCML):
    """C_perturb: writes only during active wave. P_sp and P_hist both tracked."""
    def step(self, wa, wc):
        nb=self._nbmean(); an=np.minimum(1.,self.age/300.)
        x=np.stack([nb,np.minimum(1.,wa),an],1)
        hn,out=self._gru(x); self.hid=hn
        self.vals=np.clip(VALS_DECAY*self.vals+VALS_NAV*nb+ADJ_SCALE*out,0,1)
        dev=hn-self.base_h; self.base_h+=BASE_BETA*dev
        self.streak=np.where(np.sum(dev**2,1)<.0025,self.streak+1,0)
        # update P_hist before mid_mem
        self._update_phist(wa, wc, dev)
        self.mid=(self.mid+FA*dev)*MID_DECAY
        # C_perturb gate: streak AND active wave
        wg = (self.streak >= self.SS) & (wa > 0.05)
        if wg.any():
            wi = np.where(wg)[0]
            cz = self.cell_zone[wi]
            wz = wc[wi]   # exact triggering zone
            self.n_writes += len(wi)
            self.n_same   += int((wz == cz).sum())
            self._record_phist(wi)
            self.fieldM[wg] += FA*(self.mid[wg]-self.fieldM[wg])
        self.fieldM *= FIELD_DECAY
        if self.t % DIFFUSE_EVERY == 0: self._diffuse()
        self.age += 1; self._collapse(); self.t += 1


# ── Geometric P_hist prediction ────────────────────────────────────────────────
def geometric_p_hist(r_wave, zone_width=ZONE_WIDTH):
    """
    Analytical P_hist from zone-launch area geometry.
    For each cell position x in zone (0..zone_width-1):
      p_same(x) = zone_width / (zone_width + 2*min(x, zone_width-1-x, r_wave))
    P_hist_geom = mean over all x.
    Same formula as Paper 57's p_same, but now interpreted as P_hist not P_sp.
    """
    xs = np.arange(1, zone_width)  # exclude x=0 (zone boundary)
    p_same = zone_width / (zone_width + 2*np.minimum(
        np.minimum(xs, zone_width - xs), r_wave))
    return float(np.mean(p_same))


# ── Wave environment ────────────────────────────────────────────────────────────
class WaveEnv:
    def __init__(self,vcml,WR,supp_amp=0.06,exc_amp=0.12,r_wave=2,n_zones=4,rng_seed=0):
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
    D_adj    = float(np.mean(adj))   if adj  else 0.
    D_nonadj = float(np.mean(nadj))  if nadj else 0.
    sigma_w  = float(np.mean(spreads)) if spreads else 0.
    na       = float(D_nonadj/D_adj)          if D_adj>1e-8  else float('nan')
    c_order  = float((D_nonadj-D_adj)/sigma_w) if sigma_w>1e-5 else float('nan')
    p_causal = vcml.p_causal()
    p_hist   = vcml.p_hist()
    return dict(D_adj=D_adj, D_nonadj=D_nonadj, sigma_w=sigma_w,
                na=na, c_order=c_order, p_causal=p_causal, p_hist=p_hist)


def run_one(seed, r_wave, use_cp):
    VCMLClass = FastVCMLCP if use_cp else FastVCML
    vcml = VCMLClass(seed, ss=SS, half=S2['HALF'], h=S2['H'])
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
    seed, r_wave, use_cp = args
    return (seed, r_wave, use_cp, run_one(seed, r_wave, use_cp))


def mean_s(vals):
    v=[x for x in vals if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return float(np.mean(v)) if v else float('nan')

def se_s(vals):
    v=[x for x in vals if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return float(np.std(v)/math.sqrt(len(v))) if len(v)>1 else 0.


def aggregate(trajs):
    n_tp  = min(len(t) for t in trajs)
    steps = [trajs[0][i]['step'] for i in range(n_tp)]
    keys  = ['na','c_order','sigma_w','p_causal','p_hist']
    out   = {'steps': steps}
    for k in keys:
        out[f'{k}_mean'] = [mean_s([t[i][k] for t in trajs]) for i in range(n_tp)]
        out[f'{k}_se']   = [se_s( [t[i][k] for t in trajs]) for i in range(n_tp)]
    out['final_na']       = mean_s([t[-1]['na']       for t in trajs])
    out['final_sigma_w']  = mean_s([t[-1]['sigma_w']  for t in trajs])
    out['final_p_causal'] = mean_s([t[-1]['p_causal'] for t in trajs])
    out['final_p_hist']   = mean_s([t[-1]['p_hist']   for t in trajs])
    return out


if __name__ == '__main__':
    mp.freeze_support()
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    if os.path.exists(RESULTS_FILE):
        print(f'Loading existing results from {RESULTS_FILE}')
        with open(RESULTS_FILE) as f:
            raw = json.load(f)
    else:
        all_args = [(seed, rw, cp)
                    for cp in [True, False]
                    for rw in RWAVE_LEVELS
                    for seed in SEEDS]
        print(f'Running {len(all_args)} experiments in parallel...')
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            results = pool.map(_worker, all_args)
        raw = {}
        for seed, r_wave, use_cp, traj in results:
            mode = 'CP' if use_cp else 'ref'
            k = f'{mode}_rw{r_wave}'
            if k not in raw: raw[k] = {}
            raw[k][str(seed)] = traj
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print(f'Saved to {RESULTS_FILE}')

    # Geometric P_hist predictions
    print('\nGeometric P_hist predictions:')
    print(f'{"r_wave":>8} {"delta":>6} {"P_hist_geom":>12}')
    for rw in RWAVE_LEVELS:
        d = ZONE_WIDTH / rw
        ph = geometric_p_hist(rw)
        print(f'{rw:>8} {d:>6.2f} {ph:>12.3f}')

    # Aggregate results
    summary = {}
    print(f'\nResults at step {STEPS}:')
    print(f'{"mode":>6} {"r":>4} {"P_sp":>8} {"P_hist":>8} {"P_hist_g":>10} '
          f'{"sigma_w":>9} {"na":>8} {"identity":>10}')
    print('-'*70)

    for cp_flag, mode_label in [(True,'CP'), (False,'ref')]:
        for rw in RWAVE_LEVELS:
            k = f'{mode_label}_rw{rw}'
            if k not in raw: continue
            trajs = list(raw[k].values())
            agg = aggregate(trajs)
            summary[k] = agg
            ph_geom = geometric_p_hist(rw)
            idt = 'YES' if agg['final_na'] > 1.0 else 'no'
            print(f'{mode_label:>6} {rw:>4} '
                  f'{agg["final_p_causal"]:>8.3f} '
                  f'{agg["final_p_hist"]:>8.3f} '
                  f'{ph_geom:>10.3f} '
                  f'{agg["final_sigma_w"]:>9.4f} '
                  f'{agg["final_na"]:>8.3f} '
                  f'{idt:>10}')

    # The key test: does P_hist predict the r=8 failure for C_perturb?
    print('\nThree-purity breakdown for C_perturb:')
    print(f'{"r":>4} {"P_sp":>8} {"P_temp":>8} {"P_hist":>8} {"P_hist_g":>10} {"na":>8} {"predict":>10}')
    print('-'*60)
    for rw in RWAVE_LEVELS:
        k = f'CP_rw{rw}'
        if k not in summary: continue
        p_sp   = summary[k]['final_p_causal']
        p_hist = summary[k]['final_p_hist']
        p_hist_g = geometric_p_hist(rw)
        na     = summary[k]['final_na']
        # P_temp = 1 for C_perturb by construction
        # Predicted identity: P_sp > 0.88 AND P_hist > threshold
        predict = 'YES' if (p_hist > 0.71) else 'no'  # threshold TBD from data
        print(f'{rw:>4} {p_sp:>8.3f} {"1.000":>8} {p_hist:>8.3f} {p_hist_g:>10.3f} '
              f'{na:>8.3f} {predict:>10}')

    # P_hist vs sigma_w: does 1-P_hist predict sigma_w better than 1-P_sp?
    print('\nNoise model comparison: sigma_w vs 1-P_sp vs 1-P_hist (C_perturb):')
    print(f'{"r":>4} {"1-P_sp":>8} {"1-P_hist":>10} '
          f'{"sw/(1-Psp)":>12} {"sw/(1-Phist)":>14} {"sigma_w":>9}')
    print('-'*65)
    for rw in RWAVE_LEVELS:
        k = f'CP_rw{rw}'
        if k not in summary: continue
        p_sp   = summary[k]['final_p_causal']
        p_hist = summary[k]['final_p_hist']
        sw     = summary[k]['final_sigma_w']
        imp_sp   = 1 - p_sp
        imp_hist = 1 - p_hist
        r_sp   = sw/imp_sp   if imp_sp   > 1e-4 else float('nan')
        r_hist = sw/imp_hist if imp_hist > 1e-4 else float('nan')
        print(f'{rw:>4} {imp_sp:>8.3f} {imp_hist:>10.3f} '
              f'{r_sp:>12.3f} {r_hist:>14.3f} {sw:>9.4f}')

    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved analysis to {ANALYSIS_FILE}')
