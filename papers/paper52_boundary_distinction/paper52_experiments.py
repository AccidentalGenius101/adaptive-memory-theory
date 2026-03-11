"""
Paper 52: Boundary-Coupled Distinction

Question: Does structured memory under turnover depend specifically on writes that
occur near pre-inheritance viability boundaries, rather than merely during perturbation?

The A != B primitive in VCSM is not just "subtraction." The theory claims
boundary-coupled distinction is load-bearing: writes near the viability collapse
threshold are the specific inputs that feed intergenerational propagation (copy-forward).

Nine consolidation-mode conditions compared:
  C_ref       Standard gate (streak >= SS). Baseline.
  C_near      Gate on standard AND near viability boundary (low or high side).
  C_near_low  Gate on near-LOW boundary (about to collapse: vals near BOUND_LO).
  C_near_high Gate on near-HIGH boundary (vals near BOUND_HI).
  C_perturb   Gate on actively receiving wave perturbation (wa > 0.05).
  C_rand      Matched-rate random gate (calibrated to C_near write frequency).
  C_abs       Same gate as C_near but write |mid| (unsigned contrast, sign lost).
  C_raw       Same gate as C_near but write hid (raw state, no baseline subtraction).
  C_far       Gate on FAR from boundary (stable middle-range cells).

Two-phase design:
  Formation  (STEPS_FORM=2000): WaveEnvStd, all conditions run full VCSM.
  Adversarial (STEPS_ADV=1000): WaveEnvFlipped (supp/exc convention inverted),
              measures how well each condition resists adversarial overwrite.

Key prediction: C_near > C_perturb > C_rand >> C_far on maintenance_ratio and SNR.
If C_near > C_perturb: boundary coupling specifically matters, not just perturbation timing.
If C_far is lowest: stable cells cannot write useful content (signal near zero).

9 conditions x 5 seeds = 45 runs.
"""

import numpy as np, json, os, math, random, multiprocessing as mp

# ── Constants (same as paper50 baseline) ─────────────────────────────────────
W=80; H=40; HALF=40
HS=2; IS=3
WAVE_DUR=15
SUPP_AMP=0.25; EXC_AMP=0.50
MID_DECAY=0.99; FIELD_DECAY=0.9997; BASE_BETA=0.005
FA=0.16
VALS_DECAY=0.92; VALS_NAV=0.08; ADJ_SCALE=0.03
BOUND_LO=0.05; BOUND_HI=0.95; FRAG_NOISE=0.01
INST_THRESH=0.45; INST_PROB=0.03; DIFFUSE=0.02; DIFFUSE_EVERY=2

SEEDS       = list(range(5))
STEPS_FORM  = 2000   # formation phase
STEPS_ADV   = 1000   # adversarial flip phase
N_ZONES     = 4
WR          = 4.8
SEED_BETA   = 0.25

NEAR_BAND   = 0.15   # near-boundary band: vals within NEAR_BAND of BOUND_LO or BOUND_HI
FAR_BAND    = 0.25   # far-from-boundary: vals in [BOUND_LO+FAR_BAND, BOUND_HI-FAR_BAND]

CONDITIONS  = ['C_ref','C_near','C_near_low','C_near_high',
               'C_perturb','C_rand','C_abs','C_raw','C_far']

RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper52_results.json")


# ── FastVCML base (verbatim from paper50) ────────────────────────────────────
class FastVCML:
    def __init__(self, seed, ss=10, seed_beta=0.25):
        self.rng       = np.random.RandomState(seed)
        self.SS        = ss
        self.seed_beta = seed_beta
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
        for ai in ci:
            self.cc[ai]+=1
            fm=self.fieldM[ai]; mag=np.sqrt(np.dot(fm,fm))
            sb=self.seed_beta
            nh=((1-sb)*prev[ai]+sb*fm if mag>1e-6 else prev[ai].copy())
            nh+=rng.randn(HS)*FRAG_NOISE
            self.hid[ai]=nh; self.vals[ai]=.5; self.age[ai]=0
            self.streak[ai]=0; self.mid[ai]=np.zeros(HS)


# ── FastVCML52: mode-switched consolidation gate ──────────────────────────────
class FastVCML52(FastVCML):
    """
    FastVCML with a switchable fieldM consolidation gate.

    write_mode controls which cells are allowed to write to fieldM each step.
    write_src controls what content is written (mid, |mid|, or hid).
    """
    def __init__(self, seed, ss=10, seed_beta=0.25, write_mode='C_ref', rand_prob=0.05):
        super().__init__(seed, ss, seed_beta)
        self.write_mode = write_mode
        self.rand_prob  = rand_prob
        self.total_writes = 0  # cumulative write count for rate tracking

    def step(self, wa, wc):
        # ── GRU + viability (unchanged) ──────────────────────────────────────
        nb  = self._nbmean()
        an  = np.minimum(1., self.age / 300.)
        x   = np.stack([nb, np.minimum(1., wa), an], 1)
        hn, out = self._gru(x)
        self.hid  = hn
        self.vals = np.clip(VALS_DECAY*self.vals + VALS_NAV*nb + ADJ_SCALE*out, 0, 1)
        dev = hn - self.base_h
        self.base_h += BASE_BETA * dev
        self.streak = np.where(np.sum(dev**2, 1) < .0025, self.streak+1, 0)
        self.mid = (self.mid + FA*dev) * MID_DECAY

        # ── Mode-specific consolidation gate ─────────────────────────────────
        base_gate = self.streak >= self.SS
        wm        = self.write_mode
        v         = self.vals

        if wm == 'C_ref':
            wg = base_gate
            ws = self.mid

        elif wm == 'C_near':
            near = (v < BOUND_LO + NEAR_BAND) | (v > BOUND_HI - NEAR_BAND)
            wg = base_gate & near
            ws = self.mid

        elif wm == 'C_near_low':
            # Near the LOW collapse boundary: cell is suppressed, about to die
            nl = (v >= BOUND_LO) & (v < BOUND_LO + NEAR_BAND)
            wg = base_gate & nl
            ws = self.mid

        elif wm == 'C_near_high':
            # Near the HIGH collapse boundary: cell is over-excited, about to die
            nh = (v > BOUND_HI - NEAR_BAND) & (v <= BOUND_HI)
            wg = base_gate & nh
            ws = self.mid

        elif wm == 'C_perturb':
            # Actively receiving wave perturbation (confound control)
            in_wave = wa > 0.05
            wg = base_gate & in_wave
            ws = self.mid

        elif wm == 'C_rand':
            # Matched-rate random gate (rate calibrated to C_near average)
            wg = base_gate & (self.rng.random(self.N) < self.rand_prob)
            ws = self.mid

        elif wm == 'C_abs':
            # Near boundary, but write unsigned |mid| (direction of contrast lost)
            near = (v < BOUND_LO + NEAR_BAND) | (v > BOUND_HI - NEAR_BAND)
            wg = base_gate & near
            ws = np.abs(self.mid)

        elif wm == 'C_raw':
            # Near boundary, but write raw hid instead of contrast signal
            near = (v < BOUND_LO + NEAR_BAND) | (v > BOUND_HI - NEAR_BAND)
            wg = base_gate & near
            ws = self.hid  # no baseline subtraction

        elif wm == 'C_far':
            # Far from both boundaries: stable, low-perturbation cells
            far = (v >= BOUND_LO + FAR_BAND) & (v <= BOUND_HI - FAR_BAND)
            wg = base_gate & far
            ws = self.mid

        else:
            wg = base_gate
            ws = self.mid

        # ── Apply write ───────────────────────────────────────────────────────
        if wg.any():
            self.fieldM[wg] += FA * (ws[wg] - self.fieldM[wg])
        self.total_writes += int(wg.sum())

        # ── Standard field updates (unchanged) ───────────────────────────────
        self.fieldM *= FIELD_DECAY
        if self.t % DIFFUSE_EVERY == 0:
            self._diffuse()
        self.age += 1
        self._collapse()
        self.t += 1


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


class WaveEnvFlipped(WaveEnvStd):
    """Same zone-anchored launch positions as WaveEnvStd, but supp/exc convention inverted.
    Even-class zones are now EXCITED (were suppressed); odd-class zones are SUPPRESSED.
    Used for the adversarial phase to test maintenance of original encoding."""
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
        # FLIPPED: even -> excite, odd -> suppress
        for even in [True,False]:
            amp=EXC_AMP if even else SUPP_AMP  # swapped
            par=(wc%2==0) if even else (wc%2==1)
            idx=np.where((wc>=0)&par&(wa>.05))[0]
            if not len(idx): continue
            sc=wa[idx]*amp
            if even: self.vcml.vals[idx]=np.minimum(1.,self.vcml.vals[idx]+sc*.5)   # excite
            else:    self.vcml.vals[idx]=np.maximum(0.,self.vcml.vals[idx]*(1.-sc*.5))  # suppress
        return wa.copy(),wc.copy()


# ── Metrics (same as paper50) ─────────────────────────────────────────────────
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

def all_metrics(vcml, n_zones=4):
    sg4,sw,snr=sg4_fn(vcml,n_zones)
    na=na_ratio_fn(vcml,n_zones)
    acc,dnorm=decode_acc_fn(vcml,n_zones)
    cps=float(vcml.cc.sum())/vcml.N
    return dict(sg4=sg4,sw=sw,snr=snr,na_ratio=na,
                decode_acc=acc,decode_norm=dnorm,coll_per_site=cps)


# ── C_rand calibration ────────────────────────────────────────────────────────
def calibrate_rand_prob(n_steps=500, calib_seed=42):
    """Estimate the C_near write fraction over a representative run.
    Returns mean(n_C_near_writers / N) averaged over n_steps."""
    vcml = FastVCML(seed=calib_seed, ss=10, seed_beta=SEED_BETA)
    env  = WaveEnvStd(vcml, WR=WR, rng_seed=calib_seed*100, r_wave=2, n_zones=N_ZONES)
    rates = []
    for _ in range(n_steps):
        wa, _wc = env.step()
        base_gate = vcml.streak >= vcml.SS
        near = (vcml.vals < BOUND_LO + NEAR_BAND) | (vcml.vals > BOUND_HI - NEAR_BAND)
        c_near_gate = base_gate & near
        rates.append(c_near_gate.sum() / vcml.N)
        vcml.step(wa, _wc)
    return float(np.mean(rates))


# ── Run function ──────────────────────────────────────────────────────────────
def run(seed, condition, rand_prob):
    vcml = FastVCML52(seed=seed*3+7, ss=10, seed_beta=SEED_BETA,
                      write_mode=condition, rand_prob=rand_prob)
    # Formation phase: zone-structured waves
    env_form = WaveEnvStd(vcml, WR=WR, rng_seed=seed*100, r_wave=2, n_zones=N_ZONES)
    for _ in range(STEPS_FORM):
        wa, wc = env_form.step()
        vcml.step(wa, wc)

    m_form = all_metrics(vcml, N_ZONES)
    writes_per_site = vcml.total_writes / (vcml.N * STEPS_FORM)

    # Record pre-adversarial fieldM signature for maintenance tracking
    fieldM_snap = vcml.fieldM.copy()

    # Adversarial phase: same zone positions, flipped supp/exc convention
    env_adv = WaveEnvFlipped(vcml, WR=WR, rng_seed=seed*100, r_wave=2, n_zones=N_ZONES)
    for _ in range(STEPS_ADV):
        wa, wc = env_adv.step()
        vcml.step(wa, wc)

    m_adv = all_metrics(vcml, N_ZONES)

    # Maintenance ratio: how much of formation structure survives adversarial flip
    maint_sg4  = m_adv['sg4']  / m_form['sg4']  if m_form['sg4']  > 1e-8 else float('nan')
    maint_snr  = m_adv['snr']  / m_form['snr']  if m_form['snr']  > 1e-8 else float('nan')

    # Cosine similarity of post-adv fieldM to pre-adv fieldM (zone means)
    zw = HALF // N_ZONES; cos_sims = []
    for z in range(N_ZONES):
        xlo=HALF+z*zw; xhi=HALF+(z+1)*zw
        mask=(vcml.ai_x>=xlo)&(vcml.ai_x<xhi)
        m_pre  = fieldM_snap[mask].mean(0)
        m_post = vcml.fieldM[mask].mean(0)
        n1=np.linalg.norm(m_pre); n2=np.linalg.norm(m_post)
        if n1>1e-8 and n2>1e-8: cos_sims.append(float(np.dot(m_pre,m_post)/(n1*n2)))
    cos_maint = float(np.mean(cos_sims)) if cos_sims else float('nan')

    return {
        'condition':       condition,
        'seed':            seed,
        # Formation metrics
        'sg4_form':        m_form['sg4'],
        'snr_form':        m_form['snr'],
        'na_form':         m_form['na_ratio'],
        'dnorm_form':      m_form['decode_norm'],
        'coll_form':       m_form['coll_per_site'],
        # Adversarial metrics
        'sg4_adv':         m_adv['sg4'],
        'snr_adv':         m_adv['snr'],
        'na_adv':          m_adv['na_ratio'],
        'dnorm_adv':       m_adv['decode_norm'],
        # Maintenance
        'maint_sg4':       maint_sg4,
        'maint_snr':       maint_snr,
        'cos_maint':       cos_maint,
        # Write rate
        'writes_per_site': writes_per_site,
    }


# ── Worker for multiprocessing ────────────────────────────────────────────────
def _worker(args):
    seed, cond, rand_prob = args
    return run(seed, cond, rand_prob)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load cached results if present
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            saved = json.load(f)
        done = {(r['condition'], r['seed']) for r in saved}
    else:
        saved = []; done = set()

    total = len(CONDITIONS) * len(SEEDS)
    remaining = [(s,c) for c in CONDITIONS for s in SEEDS if (c,s) not in done]
    print(f"Tasks remaining: {len(remaining)} / {total}")
    if not remaining:
        analyze(saved)
        return

    # Calibrate C_rand rate
    print("Calibrating C_rand rate...")
    rand_prob = calibrate_rand_prob(n_steps=800)
    print(f"  C_rand prob = {rand_prob:.4f}")

    mp.freeze_support()
    all_args = [(s, c, rand_prob) for s, c in remaining]
    with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
        new_results = pool.map(_worker, all_args)

    saved.extend(new_results)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(saved, f, indent=2)
    print(f"Saved {len(saved)} total results.")
    analyze(saved)


def analyze(results):
    from collections import defaultdict
    stats = defaultdict(lambda: defaultdict(list))
    for r in results:
        c = r['condition']
        for k in ['sg4_form','snr_form','na_form','dnorm_form',
                  'sg4_adv','snr_adv','na_adv','dnorm_adv',
                  'maint_sg4','maint_snr','cos_maint','writes_per_site','coll_form']:
            if k in r and r[k] is not None and not (isinstance(r[k], float) and math.isnan(r[k])):
                stats[c][k].append(r[k])

    # Build summary
    summary = {}
    for c in CONDITIONS:
        s = {}
        for k, vals in stats[c].items():
            s[k] = float(np.mean(vals))
        summary[c] = s

    # Save summary
    analysis_file = os.path.join(os.path.dirname(RESULTS_FILE), 'paper52_analysis.json')
    with open(analysis_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'':18s} {'sg4_f':6s} {'na_f':6s} {'snr_f':6s} "
          f"{'maint':6s} {'cos':6s} {'wps':6s} {'na_adv':6s}")
    print("-"*72)
    for c in CONDITIONS:
        s = summary.get(c, {})
        def g(k): return f"{s.get(k, float('nan')):.4f}"
        enc_f = 'IDENT' if s.get('na_form', 0) > 1 else 'parity'
        enc_a = 'IDENT' if s.get('na_adv',  0) > 1 else 'parity'
        print(f"{c:18s} {g('sg4_form')} {g('na_form'):7s}[{enc_f}] "
              f"{g('snr_form')} {g('maint_sg4')} {g('cos_maint')} "
              f"{g('writes_per_site')} {g('na_adv'):7s}[{enc_a}]")

    print(f"\nAnalysis saved to {analysis_file}")


if __name__ == '__main__':
    main()
