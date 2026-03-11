"""
Paper 49: C2 Completion, Gate Optimum, and Population-Code Relay

Three experiments closing the open threads from Papers 46-48:

Exp A (30 runs): beta_s sweep -- final C2 coordinate.
  Fix Omega=62.4 (WR=4.8, r_wave=2), delta=5 (N_ZONES=4, zone_w=10).
  Sweep SEED_BETA in {0.0, 0.1, 0.25, 0.5, 0.75, 1.0}.
  Prediction: sg4 and SNR increase monotonically with beta_s (stronger
  seeding builds cleaner zone structure via copy-forward). If true, beta_s
  is a genuine third C2 coordinate -- independent tuning axis.
  5 seeds x 6 beta_s values = 30 runs.

Exp B (180 runs): SS*(T_encode, adv_amp) -- optimal gate as 2D surface.
  Phase 1 [0..T_encode]: Standard SS=10, standard waves. Build structure.
  At T_encode: snapshot sg4, switch gate to condition.
  Phase 2 [T_encode..+1000]: Adversarial or standard waves + new gate.
  T_encode in {1000, 2000, 3000} x adv_amp in {0.0, 0.25, 0.50} x
  gate in {ss0, ss10, ss20, rand_p60} x 5 seeds = 180 runs.
  Metric: sg4 and SNR at end of Phase 2 (avoids commitment-epoch fidelity
  confound from Paper 47). Retention ratio = final_sg4 / snap_sg4.

Exp C (15 runs): Population-code relay -- zone-mean vs geo interface.
  L1: standard VCML with zone-structured waves (4 zones, WR=4.8).
  L2: three coupling modes:
    geo:        same (cx,cy,cls) wave events as L1 (Paper 35 geographic)
    mean_relay: random waves to L2, but births seeded from L1 zone-mean
                fieldM instead of L2's own spatial-neighbor fieldM
    ctrl:       random waves, births from own fieldM (Paper 35 control)
  Metric: relay gain G = sg4_L2 / sg4_ctrl_standalone.
  N_ZONES=4, WR=4.8, 5 seeds x 3 conditions = 15 runs.
"""
import numpy as np, json, os, math, random, multiprocessing as mp

# ── Fixed constants ────────────────────────────────────────────────────────────
W=80; H=40; HALF=40
HS=2; IS=3
WAVE_DUR=15
SUPP_AMP=0.25; EXC_AMP=0.50
MID_DECAY=0.99; FIELD_DECAY=0.9997; BASE_BETA=0.005
FA=0.16
VALS_DECAY=0.92; VALS_NAV=0.08; ADJ_SCALE=0.03
BOUND_LO=0.05; BOUND_HI=0.95; FRAG_NOISE=0.01
INST_THRESH=0.45; INST_PROB=0.03; DIFFUSE=0.02; DIFFUSE_EVERY=2
SEEDS = list(range(5))
STEPS_STD = 3000
WARMUP    = 300
SAMPLE_EVERY = 20
TAIL = 30

# Exp A
SEED_BETAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
SEED_BETA_REF = 0.25   # current standard

# Exp B
T_ENCODES  = [1000, 2000, 3000]
ADV_AMPS   = [0.0, 0.25, 0.50]
T_ADV      = 1000
B_GATES = {
    'ss0':      {'ss': 0,  'rand_gate': False, 'p_write': None},
    'ss10':     {'ss': 10, 'rand_gate': False, 'p_write': None},
    'ss20':     {'ss': 20, 'rand_gate': False, 'p_write': None},
    'rand_p60': {'ss': 10, 'rand_gate': True,  'p_write': 0.60},
}

# Exp C
N_ZONES_C = 4; WR_C = 4.8
C_COUPLINGS = ['geo', 'mean_relay', 'ctrl']

RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper49_results.json")


# ── FastVCML ──────────────────────────────────────────────────────────────────
class FastVCML:
    def __init__(self, seed, ss=10, rand_gate=False, p_write=None,
                 seed_beta=0.25):
        self.rng       = np.random.RandomState(seed)
        self.SS        = ss
        self.rand_gate = rand_gate
        self.p_write   = p_write if p_write is not None else 0.60
        self.seed_beta = seed_beta
        # optional zone-mean source for Exp C mean_relay
        self._zm_source = None   # (l1_instance, n_zones)
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
        self.n_writes=0
        self._build_nb()

    def _build_nb(self):
        N=self.N; rx=np.arange(N)%HALF; y=np.arange(N)//HALF
        nb=np.full((N,4),-1,int)
        nb[rx>0,0]=np.where(rx>0)[0]-1; nb[rx<HALF-1,1]=np.where(rx<HALF-1)[0]+1
        nb[y>0,2]=np.where(y>0)[0]-HALF; nb[y<H-1,3]=np.where(y<H-1)[0]+HALF
        self.nb=nb; self.nbc=(nb>=0).sum(1).astype(float)

    def set_zone_mean_source(self, l1, n_zones):
        """Exp C: use l1 zone-mean fieldM for L2 birth seeding."""
        self._zm_source = (l1, n_zones)

    def switch_gate(self, ss, rand_gate, p_write):
        """Exp B: switch gate configuration mid-run."""
        self.SS = ss; self.rand_gate = rand_gate
        if p_write is not None: self.p_write = p_write

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
        if self.rand_gate:
            gate=self.rng.random(self.N)<self.p_write
        elif self.SS==0:
            gate=np.ones(self.N,bool)
        else:
            gate=self.streak>=self.SS
        self.fieldM[gate]+=FA*(self.mid[gate]-self.fieldM[gate])
        self.n_writes+=int(gate.sum())
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
        if self._zm_source is not None:
            # mean_relay: seed from L1 zone-mean fieldM for this cell's zone
            l1, nz = self._zm_source
            zw = HALF // nz
            zm = self._get_zone_means(l1, nz)
        for ai in ci:
            self.cc[ai]+=1
            if self._zm_source is not None:
                # use L1 zone mean for zone of cell ai
                zone_idx = min(nz-1, int((self.ai_x[ai]-HALF)//zw))
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
        """Compute zone-mean fieldM of l1 for Exp C mean_relay."""
        zw = HALF // n_zones
        means = []
        for z in range(n_zones):
            xlo = HALF + z*zw; xhi = HALF + (z+1)*zw
            mask = (l1.ai_x >= xlo) & (l1.ai_x < xhi)
            fm = l1.fieldM[mask]
            means.append(fm.mean(0) if len(fm) else np.zeros(HS))
        return means


# ── Metrics (full suite: sg4, sw, SNR, na_ratio, decode_norm, mm_lda, coll/site)
def sg4_fn(vcml, n_zones=4):
    zw = HALF // n_zones
    means = []; spreads = []
    for z in range(n_zones):
        xlo=HALF+z*zw; xhi=HALF+(z+1)*zw
        mask=(vcml.ai_x>=xlo)&(vcml.ai_x<xhi)
        fm=vcml.fieldM[mask]
        if len(fm)>1:
            mz=fm.mean(0); means.append(mz)
            spreads.append(float(np.std(fm)))
        else:
            means.append(np.zeros(HS)); spreads.append(float('nan'))
    dists=[np.linalg.norm(means[a]-means[b])
           for a in range(n_zones) for b in range(a+1,n_zones)]
    sg4=float(np.mean(dists)) if dists else 0.
    sw_vals=[s for s in spreads if not math.isnan(s)]
    sw=float(np.mean(sw_vals)) if sw_vals else float('nan')
    snr=float(sg4/sw) if sw and sw>1e-8 else float('nan')
    return sg4, sw, snr

def na_ratio_fn(vcml, n_zones=4):
    zw = HALF // n_zones
    means=[]
    for z in range(n_zones):
        xlo=HALF+z*zw; xhi=HALF+(z+1)*zw
        mask=(vcml.ai_x>=xlo)&(vcml.ai_x<xhi)
        fm=vcml.fieldM[mask]
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
    """sg_C: nearest-centroid zone decode accuracy (Paper 48).
    Returns (raw_acc, decode_norm). decode_norm=0 is chance, 1 is perfect.
    """
    zw = HALF // n_zones
    true_zones = np.minimum(n_zones-1, (vcml.ai_x - HALF) // zw).astype(int)
    centroids = []
    for z in range(n_zones):
        mask = true_zones == z
        fm = vcml.fieldM[mask]
        centroids.append(fm.mean(0) if len(fm) else np.zeros(HS))
    centroids = np.array(centroids)
    dists = np.linalg.norm(vcml.fieldM[:, None, :] - centroids[None, :, :], axis=2)
    pred_zones = np.argmin(dists, axis=1)
    acc = float((pred_zones == true_zones).mean())
    chance = 1.0 / n_zones
    dnorm = float((acc - chance) / (1.0 - chance))
    return acc, dnorm

def mm_lda_fn(vcml, n_zones=4):
    """Mid-memory zone discriminability: between/within variance ratio.
    Orthogonal to fieldM -- confirms zone info is present in fast-timescale dynamics.
    """
    zw = HALF // n_zones
    true_zones = np.minimum(n_zones-1, (vcml.ai_x - HALF) // zw).astype(int)
    grand_mean = vcml.mid.mean(0)
    between = 0.0; within = 0.0
    for z in range(n_zones):
        mask = true_zones == z
        zm = vcml.mid[mask]
        if len(zm) < 2: continue
        zm_mean = zm.mean(0)
        between += len(zm) * float(np.sum((zm_mean - grand_mean)**2))
        within  += float(np.sum((zm - zm_mean)**2))
    if within < 1e-12: return float('nan')
    return float(between / within)

def all_metrics(vcml, n_zones=4):
    """Compute and return full metric suite as a dict."""
    sg4, sw, snr = sg4_fn(vcml, n_zones)
    na             = na_ratio_fn(vcml, n_zones)
    acc, dnorm     = decode_acc_fn(vcml, n_zones)
    lda            = mm_lda_fn(vcml, n_zones)
    cps            = float(vcml.cc.sum()) / vcml.N   # collapses per site (total)
    return dict(sg4=sg4, sw=sw, snr=snr, na_ratio=na,
                decode_acc=acc, decode_norm=dnorm,
                mm_lda=lda, coll_per_site=cps)


# ── Wave environments ─────────────────────────────────────────────────────────
class WaveEnvStd:
    """Standard zone-structured waves. Variable r_wave, adversarial support."""
    def __init__(self, vcml, WR, rng_seed=0, r_wave=2, n_zones=4,
                 adv=False, adv_amp=0.0):
        self.vcml=vcml; self.WR=WR; self.rwave=r_wave; self.nz=n_zones
        self.zw=HALF//n_zones; self.adv=adv; self.adv_amp=adv_amp
        self.rng=random.Random(rng_seed); self.waves=[]
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
        if not self.adv:
            for even in [True,False]:
                amp=SUPP_AMP if even else EXC_AMP
                par=(wc%2==0) if even else (wc%2==1)
                idx=np.where((wc>=0)&par&(wa>.05))[0]
                if not len(idx): continue
                sc=wa[idx]*amp
                if even: self.vcml.vals[idx]=np.maximum(0.,self.vcml.vals[idx]*(1.-sc*.5))
                else:    self.vcml.vals[idx]=np.minimum(1.,self.vcml.vals[idx]+sc*.5)
        else:
            s=self.adv_amp
            for even in [True,False]:
                amp=EXC_AMP*s if even else SUPP_AMP*s
                excite=even   # flipped
                par=(wc%2==0) if even else (wc%2==1)
                idx=np.where((wc>=0)&par&(wa>.05))[0]
                if not len(idx): continue
                sc=wa[idx]*amp
                if excite: self.vcml.vals[idx]=np.minimum(1.,self.vcml.vals[idx]+sc*.5)
                else:      self.vcml.vals[idx]=np.maximum(0.,self.vcml.vals[idx]*(1.-sc*.5))
        return wa.copy(), wc.copy()

    def step(self):
        exp=self.WR/WAVE_DUR
        nl=int(exp)+(1 if self.rng.random()<exp-int(exp) else 0)
        for _ in range(nl): self._launch()
        return self._apply()


class WaveEnvRandom:
    """Random-position waves (no zone structure). For Exp C ctrl/mean_relay."""
    def __init__(self, vcml, WR, rng_seed=0, r_wave=2):
        self.vcml=vcml; self.WR=WR; self.rwave=r_wave
        self.rng=random.Random(rng_seed); self.waves=[]
        N=vcml.N; self._wa=np.empty(N); self._wc=np.empty(N,int)

    def _launch(self):
        cx=self.rng.randint(HALF,HALF+HALF-1); cy=self.rng.randint(0,H-1)
        cls=self.rng.randint(0,1)  # arbitrary class, suppression/excitation random
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
        return wa.copy(), wc.copy()

    def step(self):
        exp=self.WR/WAVE_DUR
        nl=int(exp)+(1 if self.rng.random()<exp-int(exp) else 0)
        for _ in range(nl): self._launch()
        return self._apply()


# ── Run functions ─────────────────────────────────────────────────────────────
def run_A(seed, seed_beta):
    """Exp A: beta_s sweep."""
    vcml=FastVCML(seed=seed, ss=10, seed_beta=seed_beta)
    env=WaveEnvStd(vcml, WR=4.8, rng_seed=seed*7+1, r_wave=2, n_zones=4)
    for t in range(STEPS_STD):
        wa,wc=env.step(); vcml.step(wa,wc)
    m = all_metrics(vcml, 4)
    return {'exp':'A','seed':seed,'seed_beta':seed_beta, **m}


def run_B(seed, t_encode, adv_amp, gate_name):
    """Exp B: SS*(T_encode, adv_amp) 2D surface."""
    gcfg=B_GATES[gate_name]
    vcml=FastVCML(seed=seed, ss=10, seed_beta=SEED_BETA_REF)
    env=WaveEnvStd(vcml, WR=4.8, rng_seed=seed*13+2, r_wave=2, n_zones=4)
    # Phase 1: standard
    for t in range(t_encode):
        wa,wc=env.step(); vcml.step(wa,wc)
    snap = all_metrics(vcml, 4)
    # Switch gate
    vcml.switch_gate(gcfg['ss'], gcfg['rand_gate'], gcfg['p_write'])
    # Phase 2: adversarial or standard
    env2=WaveEnvStd(vcml, WR=4.8, rng_seed=seed*31+3, r_wave=2, n_zones=4,
                    adv=(adv_amp>0), adv_amp=adv_amp)
    for t in range(T_ADV):
        wa,wc=env2.step(); vcml.step(wa,wc)
    fin = all_metrics(vcml, 4)
    ret = float(fin['sg4']/snap['sg4']) if snap['sg4']>1e-6 else float('nan')
    return {
        'exp':'B','seed':seed,'t_encode':t_encode,'adv_amp':adv_amp,'gate':gate_name,
        'snap_sg4':snap['sg4'],'snap_snr':snap['snr'],'snap_dnorm':snap['decode_norm'],
        'fin_sg4':fin['sg4'],'fin_sw':fin['sw'],'fin_snr':fin['snr'],
        'fin_dnorm':fin['decode_norm'],'fin_mm_lda':fin['mm_lda'],
        'fin_coll_per_site':fin['coll_per_site'],'retention':ret
    }


def run_C(seed, coupling):
    """Exp C: population-code relay.
    coupling in {geo, mean_relay, ctrl}.
    L1: always zone-structured waves.
    L2: coupling-specific waves and/or birth seeding.
    """
    l1=FastVCML(seed=seed*3,   ss=10, seed_beta=SEED_BETA_REF)
    l2=FastVCML(seed=seed+101, ss=10, seed_beta=SEED_BETA_REF)

    # L1 always gets zone-structured waves
    env_l1=WaveEnvStd(l1, WR=WR_C, rng_seed=seed*100, r_wave=2, n_zones=N_ZONES_C)

    if coupling=='geo':
        # L2 gets SAME (cx,cy,cls) as L1 -- geographic relay
        # Implement via shared GeoEnv pattern
        env_l2=None   # handled specially below
        geo_waves=[]  # shared wave list
    elif coupling=='mean_relay':
        # L2 gets random waves, but birth seeded from L1 zone means
        l2.set_zone_mean_source(l1, N_ZONES_C)
        env_l2=WaveEnvRandom(l2, WR=WR_C, rng_seed=seed*200+7, r_wave=2)
    else:  # ctrl
        env_l2=WaveEnvRandom(l2, WR=WR_C, rng_seed=seed*300+11, r_wave=2)

    for t in range(STEPS_STD):
        wa1,wc1=env_l1.step()

        if coupling=='geo':
            wa2=wa1.copy(); wc2=wc1.copy()
            _apply_geo_to_l2(l2, wa2, wc2)
            l2.step(wa2, wc2)
        else:
            wa2,wc2=env_l2.step()
            l2.step(wa2, wc2)

        l1.step(wa1, wc1)

    m_l1 = all_metrics(l1, N_ZONES_C)
    m_l2 = all_metrics(l2, N_ZONES_C)
    return {
        'exp':'C','seed':seed,'coupling':coupling,
        'sg4_l1':m_l1['sg4'], 'sw_l1':m_l1['sw'],   'snr_l1':m_l1['snr'],
        'dnorm_l1':m_l1['decode_norm'], 'mm_lda_l1':m_l1['mm_lda'],
        'na_l1':m_l1['na_ratio'], 'coll_l1':m_l1['coll_per_site'],
        'sg4_l2':m_l2['sg4'], 'sw_l2':m_l2['sw'],   'snr_l2':m_l2['snr'],
        'dnorm_l2':m_l2['decode_norm'], 'mm_lda_l2':m_l2['mm_lda'],
        'na_l2':m_l2['na_ratio'], 'coll_l2':m_l2['coll_per_site'],
    }


def _apply_geo_to_l2(l2, wa, wc):
    """Apply the same wave amplitude/class pattern to L2's viability."""
    idx_supp=np.where((wc>=0)&(wc%2==0)&(wa>.05))[0]
    idx_exc =np.where((wc>=0)&(wc%2==1)&(wa>.05))[0]
    if len(idx_supp):
        sc=wa[idx_supp]*SUPP_AMP
        l2.vals[idx_supp]=np.maximum(0.,l2.vals[idx_supp]*(1.-sc*.5))
    if len(idx_exc):
        sc=wa[idx_exc]*EXC_AMP
        l2.vals[idx_exc]=np.minimum(1.,l2.vals[idx_exc]+sc*.5)


# ── Parallel dispatch ─────────────────────────────────────────────────────────
def _worker(args):
    exp=args[0]
    if exp=='A':   return run_A(*args[1:])
    elif exp=='B': return run_B(*args[1:])
    else:          return run_C(*args[1:])


def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    # Load existing results
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f: results=json.load(f)
    else:
        results=[]

    done_keys=set()
    for r in results:
        if r['exp']=='A':   done_keys.add(('A',r['seed'],r['seed_beta']))
        elif r['exp']=='B': done_keys.add(('B',r['seed'],r['t_encode'],r['adv_amp'],r['gate']))
        else:               done_keys.add(('C',r['seed'],r['coupling']))

    # Build task list
    all_args=[]
    for seed in SEEDS:
        for sb in SEED_BETAS:
            if ('A',seed,sb) not in done_keys:
                all_args.append(('A',seed,sb))
    for seed in SEEDS:
        for te in T_ENCODES:
            for aa in ADV_AMPS:
                for gn in B_GATES:
                    if ('B',seed,te,aa,gn) not in done_keys:
                        all_args.append(('B',seed,te,aa,gn))
    for seed in SEEDS:
        for coup in C_COUPLINGS:
            if ('C',seed,coup) not in done_keys:
                all_args.append(('C',seed,coup))

    print(f"Tasks remaining: {len(all_args)} / {30+180+15}")

    if not all_args:
        print("All done. Loading results.")
    else:
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            new=pool.map(_worker, all_args)
        results.extend(new)
        with open(RESULTS_FILE,'w') as f: json.dump(results,f,indent=2)
        print(f"Saved {len(results)} total results.")

    # ── Analysis ──────────────────────────────────────────────────────────────
    A=[r for r in results if r['exp']=='A']
    B=[r for r in results if r['exp']=='B']
    C=[r for r in results if r['exp']=='C']

    print("\n=== Exp A: beta_s sweep ===")
    print(f"{'beta_s':>8} {'sg4':>8} {'sw':>8} {'snr':>8} {'na_ratio':>10} {'dnorm':>8} {'mm_lda':>8} {'coll/site':>10}")
    for sb in SEED_BETAS:
        rows=[r for r in A if r['seed_beta']==sb]
        if not rows: continue
        print(f"{sb:8.2f} "
              f"{np.nanmean([r['sg4']          for r in rows]):8.4f} "
              f"{np.nanmean([r['sw']           for r in rows]):8.4f} "
              f"{np.nanmean([r['snr']          for r in rows]):8.4f} "
              f"{np.nanmean([r['na_ratio']     for r in rows]):10.4f} "
              f"{np.nanmean([r['decode_norm']  for r in rows]):8.4f} "
              f"{np.nanmean([r['mm_lda']       for r in rows]):8.4f} "
              f"{np.nanmean([r['coll_per_site']for r in rows]):10.4f}")

    print("\n=== Exp B: optimal gate (rows=T_encode, cols=gate, slices=adv_amp) -- SNR ===")
    for aa in ADV_AMPS:
        print(f"\n  adv_amp={aa}")
        print(f"  {'T_enc':>6} {'ss0':>8} {'ss10':>8} {'ss20':>8} {'rand':>8}  (SNR)")
        for te in T_ENCODES:
            row=[]
            for gn in ['ss0','ss10','ss20','rand_p60']:
                rows=[r for r in B if r['t_encode']==te and
                      abs(r['adv_amp']-aa)<1e-6 and r['gate']==gn]
                row.append(np.nanmean([r['fin_snr'] for r in rows]) if rows else float('nan'))
            print(f"  {te:6d} {row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f} {row[3]:8.3f}")
    print("\n  -- decode_norm (sg_C) --")
    for aa in ADV_AMPS:
        print(f"\n  adv_amp={aa}")
        print(f"  {'T_enc':>6} {'ss0':>8} {'ss10':>8} {'ss20':>8} {'rand':>8}  (dnorm)")
        for te in T_ENCODES:
            row=[]
            for gn in ['ss0','ss10','ss20','rand_p60']:
                rows=[r for r in B if r['t_encode']==te and
                      abs(r['adv_amp']-aa)<1e-6 and r['gate']==gn]
                row.append(np.nanmean([r['fin_dnorm'] for r in rows]) if rows else float('nan'))
            print(f"  {te:6d} {row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f} {row[3]:8.3f}")

    print("\n=== Exp C: population-code relay ===")
    print(f"{'coupling':>12} {'sg4_L2':>8} {'snr_L2':>8} {'dnorm_L2':>10} {'mm_lda_L2':>10} {'na_L2':>8} {'coll_L2':>10}")
    for coup in C_COUPLINGS:
        rows=[r for r in C if r['coupling']==coup]
        if not rows: continue
        print(f"{coup:>12} "
              f"{np.nanmean([r['sg4_l2']   for r in rows]):8.4f} "
              f"{np.nanmean([r['snr_l2']   for r in rows]):8.4f} "
              f"{np.nanmean([r['dnorm_l2'] for r in rows]):10.4f} "
              f"{np.nanmean([r['mm_lda_l2']for r in rows]):10.4f} "
              f"{np.nanmean([r['na_l2']    for r in rows]):8.4f} "
              f"{np.nanmean([r['coll_l2']  for r in rows]):10.4f}")

    # Save analysis summary
    summary={'A':{},'B':{},'C':{}}
    for sb in SEED_BETAS:
        rows=[r for r in A if r['seed_beta']==sb]
        if rows:
            summary['A'][str(sb)]={
                'sg4':     float(np.nanmean([r['sg4']           for r in rows])),
                'sw':      float(np.nanmean([r['sw']            for r in rows])),
                'snr':     float(np.nanmean([r['snr']           for r in rows])),
                'na':      float(np.nanmean([r['na_ratio']      for r in rows])),
                'dnorm':   float(np.nanmean([r['decode_norm']   for r in rows])),
                'mm_lda':  float(np.nanmean([r['mm_lda']        for r in rows])),
                'coll_per_site': float(np.nanmean([r['coll_per_site'] for r in rows])),
                'sg4_se':  float(np.nanstd([r['sg4'] for r in rows])/np.sqrt(len(rows)))
            }
    for te in T_ENCODES:
        for aa in ADV_AMPS:
            for gn in B_GATES:
                rows=[r for r in B if r['t_encode']==te and
                      abs(r['adv_amp']-aa)<1e-6 and r['gate']==gn]
                if rows:
                    k=f"{te}_{aa}_{gn}"
                    summary['B'][k]={
                        'snap_sg4':  float(np.nanmean([r['snap_sg4']  for r in rows])),
                        'snap_snr':  float(np.nanmean([r['snap_snr']  for r in rows])),
                        'snap_dnorm':float(np.nanmean([r['snap_dnorm']for r in rows])),
                        'fin_sg4':   float(np.nanmean([r['fin_sg4']   for r in rows])),
                        'fin_snr':   float(np.nanmean([r['fin_snr']   for r in rows])),
                        'fin_dnorm': float(np.nanmean([r['fin_dnorm'] for r in rows])),
                        'fin_mm_lda':float(np.nanmean([r['fin_mm_lda']for r in rows])),
                        'fin_coll_per_site':float(np.nanmean([r['fin_coll_per_site']for r in rows])),
                        'retention': float(np.nanmean([r['retention'] for r in rows]))
                    }
    for coup in C_COUPLINGS:
        rows=[r for r in C if r['coupling']==coup]
        if rows:
            summary['C'][coup]={
                'sg4_l1':  float(np.nanmean([r['sg4_l1']   for r in rows])),
                'snr_l1':  float(np.nanmean([r['snr_l1']   for r in rows])),
                'dnorm_l1':float(np.nanmean([r['dnorm_l1'] for r in rows])),
                'na_l1':   float(np.nanmean([r['na_l1']    for r in rows])),
                'sg4_l2':  float(np.nanmean([r['sg4_l2']   for r in rows])),
                'snr_l2':  float(np.nanmean([r['snr_l2']   for r in rows])),
                'dnorm_l2':float(np.nanmean([r['dnorm_l2'] for r in rows])),
                'mm_lda_l2':float(np.nanmean([r['mm_lda_l2']for r in rows])),
                'na_l2':   float(np.nanmean([r['na_l2']    for r in rows])),
                'coll_l2': float(np.nanmean([r['coll_l2']  for r in rows])),
            }
    an_file=RESULTS_FILE.replace('results.json','analysis.json')
    with open(an_file,'w') as f: json.dump(summary,f,indent=2)
    print(f"\nAnalysis saved to {an_file}")


if __name__=='__main__':
    mp.freeze_support()
    main()
