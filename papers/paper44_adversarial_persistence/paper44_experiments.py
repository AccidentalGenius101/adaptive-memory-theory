"""
paper44_experiments.py -- Adversarial Persistence Test  (v2)

Redesigned from v1:
  - T_ENCODE = 3000 (past commitment epoch t*~2000; fully encoded structure)
  - ALL conditions use full adaptive mechanism during encoding
  - Mechanism switches AT T_ENCODE for adversarial phase
  - Adversarial amplitude = 50% of encoding amplitude (moderate conflict)
  - This cleanly isolates mechanism effect on resistance vs imprinting

Design:
  Phase 1 [T=0 .. T_ENCODE=3000]:  FULL VCML mechanism, normal waves.
                                    All conditions identical. Build zone structure.
  Switch at T_ENCODE:               Change mechanism to condition-specific.
  Phase 2 [T_ENCODE .. +T_ADV]:    Adversarial input (supp/exc FLIPPED, 50% amp).
                                    Mechanisms differ across conditions.

Primary metric: "Fidelity" = mean cosine similarity between current zone-mean
  fieldM and zone-mean fieldM at T_ENCODE.
    Fidelity = 1  ->  original structure perfectly preserved
    Fidelity > 0  ->  partial original structure maintained
    Fidelity = 0  ->  structure orthogonal to original
    Fidelity < 0  ->  system has encoded adversarial (flipped) pattern

Three adversarial conditions:
  adaptive:  Full VCML during adversarial phase.
             Prediction: fidelity decays gradually (~800-step half-life).
  passive:   No gating + no seeding during adversarial phase (diffusion only).
             Prediction: fidelity collapses fast (< 200 steps).
  rigid:     No collapses during adversarial phase.
             Prediction: fidelity stays near 1 (frozen; no adaptation).

The three curves separate memory (slow decay) from imprinting (fast collapse)
from rigidity (no adaptation). Paper 43 Exp-B and metastability (Papers 38-42)
predict the adaptive resistance timescale.

3 conditions x 5 seeds = 15 runs.  T_total = 5000.  Results saved to JSON.
"""
import numpy as np, json, os, math, random, multiprocessing as mp

# ── Standard VCML constants (identical to Paper 43) ──────────────────────────
W=80; H=40; HALF=40
HS=2; IS=3
WAVE_DUR=15
SUPP_AMP=0.25; EXC_AMP=0.50          # encoding amplitude
SUPP_AMP_ADV=0.125; EXC_AMP_ADV=0.25  # adversarial amplitude (50%)
ZONE_K=320
FIELD_DECAY=0.9997; BASE_BETA=0.005
FA_STD=0.16
VALS_DECAY=0.92; VALS_NAV=0.08; ADJ_SCALE=0.03
BOUND_LO=0.05; BOUND_HI=0.95; FRAG_NOISE=0.01
INST_THRESH=0.45; INST_PROB=0.03; DIFFUSE=0.02; DIFFUSE_EVERY=2

WR_FIXED=4.8; N_ZONES=4; ZW=HALF//N_ZONES  # zw=10

SEEDS   = list(range(5))
T_ENCODE = 3000    # full encoding (past commitment epoch t*~2000)
T_ADV    = 2000    # adversarial phase
CPS_ADV  = list(range(100, T_ADV+1, 100))
CPS_ADV_SET = set(CPS_ADV)

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper44_results.json")

# Adversarial-phase mechanism table
# (All conditions share FULL adaptive encoding; only adversarial phase differs)
CONDITIONS = {
    'adaptive': {'no_gating': False, 'no_collapse': False, 'seed_beta': 0.25},
    'passive':  {'no_gating': True,  'no_collapse': False, 'seed_beta': 0.00},
    'rigid':    {'no_gating': False, 'no_collapse': True,  'seed_beta': 0.25},
}


# ── FastVCML (encoding phase -- always full mechanism) ────────────────────────
class FastVCML:
    """Standard FastVCML with configurable adversarial-phase mechanism."""
    def __init__(self, seed, fa=FA_STD):
        self.rng        = np.random.RandomState(seed)
        self.fa         = fa
        # Encoding-phase params (fixed, same for all conditions)
        self.SS         = 10
        self.MID_DECAY  = 0.99
        self.SEED_BETA  = 0.25
        self.no_gating  = False
        self.no_collapse = False
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

    def _gru(self, x):
        h=self.hid
        def sig(a): return 1/(1+np.exp(-np.clip(a,-8,8)))
        def tanh(a): e2=np.exp(2*np.clip(a,-8,8)); return (e2-1)/(e2+1)
        x3,h3=x[:,:,None],h[:,:,None]
        z=sig((self.Wz@x3).squeeze(-1)+(self.Uz@h3).squeeze(-1)+self.bz)
        r=sig((self.Wr@x3).squeeze(-1)+(self.Ur@h3).squeeze(-1)+self.br)
        rh=(r*h)[:,:,None]
        g=tanh((self.Wh@x3).squeeze(-1)+(self.Uh@rh).squeeze(-1)+self.bh)
        hn=(1-z)*h+z*g; out=np.tanh(np.einsum('ni,ni->n',self.Wo,hn)+self.bo)
        return hn, out

    def _diffuse(self):
        ns=np.maximum(self.nb,0); nf=self.fieldM[ns]
        nf*=(self.nb>=0)[:,:,None]
        nm=np.where(self.nbc[:,None]>0,nf.sum(1)/self.nbc[:,None],self.fieldM)
        self.fieldM+=DIFFUSE*(nm-self.fieldM)

    def step(self, wa, wc):
        nb=self._nbmean(); an=np.minimum(1.,self.age/300.)
        x=np.stack([nb,np.minimum(1.,wa),an],1)
        hn,out=self._gru(x); self.hid=hn
        self.vals=np.clip(VALS_DECAY*self.vals+VALS_NAV*nb+ADJ_SCALE*out,0,1)
        dev=hn-self.base_h; self.base_h+=BASE_BETA*dev
        self.streak=np.where(np.sum(dev**2,1)<.0025,self.streak+1,0)
        fa=self.fa
        self.mid=(self.mid+fa*dev)*self.MID_DECAY
        gate = np.ones(self.N, bool) if self.no_gating else (self.streak >= self.SS)
        self.fieldM[gate]+=fa*(self.mid[gate]-self.fieldM[gate])
        self.fieldM*=FIELD_DECAY
        if self.t%DIFFUSE_EVERY==0: self._diffuse()
        self.age+=1
        if not self.no_collapse:
            self._collapse()
        self.t+=1

    def _collapse(self):
        rng=self.rng
        bad=(self.vals<BOUND_LO)|(self.vals>BOUND_HI)
        inst=(np.sum(np.abs(self.hid-self.base_h),1)>INST_THRESH)&(rng.random(self.N)<INST_PROB)
        ci=np.where(bad|inst)[0]
        if not len(ci): return
        prev=self.hid.copy()
        sb=self.SEED_BETA
        for ai in ci:
            self.cc[ai]+=1; fm=self.fieldM[ai]; mag=np.sqrt(np.dot(fm,fm))
            if sb > 0 and mag > 1e-6:
                nh=(1-sb)*prev[ai]+sb*fm
            else:
                nh=prev[ai].copy()
            nh+=rng.randn(HS)*FRAG_NOISE
            self.hid[ai]=nh; self.vals[ai]=.5; self.age[ai]=0
            self.streak[ai]=0; self.mid[ai]=np.zeros(HS)

    def stable_zone(self, k): return np.argsort(self.cc)[:min(k,self.N//5)]

    def switch_mechanism(self, cond_name):
        """Switch to adversarial-phase mechanism at T_ENCODE."""
        p = CONDITIONS[cond_name]
        self.no_gating   = p['no_gating']
        self.no_collapse = p['no_collapse']
        self.SEED_BETA   = p['seed_beta']


# ── Wave environments ─────────────────────────────────────────────────────────
class WaveEnv:
    """Standard or adversarial (flipped) wave environment."""
    def __init__(self, vcml, WR, rng_seed=0, adversarial=False):
        self.vcml=vcml; self.WR=WR; self.adversarial=adversarial
        self.rng=random.Random(rng_seed)
        self.waves=[]
        N=vcml.N
        self._wa=np.empty(N); self._wc=np.empty(N,int)
        self.ax=vcml.ai_x; self.ay=vcml.ai_y

    def _launch(self):
        cls = self.rng.randint(0, N_ZONES-1)
        zlo = cls*ZW; zhi = (cls+1)*ZW-1
        shifted_x = self.rng.randint(zlo, zhi)
        cx = HALF + shifted_x; cy = self.rng.randint(0, H-1)
        self.waves.append([cx, cy, WAVE_DUR, cls])

    def _apply(self):
        wa=self._wa; wc=self._wc
        wa.fill(0.); wc.fill(-1)
        ax=self.ax; ay=self.ay; surv=[]
        for wave in self.waves:
            cx,cy,rem,cls=wave
            if rem<=0: continue
            dist=np.abs(ax-cx)+np.abs(ay-cy)
            act=np.maximum(0.,1.-dist*0.4); act[dist>2]=0.
            better=act>wa; wa[better]=act[better]; wc[better]=cls
            wave[2]-=1
            if wave[2]>0: surv.append(wave)
        self.waves=surv

        if not self.adversarial:
            # Normal: even zones suppressed, odd zones excited
            s_amp, e_amp = SUPP_AMP, EXC_AMP
        else:
            # Adversarial: even zones excited, odd zones suppressed (FLIPPED)
            s_amp, e_amp = EXC_AMP_ADV, SUPP_AMP_ADV  # also half amplitude

        for even in [True, False]:
            if not self.adversarial:
                amp = s_amp if even else e_amp
                suppress = even
            else:
                amp = e_amp if even else s_amp   # flipped role
                suppress = not even               # flipped direction
            par = (wc%2==0) if even else (wc%2==1)
            idx = np.where((wc>=0)&par&(wa>.05))[0]
            if not len(idx): continue
            sc = wa[idx]*amp
            if suppress:
                self.vcml.vals[idx]=np.maximum(0., self.vcml.vals[idx]*(1.-sc*.5))
            else:
                self.vcml.vals[idx]=np.minimum(1., self.vcml.vals[idx]+sc*.5)
        return wa.copy(), wc.copy()

    def step(self):
        exp=self.WR/WAVE_DUR
        nl=int(exp)+(1 if self.rng.random()<exp-int(exp) else 0)
        for _ in range(nl): self._launch()
        return self._apply()


# ── Metrics ───────────────────────────────────────────────────────────────────
def get_zone_means(vcml):
    si = vcml.stable_zone(ZONE_K)
    xp = vcml.ai_x[si]
    za = np.minimum(N_ZONES-1, (xp - HALF) // ZW)
    means = []
    for z in range(N_ZONES):
        zs = si[za==z]
        means.append(vcml.fieldM[zs].mean(0) if len(zs) else np.zeros(HS))
    return means

def cosine_fidelity(current_means, baseline_means):
    """Mean cosine similarity; 1=intact, 0=orthogonal, -1=inverted."""
    sims = []
    for cm, bm in zip(current_means, baseline_means):
        mc=np.linalg.norm(cm); mb=np.linalg.norm(bm)
        if mc > 1e-8 and mb > 1e-8:
            sims.append(float(np.dot(cm,bm)/(mc*mb)))
    return float(np.mean(sims)) if sims else 0.0

def compute_sg4(vcml):
    means = get_zone_means(vcml)
    dists = [float(np.linalg.norm(np.array(means[a])-np.array(means[b])))
             for a in range(N_ZONES) for b in range(a+1,N_ZONES)]
    return float(np.mean(dists)) if dists else 0.0


# ── Main experiment ───────────────────────────────────────────────────────────
def run_adversarial(seed, cond_name):
    # Encoding: full mechanism, identical for all conditions
    vcml = FastVCML(seed=seed*7+11)
    enc_env = WaveEnv(vcml, WR_FIXED, rng_seed=seed*200+37, adversarial=False)

    for t in range(T_ENCODE):
        wa, wc = enc_env.step()
        vcml.step(wa, wc)

    # Snapshot at end of encoding (same for all conditions)
    baseline_means = get_zone_means(vcml)
    sg4_at_encode  = compute_sg4(vcml)
    cc_at_encode   = int(np.sum(vcml.cc))

    # Switch to condition-specific mechanism
    vcml.switch_mechanism(cond_name)

    # Adversarial phase
    adv_env = WaveEnv(vcml, WR_FIXED, rng_seed=seed*200+99, adversarial=True)

    fidelity_traj = []
    sg4_traj      = []
    ts_adv        = []
    first_cross_half = None  # first t where fidelity < 0.5

    for t in range(T_ADV):
        wa, wc = adv_env.step()
        vcml.step(wa, wc)
        if (t+1) in CPS_ADV_SET:
            current_means = get_zone_means(vcml)
            fid = cosine_fidelity(current_means, baseline_means)
            fidelity_traj.append(fid)
            sg4_traj.append(compute_sg4(vcml))
            ts_adv.append(t+1)
            if first_cross_half is None and fid < 0.5:
                first_cross_half = t+1

    return {
        'cond': cond_name, 'seed': seed,
        'sg4_at_encode': sg4_at_encode,
        'cc_at_encode':  cc_at_encode,
        'cc_total':      int(np.sum(vcml.cc)),
        'ts_adv':        ts_adv,
        'fidelity':      fidelity_traj,
        'sg4':           sg4_traj,
        'first_cross_half': first_cross_half,
    }


# ── Multiprocessing ───────────────────────────────────────────────────────────
def _worker(args): seed, cond = args; return run_adversarial(seed, cond)
def key(seed, cond): return f"{cond},{seed}"

if __name__ == '__main__':
    mp.freeze_support()
    existing = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for r in json.load(f):
                existing[key(r['seed'], r['cond'])] = r
        print(f"Loaded {len(existing)} existing results.")

    todo = [(s,c) for c in CONDITIONS for s in SEEDS if key(s,c) not in existing]
    print(f"Running {len(todo)} experiments "
          f"({len(CONDITIONS)} conditions x {len(SEEDS)} seeds, T={T_ENCODE+T_ADV})...")

    if todo:
        with mp.Pool(processes=min(len(todo), mp.cpu_count())) as pool:
            for r in pool.map(_worker, todo):
                existing[key(r['seed'],r['cond'])] = r

    all_results = list(existing.values())
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE,'w') as f: json.dump(all_results, f)
    print(f"Saved {len(all_results)} results.")

    # Summary table
    print(f"\n{'Condition':<12} {'sg4_encode':<12} {'Fid@200':<10} "
          f"{'Fid@800':<10} {'Fid@2000':<10} {'t_half':<10}")
    for cond in CONDITIONS:
        rs = [r for r in all_results if r['cond']==cond]
        if not rs: continue
        def fid_at(t_tgt):
            v=[r['fidelity'][r['ts_adv'].index(t_tgt)]
               for r in rs if t_tgt in r['ts_adv']]
            return float(np.mean(v)) if v else float('nan')
        sg4s = np.mean([r['sg4_at_encode'] for r in rs])
        th   = [r['first_cross_half'] for r in rs if r['first_cross_half']]
        th_s = f"{np.mean(th):.0f}" if th else ">2000"
        print(f"{cond:<12} {sg4s:<12.4f} {fid_at(200):<10.4f} "
              f"{fid_at(800):<10.4f} {fid_at(2000):<10.4f} {th_s:<10}")
