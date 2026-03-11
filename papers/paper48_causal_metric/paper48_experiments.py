"""
Paper 48: sg_C -- The Causal Structure Metric

Motivation: Papers 46-47 established C2 causal reparametrisation (Omega, delta,
beta_s) as the operative description of the system. But the primary metric sg4
measures SPATIAL differentiation: mean pairwise L2 between spatially-defined
zone-mean fieldM vectors. When causality != spatiality, sg4 gives wrong answers.

Two cases where they decouple (found in Papers 46-47):
  (1) rand_p60 vs ss10: sg4(rand) > sg4(ss10) by 1.68x. But random writes
      include perturbation-onset signal -- spatially broad, causally noisy.
      Does the quiescence gate actually protect CAUSAL structure despite
      reducing spatial amplitude?
  (2) Delta sweep: sg4 increases as N_ZONES increases (more pairs, more variance).
      But na_ratio told us encoding quality degrades at delta<=2.5.
      Does a causal metric correctly track the degradation?

The causal structure metric sg_C (decode_acc):
  For each cell, compute its nearest-centroid zone assignment in fieldM space.
  sg_C = fraction of cells correctly classified to their home zone.
  Chance = 1/N_ZONES. sg_C >> chance = causal structure present.

  This asks: does each cell's fieldM encode WHICH ZONE'S waves hit it?
  It is invariant to spatial arrangement, works on any substrate.

Three complementary measures:
  sg4:         between-zone spread (spatial amplitude, existing metric)
  sigma_w:     within-zone fieldM spread (spatial noise)
  decode_acc:  fraction correctly classified (sg_C, causal metric)

Relationship: decode_acc = f(sg4 / sigma_w). Same signal encoded in all three
but decode_acc is not contaminated by within-zone noise.

Two experiments:

Exp A (30 runs): Gate sweep -- does sg_C flip the Paper 46 result?
  N_ZONES=4, SS in {0,5,10,15,20,rand_p60} x 5 seeds = 30 runs, STEPS=3000.
  Prediction: rand_p60 has higher sg4 but LOWER sg_C than ss10.
  If confirmed: the quiescence gate protects causal structure; Paper 46's
  "unexpected" result was a measurement artifact of using sg4.

Exp B (25 runs): Delta sweep under sg_C -- does sg_C correctly track degradation?
  N_ZONES in {2,4,5,8,10} x SS=10 x 5 seeds = 25 runs.
  (N_ZONES=4 overlaps with Exp A; key comparison is N_ZONES=2,5,8,10.)
  Prediction: sg4 increases (more pairs), sg_C decreases (bleed degrades
  causal classification). sg_C tracks Paper 47's na_ratio finding correctly.

Total: 50 runs (5 non-overlapping from N_ZONES=4,SS=10 shared).
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
BOUND_LO=0.05; BOUND_HI=0.95; FRAG_NOISE=0.01; SEED_BETA=0.25
INST_THRESH=0.45; INST_PROB=0.03; DIFFUSE=0.02; DIFFUSE_EVERY=2

SEEDS        = list(range(5))
WR_STD       = 4.8
R_WAVE_STD   = 2
STEPS        = 3000
WARMUP       = 300
SAMPLE_EVERY = 20
TAIL_STEPS   = 30

# Exp A
A_GATE_CONDITIONS = {
    'ss0':      {'ss': 0,   'rand_gate': False, 'p_write': None},
    'ss5':      {'ss': 5,   'rand_gate': False, 'p_write': None},
    'ss10':     {'ss': 10,  'rand_gate': False, 'p_write': None},
    'ss15':     {'ss': 15,  'rand_gate': False, 'p_write': None},
    'ss20':     {'ss': 20,  'rand_gate': False, 'p_write': None},
    'rand_p60': {'ss': 10,  'rand_gate': True,  'p_write': 0.60},
}

# Exp B
N_ZONES_SWEEP = [2, 4, 5, 8, 10]

RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper48_results.json")


# ── FastVCML ──────────────────────────────────────────────────────────────────
class FastVCML:
    def __init__(self, seed, ss=10, rand_gate=False, p_write=None):
        self.rng       = np.random.RandomState(seed)
        self.SS        = ss
        self.rand_gate = rand_gate
        self.p_write   = p_write if p_write is not None else 0.60
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
        self.n_writes = 0
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
        if self.rand_gate:
            gate = self.rng.random(self.N) < self.p_write
        elif self.SS == 0:
            gate = np.ones(self.N, bool)
        else:
            gate = self.streak >= self.SS
        self.fieldM[gate] += FA*(self.mid[gate]-self.fieldM[gate])
        self.n_writes += int(gate.sum())
        self.fieldM *= FIELD_DECAY
        if self.t % DIFFUSE_EVERY == 0: self._diffuse()
        self.age += 1
        self._collapse()
        self.t += 1

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


# ── Wave environment ───────────────────────────────────────────────────────────
class WaveEnv:
    def __init__(self, vcml, WR, rng_seed=0, r_wave=2, n_zones=4):
        self.vcml=vcml; self.WR=WR; self.r_wave=r_wave
        self.n_zones=n_zones; self.zw=HALF//n_zones
        self.rng=random.Random(rng_seed)
        self.waves=[]
        N=vcml.N
        self._wa=np.empty(N); self._wc=np.empty(N,int)
        self.ax=vcml.ai_x; self.ay=vcml.ai_y

    def _launch(self):
        cls = self.rng.randint(0, self.n_zones-1)
        zlo = cls*self.zw; zhi = (cls+1)*self.zw-1
        sx  = self.rng.randint(zlo, zhi)
        cx  = HALF + sx; cy = self.rng.randint(0, H-1)
        self.waves.append([cx, cy, WAVE_DUR, cls])

    def _apply(self):
        wa=self._wa; wc=self._wc
        wa.fill(0.); wc.fill(-1)
        ax=self.ax; ay=self.ay; rw=self.r_wave; surv=[]
        for wave in self.waves:
            cx,cy,rem,cls=wave
            if rem<=0: continue
            dist=np.abs(ax-cx)+np.abs(ay-cy)
            act=np.maximum(0.,1.-dist/rw); act[dist>rw]=0.
            better=act>wa; wa[better]=act[better]; wc[better]=cls
            wave[2]-=1
            if wave[2]>0: surv.append(wave)
        self.waves=surv
        for even in [True, False]:
            amp = SUPP_AMP if even else EXC_AMP
            suppress = even
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


# ── Metrics ────────────────────────────────────────────────────────────────────
def _zone_labels_and_centroids(vcml, n_zones):
    """Returns (true_zones array, centroids array shape (n_zones, HS))."""
    zw = HALF // n_zones
    true_zones = np.minimum(n_zones-1, (vcml.ai_x - HALF) // zw)
    centroids = np.array([
        vcml.fieldM[true_zones==z].mean(0) if (true_zones==z).any() else np.zeros(HS)
        for z in range(n_zones)
    ])
    return true_zones, centroids

def sg4_fn(vcml, n_zones=4):
    _, centroids = _zone_labels_and_centroids(vcml, n_zones)
    dists = [float(np.linalg.norm(centroids[i]-centroids[j]))
             for i in range(n_zones) for j in range(i+1, n_zones)]
    return float(np.mean(dists)) if dists else 0.0

def decode_acc_fn(vcml, n_zones=4):
    """sg_C: nearest-centroid zone decode accuracy.
    Asks: does each cell's fieldM encode which zone's waves hit it?
    Chance = 1/n_zones. Returns raw accuracy (not normalized).
    """
    true_zones, centroids = _zone_labels_and_centroids(vcml, n_zones)
    # (N, n_zones) distance matrix
    dists = np.linalg.norm(
        vcml.fieldM[:, None, :] - centroids[None, :, :], axis=2
    )
    pred_zones = np.argmin(dists, axis=1)
    return float((pred_zones == true_zones).mean())

def sigma_within_fn(vcml, n_zones=4):
    """Mean within-zone standard deviation of fieldM vectors."""
    zw = HALF // n_zones
    spreads = []
    for z in range(n_zones):
        xlo = HALF + z*zw; xhi = HALF + (z+1)*zw
        mask = (vcml.ai_x >= xlo) & (vcml.ai_x < xhi)
        fm = vcml.fieldM[mask]
        if len(fm) > 1:
            spreads.append(float(np.std(fm)))
    return float(np.mean(spreads)) if spreads else float('nan')

def nonadj_adj_fn(vcml, n_zones=4):
    _, centroids = _zone_labels_and_centroids(vcml, n_zones)
    adj=[]; nonadj=[]
    for i in range(n_zones):
        for j in range(i+1, n_zones):
            d = float(np.linalg.norm(centroids[i]-centroids[j]))
            if j-i==1: adj.append(d)
            else:       nonadj.append(d)
    if not adj or not nonadj: return float('nan')
    return float(np.mean(nonadj)/np.mean(adj))


# ── Run function ───────────────────────────────────────────────────────────────
def run(seed, ss=10, rand_gate=False, p_write=None, n_zones=4):
    vcml = FastVCML(seed, ss=ss, rand_gate=rand_gate, p_write=p_write)
    env  = WaveEnv(vcml, WR=WR_STD, rng_seed=seed+1000, r_wave=R_WAVE_STD,
                   n_zones=n_zones)
    sg4_trace = []
    dec_trace = []
    for t in range(STEPS):
        wa, wc = env.step()
        vcml.step(wa, wc)
        if t >= WARMUP and t % SAMPLE_EVERY == 0:
            sg4_trace.append(sg4_fn(vcml, n_zones))
            dec_trace.append(decode_acc_fn(vcml, n_zones))
    sg4_tail = float(np.mean(sg4_trace[-TAIL_STEPS:])) if sg4_trace else float('nan')
    dec_tail = float(np.mean(dec_trace[-TAIL_STEPS:])) if dec_trace else float('nan')
    sw       = sigma_within_fn(vcml, n_zones)
    chance   = 1.0 / n_zones
    return {
        'sg4':           sg4_tail,
        'decode_acc':    dec_tail,
        'decode_norm':   float((dec_tail - chance) / (1.0 - chance)) if dec_tail == dec_tail else float('nan'),
        'sigma_within':  sw,
        'snr':           float(sg4_tail / sw) if sw > 0 else float('nan'),
        'na_ratio':      nonadj_adj_fn(vcml, n_zones),
        'write_rate':    float(vcml.n_writes) / (vcml.N * STEPS),
        'coll_site':     float(vcml.cc.sum()) / vcml.N,
    }


# ── Worker ─────────────────────────────────────────────────────────────────────
def _worker(args):
    tag, seed, params = args
    return tag, seed, run(seed, **params)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    todo = []

    # Exp A: gate sweep at N_ZONES=4
    for cond_name, cparams in A_GATE_CONDITIONS.items():
        for seed in SEEDS:
            key = f"A,{cond_name},{seed}"
            if key not in results:
                todo.append(('A', seed, {
                    'ss': cparams['ss'], 'rand_gate': cparams['rand_gate'],
                    'p_write': cparams['p_write'], 'n_zones': 4,
                }, key))

    # Exp B: delta sweep at SS=10 (N_ZONES=4 overlaps with Exp A ss10)
    for n_zones in N_ZONES_SWEEP:
        for seed in SEEDS:
            key = f"B,nz{n_zones},{seed}"
            if key not in results:
                todo.append(('B', seed, {
                    'ss': 10, 'rand_gate': False, 'p_write': None,
                    'n_zones': n_zones,
                }, key))

    print(f"Total tasks: {len(todo)} (skipping {len(results)} cached)")

    if todo:
        worker_args = [(t[0], t[1], t[2]) for t in todo]
        with mp.Pool(processes=min(len(todo), mp.cpu_count())) as pool:
            raw = pool.map(_worker, worker_args)
        for (_, _, _, key), (_, _, result) in zip(todo, raw):
            results[key] = result
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} results -> {RESULTS_FILE}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\n=== Exp A: Gate sweep -- sg4 vs sg_C ===")
    print(f"N_ZONES=4, chance=0.250")
    print(f"\n{'Condition':<12} {'sg4':>8} {'sg4 SE':>7} {'decode_acc':>11} {'dec SE':>7} "
          f"{'dec_norm':>9} {'sigma_w':>8} {'SNR':>7}")

    a_summary = {}
    for cond_name in A_GATE_CONDITIONS:
        sg4s  = [results[f"A,{cond_name},{s}"]['sg4']
                 for s in SEEDS if f"A,{cond_name},{s}" in results]
        decs  = [results[f"A,{cond_name},{s}"]['decode_acc']
                 for s in SEEDS if f"A,{cond_name},{s}" in results]
        dnorm = [results[f"A,{cond_name},{s}"]['decode_norm']
                 for s in SEEDS if f"A,{cond_name},{s}" in results]
        sws   = [results[f"A,{cond_name},{s}"]['sigma_within']
                 for s in SEEDS if f"A,{cond_name},{s}" in results]
        snrs  = [results[f"A,{cond_name},{s}"]['snr']
                 for s in SEEDS if f"A,{cond_name},{s}" in results]
        if not sg4s: continue
        m4=np.mean(sg4s); se4=np.std(sg4s)/np.sqrt(len(sg4s))
        md=np.mean(decs); sed=np.std(decs)/np.sqrt(len(decs))
        mdn=np.mean(dnorm); msw=np.mean(sws); msnr=np.mean(snrs)
        a_summary[cond_name] = {
            'sg4_mean': float(m4), 'sg4_se': float(se4),
            'sg4_vals': [float(v) for v in sg4s],
            'decode_mean': float(md), 'decode_se': float(sed),
            'decode_vals': [float(v) for v in decs],
            'decode_norm': float(mdn),
            'sigma_within': float(msw), 'snr': float(msnr),
        }
        print(f"{cond_name:<12} {m4:>8.4f} {se4:>7.4f} {md:>11.4f} {sed:>7.4f} "
              f"{mdn:>9.3f} {msw:>8.4f} {msnr:>7.3f}")

    # Key comparison: rank order of sg4 vs decode_acc
    cond_order = list(A_GATE_CONDITIONS.keys())
    sg4_ranks  = sorted(cond_order, key=lambda c: -a_summary.get(c,{}).get('sg4_mean',0))
    dec_ranks  = sorted(cond_order, key=lambda c: -a_summary.get(c,{}).get('decode_mean',0))
    print(f"\nsg4 rank (high->low): {' > '.join(sg4_ranks)}")
    print(f"sg_C rank (high->low): {' > '.join(dec_ranks)}")
    if sg4_ranks != dec_ranks:
        print("-> METRICS DISAGREE: causal structure != spatial amplitude")
    else:
        print("-> Metrics agree on rank ordering")

    # Does the gate flip?
    ss10_dec  = a_summary.get('ss10',  {}).get('decode_mean', float('nan'))
    rand_dec  = a_summary.get('rand_p60',{}).get('decode_mean', float('nan'))
    ss10_sg4  = a_summary.get('ss10',  {}).get('sg4_mean', float('nan'))
    rand_sg4  = a_summary.get('rand_p60',{}).get('sg4_mean', float('nan'))
    print(f"\nKey comparison (ss10 vs rand_p60):")
    print(f"  sg4:     ss10={ss10_sg4:.4f}  rand_p60={rand_sg4:.4f}  "
          f"ratio={rand_sg4/ss10_sg4:.3f}x  (rand {'>' if rand_sg4>ss10_sg4 else '<'} ss10)")
    print(f"  decode:  ss10={ss10_dec:.4f}  rand_p60={rand_dec:.4f}  "
          f"ratio={rand_dec/ss10_dec:.3f}x  (rand {'>' if rand_dec>ss10_dec else '<'} ss10)")
    if rand_sg4 > ss10_sg4 and ss10_dec > rand_dec:
        print("  -> FLIP CONFIRMED: gate protects causal structure, sg4 was misleading")
    elif rand_sg4 > ss10_sg4 and rand_dec > ss10_dec:
        print("  -> No flip: rand_p60 beats ss10 on both metrics")
    else:
        print("  -> Mixed result")

    print("\n=== Exp B: Delta sweep -- sg4 vs sg_C ===")
    print(f"\n{'N_ZONES':>8} {'delta':>7} {'sg4':>8} {'sg4 SE':>7} "
          f"{'decode_acc':>11} {'dec_norm':>9} {'na_ratio':>10}")

    b_summary = {}
    for n_zones in N_ZONES_SWEEP:
        zw    = HALF // n_zones
        delta = zw / R_WAVE_STD
        chance= 1.0 / n_zones
        sg4s  = [results[f"B,nz{n_zones},{s}"]['sg4']
                 for s in SEEDS if f"B,nz{n_zones},{s}" in results]
        decs  = [results[f"B,nz{n_zones},{s}"]['decode_acc']
                 for s in SEEDS if f"B,nz{n_zones},{s}" in results]
        dnorm = [results[f"B,nz{n_zones},{s}"]['decode_norm']
                 for s in SEEDS if f"B,nz{n_zones},{s}" in results]
        nas   = [results[f"B,nz{n_zones},{s}"]['na_ratio']
                 for s in SEEDS if f"B,nz{n_zones},{s}" in results]
        if not sg4s: continue
        m4=np.mean(sg4s); se4=np.std(sg4s)/np.sqrt(len(sg4s))
        md=np.mean(decs); mdn=np.mean(dnorm)
        na_m=float(np.nanmean(nas)) if nas else float('nan')
        b_summary[n_zones] = {
            'delta': delta, 'sg4_mean': float(m4), 'sg4_se': float(se4),
            'sg4_vals': [float(v) for v in sg4s],
            'decode_mean': float(md), 'decode_norm': float(mdn),
            'decode_vals': [float(v) for v in decs],
            'na_ratio': na_m,
        }
        print(f"{n_zones:>8} {delta:>7.1f} {m4:>8.4f} {se4:>7.4f} "
              f"{md:>11.4f} {mdn:>9.3f} {na_m:>10.3f}")

    # sg4 vs sg_C trend for delta sweep
    nz_vals  = [nz for nz in N_ZONES_SWEEP if nz in b_summary]
    sg4_trend = [b_summary[nz]['sg4_mean'] for nz in nz_vals]
    dec_trend = [b_summary[nz]['decode_mean'] for nz in nz_vals]
    sg4_mono = all(sg4_trend[i] <= sg4_trend[i+1] for i in range(len(sg4_trend)-1))
    dec_mono = all(dec_trend[i] >= dec_trend[i+1] for i in range(len(dec_trend)-1))
    print(f"\nsg4 monotone increasing with N_ZONES (spatial artifact): {sg4_mono}")
    print(f"sg_C monotone decreasing with N_ZONES (causal degradation): {dec_mono}")
    if sg4_mono and dec_mono:
        print("-> PERFECT DISSOCIATION: sg4 and sg_C move in opposite directions")
    elif not sg4_mono and dec_mono:
        print("-> sg_C tracks causal degradation; sg4 does not monotonically increase")

    # Save analysis
    analysis = {'exp_a': a_summary, 'exp_b': {str(k): v for k,v in b_summary.items()}}
    analysis_file = os.path.join(os.path.dirname(__file__),
                                 "results", "paper48_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis -> {analysis_file}")
