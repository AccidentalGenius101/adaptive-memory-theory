"""
Paper 47: C2 Coordinate Validation

Two experiments:

Exp A (25 runs): delta-sweep -- third C2 coordinate.
  delta = W_zone / r_wave = (HALF/N_ZONES) / r_wave, with r_wave=2 fixed.
  N_ZONES in {2, 4, 5, 8, 10} -> delta in {10, 5, 4, 2.5, 2}.
  Fixed Omega = WR * phi_w(r_wave) = 4.8 * 13 = 62.4.
  5 seeds x 5 zone configs = 25 runs, STEPS=3000.

  Prediction: sg4 peaks at delta~4-5 (zone wide enough for copy-forward loop
    to build coherent gradient without bleed). Collapses at delta<=2 (zone
    too narrow -- wave footprint overlaps neighboring zone).

Exp B (120 runs): SS x adversarial pressure -- gate selectivity trade-off.
  Phase 1 [T=0..T_ENCODE=2000]: Full VCML, SS=10, standard waves.
    All conditions identical. Builds zone structure past commitment epoch.
  At T_ENCODE: switch gate to condition-specific SS or rand_p60.
  Phase 2 [T_ENCODE..+T_ADV=1000]: Adversarial waves (flipped polarity,
    scaled by adv_amp) + condition-specific gate.
  adv_amp in {0.0, 0.125, 0.25, 0.50} x SS in {0,5,10,15,20,rand_p60} x
    5 seeds = 120 runs.
  Metric: final fidelity = mean cosine similarity of zone means at end of
    adversarial phase vs snapshot at T_ENCODE.

  Adversarial filter hypothesis (Paper 46): higher SS is a stronger filter
    against writing adversarially-driven mid_mem.
  Prediction: optimal SS increases linearly with adv_amp. At adv_amp=0,
    all conditions maintain fidelity. At high adv_amp, high SS protects,
    ss0 imprints adversarial pattern.
"""
import numpy as np, json, os, math, random, multiprocessing as mp
from scipy import stats as _stats

# ── Fixed constants (identical to Papers 43-46) ──────────────────────────────
W=80; H=40; HALF=40
HS=2; IS=3
WAVE_DUR=15
SUPP_AMP=0.25; EXC_AMP=0.50
MID_DECAY=0.99; FIELD_DECAY=0.9997; BASE_BETA=0.005
FA=0.16
VALS_DECAY=0.92; VALS_NAV=0.08; ADJ_SCALE=0.03
BOUND_LO=0.05; BOUND_HI=0.95; FRAG_NOISE=0.01; SEED_BETA=0.25
INST_THRESH=0.45; INST_PROB=0.03; DIFFUSE=0.02; DIFFUSE_EVERY=2

SEEDS = list(range(5))

# Standard wave params
WR_STD     = 4.8
R_WAVE_STD = 2
def phi_w(r): return 2*r*r + 2*r + 1
OMEGA_STD  = WR_STD * phi_w(R_WAVE_STD)   # 4.8 * 13 = 62.4

# ── Exp A ─────────────────────────────────────────────────────────────────────
N_ZONES_SWEEP = [2, 4, 5, 8, 10]    # delta = (HALF/NZ)/2 = {10,5,4,2.5,2}
STEPS_A      = 3000
WARMUP_A     = 300
SAMPLE_EVERY = 20
TAIL_STEPS   = 30

# ── Exp B ─────────────────────────────────────────────────────────────────────
T_ENCODE = 2000    # standard encoding phase (all conditions)
T_ADV    = 1000    # adversarial phase
CPS_ADV  = list(range(100, T_ADV+1, 100))   # 10 checkpoints
CPS_ADV_SET = set(CPS_ADV)

ADV_AMPS = [0.0, 0.125, 0.25, 0.50]   # scale on encoding amplitudes

B_GATE_CONDITIONS = {
    'ss0':      {'ss': 0,   'rand_gate': False, 'p_write': None},
    'ss5':      {'ss': 5,   'rand_gate': False, 'p_write': None},
    'ss10':     {'ss': 10,  'rand_gate': False, 'p_write': None},
    'ss15':     {'ss': 15,  'rand_gate': False, 'p_write': None},
    'ss20':     {'ss': 20,  'rand_gate': False, 'p_write': None},
    'rand_p60': {'ss': 10,  'rand_gate': True,  'p_write': 0.60},  # ss ignored
}

RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper47_results.json")


# ── FastVCML (unified: variable SS, rand_gate, supports phase switch) ─────────
class FastVCML:
    def __init__(self, seed, ss=10, rand_gate=False, p_write=None):
        self.rng        = np.random.RandomState(seed)
        self.SS         = ss
        self.rand_gate  = rand_gate
        self.p_write    = p_write if p_write is not None else 0.60
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

    def switch_gate(self, ss, rand_gate, p_write):
        """Switch gate for Exp B phase transition."""
        self.SS        = ss
        self.rand_gate = rand_gate
        if p_write is not None:
            self.p_write = p_write

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


# ── Wave environment (variable n_zones, r_wave, adversarial amp) ──────────────
class WaveEnv:
    def __init__(self, vcml, WR, rng_seed=0, r_wave=2, n_zones=4,
                 adversarial=False, adv_amp=0.0):
        self.vcml       = vcml
        self.WR         = WR
        self.r_wave     = r_wave
        self.n_zones    = n_zones
        self.zw         = HALF // n_zones
        self.adversarial = adversarial
        self.adv_amp    = adv_amp
        self.rng        = random.Random(rng_seed)
        self.waves      = []
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

        if not self.adversarial:
            # Normal: even zones suppressed, odd zones excited
            for even in [True, False]:
                amp      = SUPP_AMP if even else EXC_AMP
                suppress = even
                par = (wc%2==0) if even else (wc%2==1)
                idx = np.where((wc>=0)&par&(wa>.05))[0]
                if not len(idx): continue
                sc = wa[idx]*amp
                if suppress:
                    self.vcml.vals[idx]=np.maximum(0., self.vcml.vals[idx]*(1.-sc*.5))
                else:
                    self.vcml.vals[idx]=np.minimum(1., self.vcml.vals[idx]+sc*.5)
        else:
            # Adversarial: flipped polarity, amplitudes scaled by adv_amp.
            # even zones: now EXCITED (was suppressed) at EXC_AMP*adv_amp
            # odd zones:  now SUPPRESSED (was excited) at SUPP_AMP*adv_amp
            s = self.adv_amp
            for even in [True, False]:
                amp      = EXC_AMP*s if even else SUPP_AMP*s
                suppress = not even   # flipped direction
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
def get_zone_means(vcml, n_zones):
    zw = HALF // n_zones
    zmeans = []
    for z in range(n_zones):
        xlo = HALF + z*zw; xhi = HALF + (z+1)*zw
        mask = (vcml.ai_x >= xlo) & (vcml.ai_x < xhi)
        fm = vcml.fieldM[mask]
        zmeans.append(fm.mean(0) if len(fm) else np.zeros(HS))
    return zmeans

def sg4_fn(vcml, n_zones=4):
    zmeans = get_zone_means(vcml, n_zones)
    dists = []
    for i in range(n_zones):
        for j in range(i+1, n_zones):
            dists.append(float(np.linalg.norm(zmeans[i]-zmeans[j])))
    return float(np.mean(dists)) if dists else 0.0

def nonadj_adj_fn(vcml, n_zones=4):
    zmeans = get_zone_means(vcml, n_zones)
    adj=[]; nonadj=[]
    for i in range(n_zones):
        for j in range(i+1, n_zones):
            d = float(np.linalg.norm(zmeans[i]-zmeans[j]))
            if j-i==1: adj.append(d)
            else:       nonadj.append(d)
    if not adj or not nonadj: return float('nan')
    return float(np.mean(nonadj)/np.mean(adj))

def cosine_fidelity(current_means, baseline_means):
    sims=[]
    for cm,bm in zip(current_means, baseline_means):
        mc=np.linalg.norm(cm); mb=np.linalg.norm(bm)
        if mc>1e-8 and mb>1e-8:
            sims.append(float(np.dot(cm,bm)/(mc*mb)))
    return float(np.mean(sims)) if sims else 0.0


# ── Run functions ─────────────────────────────────────────────────────────────
def run_exp_a(seed, n_zones):
    """Exp A: delta-sweep. Standard VCML, variable n_zones."""
    vcml = FastVCML(seed, ss=10)
    env  = WaveEnv(vcml, WR=WR_STD, rng_seed=seed+1000, r_wave=R_WAVE_STD,
                   n_zones=n_zones)
    sg4_trace = []
    for t in range(STEPS_A):
        wa, wc = env.step()
        vcml.step(wa, wc)
        if t >= WARMUP_A and t % SAMPLE_EVERY == 0:
            sg4_trace.append(sg4_fn(vcml, n_zones))
    sg4_tail = float(np.mean(sg4_trace[-TAIL_STEPS:])) if sg4_trace else float('nan')
    return {
        'sg4':       sg4_tail,
        'sg4_trace': [float(v) for v in sg4_trace[-TAIL_STEPS:]],
        'coll_site': float(vcml.cc.sum()) / vcml.N,
        'na_ratio':  nonadj_adj_fn(vcml, n_zones),
    }


def run_exp_b(seed, adv_amp, ss, rand_gate, p_write):
    """Exp B: two-phase run. Phase1=standard encoding, Phase2=gate x adversarial."""
    # Phase 1: standard encoding (SS=10 for all conditions)
    vcml    = FastVCML(seed, ss=10, rand_gate=False)
    enc_env = WaveEnv(vcml, WR=WR_STD, rng_seed=seed+2000, r_wave=R_WAVE_STD,
                      n_zones=4, adversarial=False)
    for _ in range(T_ENCODE):
        wa, wc = enc_env.step()
        vcml.step(wa, wc)

    # Snapshot at end of encoding
    baseline_means = get_zone_means(vcml, n_zones=4)
    sg4_at_encode  = sg4_fn(vcml, n_zones=4)

    # Switch to condition-specific gate
    vcml.switch_gate(ss=ss, rand_gate=rand_gate, p_write=p_write)

    # Phase 2: adversarial waves (or normal if adv_amp=0)
    adv_env = WaveEnv(vcml, WR=WR_STD, rng_seed=seed+3000, r_wave=R_WAVE_STD,
                      n_zones=4, adversarial=(adv_amp > 0), adv_amp=adv_amp)

    fidelity_traj   = []
    ts_adv          = []
    first_cross_half = None

    for t in range(T_ADV):
        wa, wc = adv_env.step()
        vcml.step(wa, wc)
        if (t+1) in CPS_ADV_SET:
            cur = get_zone_means(vcml, n_zones=4)
            fid = cosine_fidelity(cur, baseline_means)
            fidelity_traj.append(float(fid))
            ts_adv.append(t+1)
            if first_cross_half is None and fid < 0.5:
                first_cross_half = t+1

    final_fid = fidelity_traj[-1] if fidelity_traj else float('nan')
    return {
        'sg4_at_encode':    float(sg4_at_encode),
        'final_fidelity':   float(final_fid),
        'fidelity_traj':    fidelity_traj,
        'ts_adv':           ts_adv,
        'first_cross_half': first_cross_half,
        'coll_site':        float(vcml.cc.sum()) / vcml.N,
    }


# ── Worker ────────────────────────────────────────────────────────────────────
def _worker(args):
    tag, seed, params = args
    if tag == 'A':
        result = run_exp_a(seed, **params)
    else:
        result = run_exp_b(seed, **params)
    return tag, seed, result


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    todo = []

    # Exp A: delta-sweep
    for n_zones in N_ZONES_SWEEP:
        for seed in SEEDS:
            key = f"A,nz{n_zones},{seed}"
            if key not in results:
                todo.append(('A', seed, {'n_zones': n_zones}, key))

    # Exp B: gate x adversarial amplitude
    for adv_amp in ADV_AMPS:
        for cond_name, cparams in B_GATE_CONDITIONS.items():
            for seed in SEEDS:
                key = f"B,amp{adv_amp:.3f},{cond_name},{seed}"
                if key not in results:
                    params = {
                        'adv_amp':   adv_amp,
                        'ss':        cparams['ss'],
                        'rand_gate': cparams['rand_gate'],
                        'p_write':   cparams['p_write'],
                    }
                    todo.append(('B', seed, params, key))

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
    print("\n=== Exp A: delta-sweep (third C2 coordinate) ===")
    print(f"Omega = {OMEGA_STD:.1f} (WR={WR_STD} x phi_w(r={R_WAVE_STD})={phi_w(R_WAVE_STD)})")
    print(f"\n{'N_ZONES':>8} {'delta':>7} {'ZW':>5} {'sg4 mean':>10} {'sg4 SE':>8} "
          f"{'na_ratio':>10} {'coll/site':>10}")
    a_summary = {}
    for n_zones in N_ZONES_SWEEP:
        zw    = HALF // n_zones
        delta = zw / R_WAVE_STD
        sg4s  = [results[f"A,nz{n_zones},{s}"]['sg4']
                 for s in SEEDS if f"A,nz{n_zones},{s}" in results]
        nas   = [results[f"A,nz{n_zones},{s}"]['na_ratio']
                 for s in SEEDS if f"A,nz{n_zones},{s}" in results]
        colls = [results[f"A,nz{n_zones},{s}"]['coll_site']
                 for s in SEEDS if f"A,nz{n_zones},{s}" in results]
        if not sg4s: continue
        m  = float(np.mean(sg4s)); se = float(np.std(sg4s)/np.sqrt(len(sg4s)))
        na_m = float(np.nanmean(nas)) if nas else float('nan')
        co_m = float(np.mean(colls)) if colls else float('nan')
        a_summary[n_zones] = {
            'delta': delta, 'zw': zw,
            'sg4_mean': m, 'sg4_se': se, 'sg4_vals': [float(v) for v in sg4s],
            'na_ratio': na_m, 'coll_site': co_m,
        }
        print(f"{n_zones:>8} {delta:>7.1f} {zw:>5} {m:>10.4f} {se:>8.4f} "
              f"{na_m:>10.3f} {co_m:>10.4f}")

    print("\n=== Exp B: gate x adversarial pressure ===")
    print(f"T_ENCODE={T_ENCODE}, T_ADV={T_ADV}")
    header = f"{'adv_amp':>8}  " + "  ".join(f"{c:>10}" for c in B_GATE_CONDITIONS)
    print("\nFinal fidelity (mean over 5 seeds):")
    print(header)
    b_summary = {}
    for adv_amp in ADV_AMPS:
        row = f"{adv_amp:>8.3f}  "
        for cond_name in B_GATE_CONDITIONS:
            fids = [results[f"B,amp{adv_amp:.3f},{cond_name},{s}"]['final_fidelity']
                    for s in SEEDS if f"B,amp{adv_amp:.3f},{cond_name},{s}" in results]
            if fids:
                m  = float(np.mean(fids)); se = float(np.std(fids)/np.sqrt(len(fids)))
                b_summary[(adv_amp, cond_name)] = {
                    'fidelity_mean': m, 'fidelity_se': se,
                    'fidelity_vals': [float(v) for v in fids],
                }
                row += f"  {m:>10.4f}"
            else:
                row += f"  {'---':>10}"
        print(row)

    # Per-amplitude: which condition has highest fidelity?
    print("\nBest gate per adv_amp:")
    for adv_amp in ADV_AMPS:
        best_cond = None; best_fid = -99.
        for cond_name in B_GATE_CONDITIONS:
            k = (adv_amp, cond_name)
            if k in b_summary and b_summary[k]['fidelity_mean'] > best_fid:
                best_fid = b_summary[k]['fidelity_mean']
                best_cond = cond_name
        print(f"  adv_amp={adv_amp:.3f}: best={best_cond} (fidelity={best_fid:.4f})")

    # Save analysis
    analysis = {
        'exp_a': {str(nz): v for nz, v in a_summary.items()},
        'exp_b': {f"{a:.3f}_{c}": v for (a,c),v in b_summary.items()},
    }
    analysis_file = os.path.join(os.path.dirname(__file__),
                                 "results", "paper47_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis -> {analysis_file}")
