"""
VCSMLattice — VCSM on a coupled-map lattice

Instantiation of VCSM learning rule on a 2D grid substrate.
VCSM-L is the primary substrate used in Track 1 and Papers 1-3.

================================================================================
ALGORITHM
================================================================================

VCSMLattice is a W×H grid of GRU cells operating under a hard viability constraint.
No loss function. No backpropagation. No global signal. The only pressure is survival.

Memory accumulates at the state level (not the weight level), gated by survival,
across three timescales (see VCSM core rule).

Sites collapse when viability vals[i] exits [BOUND_LOW, BOUND_HIGH].
Collapsed sites are immediately reborn from the local debris pool,
with hid seeded from fieldM[location] (the intergenerational memory).
"""

import random
import math
from statistics import mean
from .. import vcsm_core as core
from ..config import VCSMConfig


class VCSMLattice:
    """
    Viability-Constrained Coupled Map Lattice with VCSM learning rule.

    The grid consists of W*H sites. Each site has:
      - A GRU cell (weights shared globally; state is per-site)
      - A scalar viability value vals[i] in [0,1]
      - VCSM memory layers: baseline_h, mid_mem, fieldM

    Sites collapse when vals[i] exits [BOUND_LOW, BOUND_HIGH].
    Collapsed sites are immediately reborn from the local debris pool,
    with hid seeded from fieldM[location] (the intergenerational memory).
    """

    def __init__(self, cfg: VCSMConfig = None, seed: int = 0):
        self.cfg = cfg or VCSMConfig()
        self.seed = seed
        random.seed(seed)

        C = self.cfg
        self.W, self.H, self.HS = C.W, C.H, C.HS
        self.TOTAL = C.W * C.H
        self.HALF = C.HALF

        # Per-site GRU weights (each site has independent random weights)
        self.weights = [core.make_gru_weights(C.HS, C.INPUT_SIZE) for _ in range(self.TOTAL)]

        # Site state
        self.vals      = [0.5] * self.TOTAL
        self.hid       = [[0.0] * C.HS for _ in range(self.TOTAL)]
        self.is_active = [False] * self.TOTAL

        # VCSM memory layers
        self.baseline_h = [[0.0] * C.HS for _ in range(self.TOTAL)]
        self.mid_mem    = [[0.0] * C.HS for _ in range(self.TOTAL)]
        self.fieldM     = [[0.0] * C.HS for _ in range(self.TOTAL)]

        # Site metadata
        self.age             = [0] * self.TOTAL
        self.calm_streak     = [0] * self.TOTAL
        self.collapse_count  = [0] * self.TOTAL

        # Debris pools: site -> list of {"hid": [...], "age": int}
        self.debris = [[] for _ in range(self.TOTAL)]

        # Grid connectivity
        self._build_neighbors()

        # Condition flags
        self.amnesic = False
        self.static  = False

        # Birth tracking
        self.birth_gen    = [0] * self.TOTAL
        self.birth_events = []

        # Step counter
        self.t = 0

        # Initialize sites
        self._init_grid()

    # ----------------------------------------------------------
    # Grid setup
    # ----------------------------------------------------------

    def _build_neighbors(self):
        """4-connected neighbour list for each site."""
        W, H = self.W, self.H
        self.neighbors = []
        for i in range(self.TOTAL):
            x, y = i % W, i // W
            nbs = []
            if x > 0:     nbs.append(i - 1)
            if x < W - 1: nbs.append(i + 1)
            if y > 0:     nbs.append(i - W)
            if y < H - 1: nbs.append(i + W)
            self.neighbors.append(nbs)

    def _init_grid(self):
        """Initialize all ACTIVE sites with random state."""
        C = self.cfg
        for i in range(self.TOTAL):
            x = i % self.W
            if x >= self.HALF:
                self.is_active[i] = True
                self.vals[i]      = random.uniform(0.3, 0.7)
                self.hid[i]       = [random.gauss(0, 0.1) for _ in range(self.HS)]
                self.age[i]       = random.randint(0, 50)
            else:
                self.is_active[i] = False
                self.vals[i]      = 0.5

    # ----------------------------------------------------------
    # VCSM Step 1 — Calm
    # ----------------------------------------------------------

    def _update_baseline(self):
        """[VCSM: Calm] baseline[i] += BETA * (hid[i] - baseline[i])"""
        beta = self.cfg.BASELINE_BETA
        TOTAL = self.TOTAL; HS = self.HS
        is_active = self.is_active; hid = self.hid
        baseline_h = self.baseline_h; calm_streak = self.calm_streak
        for i in range(TOTAL):
            if not is_active[i]:
                continue
            hi = hid[i]; bi = baseline_h[i]
            dev_sq = 0.0
            for k in range(HS):
                d = hi[k] - bi[k]
                bi[k] += beta * d
                dev_sq += d * d
            if dev_sq < 0.0025:
                calm_streak[i] += 1
            else:
                calm_streak[i] = 0

    # ----------------------------------------------------------
    # VCSM Step 2 — Perturb
    # ----------------------------------------------------------

    def _update_mid_mem(self):
        """[VCSM: Perturb] mid_mem[i] += ALPHA_MID * (hid[i] - baseline[i]); mid_mem[i] *= MID_DECAY"""
        C = self.cfg
        ALPHA_MID = C.ALPHA_MID; MID_DECAY = C.MID_DECAY
        TOTAL = self.TOTAL; HS = self.HS
        is_active = self.is_active; hid = self.hid
        baseline_h = self.baseline_h; mid_mem = self.mid_mem
        for i in range(TOTAL):
            if not is_active[i]:
                continue
            hi = hid[i]; bi = baseline_h[i]; mi = mid_mem[i]
            for k in range(HS):
                mi[k] = (mi[k] + ALPHA_MID * (hi[k] - bi[k])) * MID_DECAY

    # ----------------------------------------------------------
    # VCSM Step 3 — Consolidate
    # ----------------------------------------------------------

    def _consolidate(self):
        """[VCSM: Consolidate] fieldM[i] += FIELD_ALPHA * (mid_mem[i] - fieldM[i])"""
        C = self.cfg
        FIELD_ALPHA = C.FIELD_ALPHA; STABLE_STEPS = C.STABLE_STEPS
        TOTAL = self.TOTAL; HS = self.HS
        is_active = self.is_active; calm_streak = self.calm_streak
        mid_mem = self.mid_mem; fieldM = self.fieldM
        for i in range(TOTAL):
            if not is_active[i]:
                continue
            if calm_streak[i] < STABLE_STEPS:
                continue
            mi = mid_mem[i]; fi = fieldM[i]
            for k in range(HS):
                fi[k] += FIELD_ALPHA * (mi[k] - fi[k])

    # ----------------------------------------------------------
    # Field dynamics
    # ----------------------------------------------------------

    def _field_decay(self):
        """fieldM[i] *= FIELD_DECAY"""
        d = self.cfg.FIELD_DECAY
        TOTAL = self.TOTAL; HS = self.HS; fieldM = self.fieldM
        for i in range(TOTAL):
            fi = fieldM[i]
            for k in range(HS):
                fi[k] *= d

    def _field_diffuse(self):
        """Spatial smoothing: blend each site's fieldM with mean of neighbours."""
        rate = self.cfg.FIELD_DIFFUSE_RATE; keep = 1.0 - rate
        TOTAL = self.TOTAL; HS = self.HS
        is_active = self.is_active; fieldM = self.fieldM
        neighbors = self.neighbors
        new_field = [fi[:] for fi in fieldM]
        for i in range(TOTAL):
            if not is_active[i]:
                continue
            nbs = [j for j in neighbors[i] if is_active[j]]
            if not nbs:
                continue
            n_nbs = len(nbs); fi = fieldM[i]; nf = new_field[i]
            for k in range(HS):
                nb_sum = 0.0
                for j in nbs: nb_sum += fieldM[j][k]
                nf[k] = keep * fi[k] + rate * nb_sum / n_nbs
        self.fieldM = new_field

    # ----------------------------------------------------------
    # GRU dynamics
    # ----------------------------------------------------------

    def _sim_step(self, inputs: dict):
        """Advance GRU state and update vals incrementally."""
        C = self.cfg
        VALS_DECAY = C.VALS_DECAY; VALS_NAV = C.VALS_NAV; ADJ_SCALE = C.ADJ_SCALE
        TOTAL = self.TOTAL; HS = self.HS; IS = self.cfg.INPUT_SIZE
        is_active = self.is_active; hid = self.hid; vals = self.vals
        age = self.age; weights = self.weights
        zero_input = [0.0] * IS
        for i in range(TOTAL):
            if not is_active[i]:
                continue
            x = inputs.get(i, zero_input)
            new_h, adj = core.gru_step(weights[i], hid[i], x, HS, IS)
            hid[i] = new_h
            nb_mean = x[0]
            new_v = VALS_DECAY * vals[i] + VALS_NAV * nb_mean + ADJ_SCALE * adj
            vals[i] = max(0.0, min(1.0, new_v))
            age[i] += 1

    # ----------------------------------------------------------
    # VCSM Step 4 — Birth
    # ----------------------------------------------------------

    def _check_viability(self, i) -> bool:
        """Return True if site i should collapse this step."""
        C = self.cfg
        v = self.vals[i]

        if v < C.BOUND_LOW or v > C.BOUND_HIGH:
            return True

        hi = self.hid[i]; bi = self.baseline_h[i]
        dev = 0.0
        for k in range(self.HS):
            d = hi[k] - bi[k]
            dev += d if d >= 0 else -d
        if dev > C.INST_COLLAPSE_THRESH:
            if random.random() < C.INST_COLLAPSE_PROB:
                return True

        return False

    def _collapse_and_rebirth(self, i):
        """[VCSM: Birth] Collapse site i and immediately rebirth from local debris + fieldM."""
        C = self.cfg
        snap = {"hid": self.hid[i][:], "age": self.age[i]}
        self.debris[i].append(snap)
        if len(self.debris[i]) > C.MAX_RESIDUE:
            self.debris[i].pop(0)

        pool = list(self.debris[i])
        for j in self.neighbors[i]:
            pool.extend(self.debris[j])

        if pool:
            pool_sorted = sorted(pool, key=lambda d: d["age"], reverse=True)
            top = max(1, len(pool_sorted) // 3)
            donor = random.choice(pool_sorted[:top])
            new_hid = [donor["hid"][k] + random.gauss(0, C.SEED_FRAG_NOISE)
                       for k in range(self.HS)]
        else:
            new_hid = [random.gauss(0, 0.1) for _ in range(self.HS)]

        if not self.amnesic:
            fm = self.fieldM[i]
            fm_mag = core.l2(fm)
            if fm_mag > 1e-6:
                beta = C.FIELD_SEED_BETA
                for k in range(self.HS):
                    new_hid[k] = (1 - beta) * new_hid[k] + beta * fm[k]

        self.hid[i]         = new_hid
        self.vals[i]        = 0.5
        self.age[i]         = 0
        self.calm_streak[i] = 0
        self.mid_mem[i]     = [0.0] * self.HS
        self.is_active[i]   = True
        self.collapse_count[i] += 1

        self.birth_gen[i] += 1
        self.birth_events.append((i, self.birth_gen[i]))

    # ----------------------------------------------------------
    # Main step
    # ----------------------------------------------------------

    def step(self, inputs: dict = None):
        """Advance the lattice by one timestep."""
        if inputs is None:
            inputs = {}

        self._sim_step(inputs)
        self._update_baseline()
        self._update_mid_mem()
        self._consolidate()

        self._field_decay()
        if self.t % self.cfg.FIELD_SMOOTH_EVERY == 0:
            self._field_diffuse()

        self.birth_events = []
        if not self.static:
            for i in range(self.TOTAL):
                if not self.is_active[i]:
                    continue
                if i % self.W < self.HALF:
                    continue
                if self._check_viability(i):
                    self._collapse_and_rebirth(i)

        self.t += 1

    # ----------------------------------------------------------
    # Readout utilities
    # ----------------------------------------------------------

    def get_zones(self, k: int = None) -> dict:
        """Classify active ACTIVE-zone sites into stable and churn zones."""
        k = k or self.cfg.ZONE_K
        active = [i for i in range(self.TOTAL)
                  if self.is_active[i] and i % self.W >= self.HALF]
        k = min(k, len(active) // 5)
        if not active or k == 0:
            return {"stable": [], "churn": []}
        by_count = sorted(active, key=lambda i: self.collapse_count[i])
        return {"stable": by_count[:k], "churn": by_count[-k:]}

    def get_field_map(self) -> list:
        """Return fieldM as a list of shape (TOTAL, HS)."""
        return [row[:] for row in self.fieldM]

    def get_mid_map(self) -> list:
        """Return mid_mem as a list of shape (TOTAL, HS)."""
        return [row[:] for row in self.mid_mem]

    def get_state_snapshot(self) -> dict:
        """Return a full state snapshot for analysis or logging."""
        zones = self.get_zones()
        active = [i for i in range(self.TOTAL) if self.is_active[i]]
        return {
            "t":              self.t,
            "n_active":       len(active),
            "n_collapses":    sum(self.collapse_count),
            "stable_zone":    zones["stable"],
            "churn_zone":     zones["churn"],
            "fieldM":         self.get_field_map(),
            "mid_mem":        self.get_mid_map(),
            "hid":            [row[:] for row in self.hid],
            "vals":           self.vals[:],
            "collapse_count": self.collapse_count[:],
        }

    def field_at(self, site_index: int) -> list:
        """Return fieldM vector at site_index (length HS)."""
        return self.fieldM[site_index][:]

    def zone_mean_field(self, site_indices: list) -> list:
        """Return mean fieldM vector over a list of site indices."""
        if not site_indices:
            return [0.0] * self.HS
        return [
            sum(self.fieldM[i][k] for i in site_indices) / len(site_indices)
            for k in range(self.HS)
        ]

    def spatial_sep(self, left_sites: list, right_sites: list) -> float:
        """Spatial separation metric: L2 distance between mean fieldM of left and right site groups."""
        mu_l = self.zone_mean_field(left_sites)
        mu_r = self.zone_mean_field(right_sites)
        return core.l2([mu_l[k] - mu_r[k] for k in range(self.HS)])


# ============================================================
# WaveEnvironment — input/perturbation generator for lattice
# ============================================================

class WaveEnvironment:
    """
    Spatial wave input generator for VCSMLattice.
    Matches the Track 1 experimental setup.

    Two classes of waves:
      Class 0: hits LEFT active zone  (x in [HALF, lx_mid))
      Class 1: hits RIGHT active zone (x in [lx_mid, W))

    Each wave activates a column of sites for WAVE_DURATION steps.
    """

    def __init__(self, vcml: VCSMLattice, wave_ratio: float = 0.6,
                 wave_duration: int = 15):
        self.vcml          = vcml
        self.wave_ratio    = wave_ratio
        self.wave_duration = wave_duration
        self.HALF          = vcml.HALF
        self.lx_mid        = vcml.HALF + vcml.W // 4
        self.W             = vcml.W
        self.H             = vcml.H
        self.active_waves  = []
        self.current_class = 0

    def _launch_wave(self):
        """Launch a new class-structured wave."""
        cls = random.randint(0, 1)
        if cls == 0:
            cx = random.randint(self.HALF, self.lx_mid - 1)
        else:
            cx = random.randint(self.lx_mid, self.W - 1)
        cy = random.randint(0, self.H - 1)
        self.active_waves.append([cx, cy, self.wave_duration, cls])
        self.current_class = cls

    def get_inputs(self, t: int) -> tuple:
        """
        Return (input_dict, current_class) for this timestep.
        Launches new waves with probability wave_ratio and advances active ones.
        """
        vcml = self.vcml
        if random.random() < self.wave_ratio / self.wave_duration:
            self._launch_wave()

        wave_act = [0.0] * vcml.TOTAL

        remaining = []
        for wave in self.active_waves:
            cx, cy, steps, cls = wave
            if steps <= 0:
                continue
            for i in range(vcml.TOTAL):
                x, y = i % self.W, i // self.W
                dist = abs(x - cx) + abs(y - cy)
                if dist <= 2:
                    wave_act[i] += max(0.0, 1.0 - dist * 0.4)
            wave[2] -= 1
            if wave[2] > 0:
                remaining.append(wave)
        self.active_waves = remaining

        # Direct wave perturbation to vals
        for i in range(vcml.TOTAL):
            if not vcml.is_active[i] or i % self.W < self.HALF:
                continue
            wa = wave_act[i]
            if wa > 0.05:
                v = vcml.vals[i]
                if self.current_class == 0:
                    vcml.vals[i] = max(0.0, v * (1.0 - wa * 0.5))
                else:
                    vcml.vals[i] = min(1.0, v + wa * 0.5)

        inputs = {}
        for i in range(vcml.TOTAL):
            if not vcml.is_active[i] or i % self.W < self.HALF:
                continue
            nbs = [j for j in vcml.neighbors[i] if vcml.is_active[j]]
            nb_mean = mean(vcml.vals[j] for j in nbs) if nbs else 0.5
            age_norm = min(1.0, vcml.age[i] / 300.0)
            inputs[i] = [nb_mean, min(1.0, wave_act[i]), age_norm]

        return inputs, self.current_class
