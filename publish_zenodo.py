"""
publish_zenodo.py  --  Auto-publish a VCML paper PDF to Zenodo

Usage:
    python publish_zenodo.py 76          # publish paper 76
    python publish_zenodo.py 76 --draft  # upload but don't publish (keep as draft)
    python publish_zenodo.py --all       # publish all papers with PDFs

Prerequisites:
    1. Create a free account at https://zenodo.org
    2. Go to Account -> Applications -> Personal Access Tokens
    3. Create a token with scope: deposit:write + deposit:actions
    4. Set environment variable:  ZENODO_TOKEN=your_token_here

For testing use sandbox (no real DOIs, free):
    Set ZENODO_SANDBOX=1  (uses sandbox.zenodo.org instead)
"""

import os, sys, json, re, time, argparse
from pathlib import Path
import requests

# load .env file if present (ZENODO_TOKEN=xxx in .env)
_env_file = Path(__file__).parent / '.env'
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

# ── config ────────────────────────────────────────────────────────────
PAPERS_DIR = Path(__file__).parent / 'papers'

TOKEN     = os.environ.get('ZENODO_TOKEN', '')
SANDBOX   = os.environ.get('ZENODO_SANDBOX', '0') == '1'
BASE_URL  = 'https://sandbox.zenodo.org/api' if SANDBOX else 'https://zenodo.org/api'

HEADERS   = {'Authorization': f'Bearer {TOKEN}'}

GITHUB_URL = 'https://github.com/AccidentalGenius101/adaptive-memory-theory'

# Default metadata applied to every paper
DEFAULT_META = {
    'upload_type':        'publication',
    'publication_type':   'preprint',
    'access_right':       'open',
    'license':            'cc-by-4.0',
    'creators': [
        {'name': 'Aubin-Moreau, Gabriel', 'affiliation': 'Independent Research'},
    ],
    'keywords': ['VCML', 'VCSM', 'universality class', 'non-equilibrium', 'Monte Carlo',
                 'critical exponents', 'causal purity', 'zone-mean order parameter',
                 'GPU simulation', 'CUDA'],
    'communities': [],
    'related_identifiers': [
        {'identifier': GITHUB_URL,
         'relation':   'isSupplementTo',
         'scheme':     'url'},
    ],
    'notes': f'Full source code and data at {GITHUB_URL}',
}

# ── clean plain-text titles and abstracts per paper ───────────────────
# These override the LaTeX extraction (which mangles math symbols).
PAPER_META = {
    63: {
        'title': 'VCML Paper 63: Critical Exponents beta=0.65, nu=0.97 via Finite-Size Scaling',
        'description': (
            'Fine scan of causal purity P in [0, 0.08] at L in {20,30,40,60,80} with 8 seeds '
            'and 12k steps under extensive drive. Extracts order-parameter exponent beta=0.65, '
            'correlation-length exponent nu=0.97, and beta/nu=0.668. Transition point p_c~0.018 '
            'with evidence consistent with p_c=0+. Exponents match no known 2D non-equilibrium '
            'universality class, suggesting a novel class defined by causal-purity control of noise.'
        ),
    },
    64: {
        'title': 'VCML Paper 64: p_c = 0+ Confirmed — Causal Order for Any Positive Purity',
        'description': (
            'Large-L Binder cumulant test at L in {40,60,80,100,120} with 12 seeds and 12k steps '
            'under extensive drive. U4(L=120) > U4(L=40) for all P >= 0.002: the system is in the '
            'ordered phase for any positive causal purity. No disorder threshold exists. '
            'p_c = 0+ strongly confirmed. Cosmological parallel: causality implies order without fine-tuning.'
        ),
    },
    67: {
        'title': 'VCML Paper 67: Probabilistic Wave Firing Protocol and p_c = 0+ Reconfirmed',
        'description': (
            'Fixes a systematic wave-density deficit at large L caused by integer rounding of WAVE_EVERY. '
            'New protocol: probabilistic firing with probability min(1, (L_base/L)^2) per step. '
            'Reconfirms p_c = 0+ at L=160. Discovers that extra waves hurt ordering at low causal purity '
            '(optimal wave density is signal-dependent). Establishes probabilistic firing as the '
            'standard protocol for L >= 100.'
        ),
    },
    68: {
        'title': 'VCML Paper 68: Manna Universality Class Falsified; Novel Class Conjectured',
        'description': (
            'Fine P scan at L=160 rules out Manna class: phi flux creation/decay ratio = 3.23 >> 1, '
            'violating exact activity conservation. Power-law order-parameter scaling wins over BKT '
            'and exponential fits. FSS slopes confirm disorder-to-order crossover and p_c=0+. '
            'Conjecture: novel universality class defined by directed dynamics + Z2 symmetry + '
            'long-range wave coupling.'
        ),
    },
    69: {
        'title': 'VCML Paper 69: Minimum Wave-Range Threshold for Zone-Mean Ordering',
        'description': (
            'Systematic r_w sweep at L in {40,80,160} establishes that |M| ~ A(r_w)^0.67 '
            '(ordering scales with wave area). At r_w=1 the protocol breaks down for L>=40 '
            '(wave probability caps at 1.0) and the ordered phase is inaccessible at P<=0.050. '
            'Minimum threshold conjecture: r_w >= zone_width/20. '
            'Directed-percolation hypothesis is untestable at r_w=1.'
        ),
    },
    70: {
        'title': 'VCML Paper 70: Phase Boundary in (r_w, P) Space; RG Dimension d_wave = 1/2',
        'description': (
            'Joint (r_w, P) scan maps the ordering threshold. Protocol B (matched coverage) gives '
            'r_w*(P) ~ P^{-0.527} ~ P^{-1/2}, confirming that the composite variable xi = r_w * sqrt(P) '
            'is the scale-invariant critical coupling. The wave operator has RG scaling dimension 1/2. '
            'The VCML field action contains a non-local smeared coupling term. '
            'Zone-bleeding anomaly explained by wave straddling the zone boundary at r_w=2.'
        ),
    },
    71: {
        'title': 'VCML Paper 71: RG Fixed Point via xi=r_w*sqrt(P); beta=0.628, nu=0.98',
        'description': (
            'Establishes xi = r_w * sqrt(P) as the scale-invariant critical coupling. '
            'Protocol-A iso-xi universality fails (r_w-dependent amplitude breaks noise equalization); '
            'Protocol-B iso-xi universality holds. Extracts beta=0.628 (R2=0.922) and nu=0.98 '
            'via FSS slopes beta/nu=0.64. Indirect eta=1.28 from eta=2*beta/nu. '
            'Exponents (beta, nu, eta) match no known 2D non-equilibrium class.'
        ),
    },
    72: {
        'title': 'VCML Paper 72: Two-Point Correlator Null — Zone-Mean Transition, eta Undefined',
        'description': (
            'Direct measurement of the anomalous dimension eta from the spatial two-point correlator '
            'G(r) of the phi field at criticality. Result: G(r) ~ 0 at all r > 0 (amplitudes at '
            'statistical noise floor). The VCML ordering transition is a zone-mean transition with '
            'effective dimension d_eff = 0. The anomalous dimension eta is not defined at the cell '
            'level. The appropriate order parameter is the temporal autocorrelator, not a spatial one.'
        ),
    },
    73: {
        'title': 'VCML Paper 73: Dynamic Exponent z — Intrinsic VCSM Timescale tau~200 Identified',
        'description': (
            'Temporal autocorrelator C(t) and correlation time tau_corr measured across three phases. '
            'Key result: tau_VCSM ~ 200 steps is an intrinsic architectural timescale set by the '
            'gate parameters {SS=8, FA=0.30, FIELD_DECAY=0.999}, masking the critical divergence '
            'tau_crit ~ P^{-z*nu} for P in [0.007, 0.050]. C(t) ~ t^{-2.2} (power law wins over '
            'exponential). Critical timescale only visible at P < 0.002. z measurement requires '
            'deeper P scan.'
        ),
    },
    74: {
        'title': 'VCML Paper 74: Sub-Diffusive Dynamic Exponent z=0.48; Phi-Field Timescale tau~1000',
        'description': (
            'Deep-P GPU scan (P in {0.0001..0.010}, L=80, 8 seeds, up to 150k steps) extracts the '
            'dynamic exponent from tau_corr ~ P^{-z*nu}. Global fit (Phases A+B): z=0.48, R2=0.71, nu=0.98. '
            'Sub-diffusive: z=0.48 << z_ModelA=2, z < z_KPZ=1.5. The phi-field timescale tau~1000 '
            'is distinct from the Ising spin timescale tau~200 identified in Paper 73. '
            'Completes the VCML Minecraft seed: beta=0.63, nu=0.98, z=0.48, P_c=0+. '
            'GPU implementation: torch.compile(mode=default) with 8-seed batching on RTX 3060 (~8x speedup).'
        ),
    },
    75: {
        'title': 'VCML Paper 75: FSS Blocking at Deep P; Gamma Null; d_eff=0 Hyperscaling Violation',
        'description': (
            'Attempts to confirm z=0.48 at deeper P and to measure the susceptibility exponent gamma '
            'from chi=var(M). Both produce informative null results. FSS blocking: xi(P=1e-4) ~ 6300 >> L=80; '
            'all deep-P measurements are finite-size dominated (z_B=-0.27, unphysical). '
            'Paper 74 z=0.48 is confirmed by exclusion as the only accessible scaling window. '
            'Gamma null: chi ~ L^{-2} everywhere (geometric dilution). '
            'Hyperscaling violation: d_eff=0 zone-mean order parameter breaks Josephson relation. '
            'CUDAGraph v3 backend: 229k steps/min at L=80, B=8 (3x speedup over v2).'
        ),
    },
    76: {
        'title': 'VCML Paper 76: Susceptibility Exponent Gamma Blocked by d_eff=0 Zone-Mean Kinematics',
        'description': (
            'Second attempt to measure gamma using the intensive susceptibility chi_int = L^2 * var(M) '
            'to cancel geometric dilution. Three phases scan P in [0.003, 0.050] at L=80, '
            'L in {40..160} at P=0.010 (FSS), and P in [0.002, 0.030] at L=120. '
            'Result: chi_int ~ 0.0024 flat everywhere. Root cause: the phi field has zero spatial '
            'correlations (G_phi(r)~0; Paper 72), so CLT gives var(M) = 4*sigma^2/L^2 exactly, '
            'and chi_int = 4*sigma^2 = const. At d_eff=0 the susceptibility integral vanishes. '
            'Direct gamma measurement is kinematically blocked. '
            'Indirect estimate from Fisher scaling: gamma = nu*(2-eta) = 0.71. '
            'Universality class fully characterized by the three directly measured exponents '
            'beta=0.628, nu=0.98, z=0.48.'
        ),
    },
}

# ── paper catalogue (auto-detect from directory structure) ────────────
def find_paper_pdf(paper_num: int) -> tuple[Path | None, Path | None]:
    """Return (pdf_path, tex_path) for a paper number, or (None, None)."""
    # search all subdirs for pattern paper{N}_* or paper{N:02d}_*
    for d in sorted(PAPERS_DIR.iterdir()):
        if not d.is_dir(): continue
        stem = d.name
        # match paper76_*, paper076_*, paper76gamma*, etc.
        m = re.match(r'paper0*(\d+)', stem)
        if m and int(m.group(1)) == paper_num:
            # find PDF
            pdfs = list(d.glob('*.pdf')) + list((d / 'results').glob('*.pdf') if (d / 'results').exists() else [])
            texs = list(d.glob('*.tex'))
            pdf  = pdfs[0] if pdfs else None
            tex  = texs[0] if texs else None
            return pdf, tex
    return None, None

def extract_title_from_tex(tex_path: Path) -> str | None:
    """Extract \\title{...} from a .tex file."""
    try:
        text = tex_path.read_text(encoding='utf-8', errors='replace')
        m = re.search(r'\\title\{(.+?)\}', text, re.DOTALL)
        if m:
            # strip LaTeX commands for a clean title
            t = m.group(1)
            t = re.sub(r'\\textbf\{(.+?)\}', r'\1', t)
            t = re.sub(r'\\[a-zA-Z]+\{(.+?)\}', r'\1', t)
            t = re.sub(r'\\[a-zA-Z]+', '', t)
            t = re.sub(r'\s+', ' ', t).strip()
            return t
    except Exception:
        pass
    return None

def extract_abstract_from_tex(tex_path: Path) -> str | None:
    """Extract abstract environment from .tex."""
    try:
        text = tex_path.read_text(encoding='utf-8', errors='replace')
        m = re.search(r'\\begin\{abstract\}(.+?)\\end\{abstract\}', text, re.DOTALL)
        if m:
            ab = m.group(1)
            # strip LaTeX
            ab = re.sub(r'\\[a-zA-Z]+\{(.+?)\}', r'\1', ab)
            ab = re.sub(r'\\[a-zA-Z]+', ' ', ab)
            ab = re.sub(r'\s+', ' ', ab).strip()
            return ab
    except Exception:
        pass
    return None

# ── Zenodo API helpers ────────────────────────────────────────────────
def _check_token():
    if not TOKEN:
        print('ERROR: ZENODO_TOKEN environment variable not set.')
        print('  Get a token at https://zenodo.org/account/settings/applications/')
        sys.exit(1)

def create_deposition() -> dict:
    r = requests.post(f'{BASE_URL}/deposit/depositions',
                      json={}, headers=HEADERS)
    r.raise_for_status()
    return r.json()

def upload_file(bucket_url: str, pdf_path: Path) -> dict:
    """Upload using the bucket URL returned by create_deposition."""
    with open(pdf_path, 'rb') as f:
        r = requests.put(
            f'{bucket_url}/{pdf_path.name}',
            data=f,
            headers={**HEADERS, 'Content-Type': 'application/octet-stream'},
        )
    r.raise_for_status()
    return r.json()

def set_metadata(deposition_id: int, meta: dict) -> dict:
    r = requests.put(
        f'{BASE_URL}/deposit/depositions/{deposition_id}',
        json={'metadata': meta},
        headers={**HEADERS, 'Content-Type': 'application/json'}
    )
    r.raise_for_status()
    return r.json()

def publish(deposition_id: int) -> dict:
    r = requests.post(
        f'{BASE_URL}/deposit/depositions/{deposition_id}/actions/publish',
        headers=HEADERS
    )
    r.raise_for_status()
    return r.json()

# ── main publish function ─────────────────────────────────────────────
def publish_paper(paper_num: int, draft: bool = False, verbose: bool = True) -> dict | None:
    """Publish one paper. Returns result dict with DOI, or None on failure."""
    pdf, tex = find_paper_pdf(paper_num)

    if pdf is None:
        print(f'  [P{paper_num}] No PDF found. Skipping.')
        return None

    # build metadata — use hand-written dict first, fall back to LaTeX extraction
    if paper_num in PAPER_META:
        title    = PAPER_META[paper_num]['title']
        abstract = PAPER_META[paper_num]['description']
    else:
        title    = (extract_title_from_tex(tex) if tex else None) or f'VCML Theory Paper {paper_num}'
        abstract = (extract_abstract_from_tex(tex) if tex else None) or f'VCML/VCSM theory paper {paper_num}.'

    meta = {
        **DEFAULT_META,
        'title':       title,
        'description': abstract,
        'publication_date': time.strftime('%Y-%m-%d'),
        'related_identifiers': [],
    }

    if verbose:
        print(f'\n[P{paper_num}] PDF: {pdf.name}')
        print(f'  Title: {title[:80]}...' if len(title)>80 else f'  Title: {title}')
        env_label = '(SANDBOX)' if SANDBOX else '(LIVE)'
        print(f'  Uploading to Zenodo {env_label}...')

    try:
        # 1. Create empty deposition
        dep  = create_deposition()
        did  = dep['id']
        if verbose: print(f'  Deposition created: id={did}')

        # 2. Upload PDF via bucket URL from deposition response
        bucket_url = dep['links']['bucket']
        upload_file(bucket_url, pdf)
        if verbose: print(f'  File uploaded: {pdf.name}')

        # 3. Set metadata
        set_metadata(did, meta)
        if verbose: print(f'  Metadata set.')

        # 4. Publish (or leave as draft)
        if draft:
            url = f'https://{"sandbox." if SANDBOX else ""}zenodo.org/deposit/{did}'
            print(f'  Draft saved. Review at: {url}')
            return {'id': did, 'doi': None, 'url': url, 'status': 'draft'}
        else:
            result = publish(did)
            doi    = result.get('doi', result.get('doi_url', 'N/A'))
            url    = result.get('links', {}).get('html', '')
            print(f'  Published! DOI: {doi}')
            print(f'  URL: {url}')
            return {'id': did, 'doi': doi, 'url': url, 'status': 'published'}

    except requests.HTTPError as e:
        print(f'  ERROR: {e}')
        if e.response is not None:
            print(f'  Response: {e.response.text[:400]}')
        return None

# ── CLI ───────────────────────────────────────────────────────────────
def main():
    _check_token()

    ap = argparse.ArgumentParser(description='Publish VCML papers to Zenodo')
    ap.add_argument('paper_nums', nargs='*', type=int,
                    help='Paper numbers to publish (e.g. 74 75 76)')
    ap.add_argument('--all',   action='store_true', help='Publish all papers with PDFs')
    ap.add_argument('--draft', action='store_true', help='Upload as draft (do not publish)')
    ap.add_argument('--list',  action='store_true', help='List papers with PDFs and exit')
    args = ap.parse_args()

    if args.list:
        print('Papers with PDFs:')
        for d in sorted(PAPERS_DIR.iterdir()):
            if not d.is_dir(): continue
            m = re.match(r'paper0*(\d+)', d.name)
            if m:
                n = int(m.group(1))
                pdf, _ = find_paper_pdf(n)
                if pdf:
                    print(f'  P{n:3d}: {pdf}')
        return

    if args.all:
        nums = []
        for d in sorted(PAPERS_DIR.iterdir()):
            if not d.is_dir(): continue
            m = re.match(r'paper0*(\d+)', d.name)
            if m:
                n = int(m.group(1))
                pdf, _ = find_paper_pdf(n)
                if pdf: nums.append(n)
        print(f'Publishing {len(nums)} papers...')
    elif args.paper_nums:
        nums = args.paper_nums
    else:
        ap.print_help()
        return

    results = {}
    for n in nums:
        r = publish_paper(n, draft=args.draft)
        if r: results[n] = r
        time.sleep(1)   # be polite to the API

    # Save DOI record
    doi_file = Path(__file__).parent / 'zenodo_dois.json'
    existing = {}
    if doi_file.exists():
        existing = json.loads(doi_file.read_text())
    existing.update({str(k): v for k, v in results.items()})
    doi_file.write_text(json.dumps(existing, indent=2))
    print(f'\nDOI record saved to {doi_file}')

if __name__ == '__main__':
    main()
