"""
Substrate: A minimal exploration of differential persistence.

No agents. No genomes. No fitness function. No goals. Just a memory grid where:
  * Cells have values that change based on neighbor interactions
  * Every cell leaves a TRACE of its previous value
  * Cells that can "see" the delta between current and trace adapt
  * Cells that can't see the delta just persist blindly

The question: does the ability to see change produce qualitatively different
behavior than blind persistence?

Run on Pythonista (iOS) or any Python with console output.

---
This is the original substrate that started the VCSM project.
The trace/delta mechanism here became the contrastive baseline in VCSM.
The death/renewal mechanism became mandatory turnover.
The left/right split (blind vs. seeing) became zone differentiation.
The "adaptations" counter became the causal purity metric.
"""

import random
import time
import os

# --- Substrate Configuration ---
WIDTH               = 40
HEIGHT              = 20
STEPS               = 500
SHOW_EVERY          = 10
CATASTROPHE_INTERVAL = 75


# --- The Grid ---
# Each cell has:
#   value:        current energy level (0.0 - 1.0)
#   trace:        memory of previous value
#   can_see:      whether this cell compares current to trace
#   age:          how long this cell has persisted

def make_cell(can_see=False):
    v = random.random()
    return {
        'value':       v,
        'trace':       v,    # starts with no delta
        'can_see':     can_see,
        'age':         0,
        'adaptations': 0,    # times it changed behavior based on delta
    }

def make_grid():
    grid = []
    for y in range(HEIGHT):
        row = []
        for x in range(WIDTH):
            # Left half: blind cells. Right half: seeing cells.
            can_see = (x >= WIDTH // 2)
            row.append(make_cell(can_see=can_see))
        grid.append(row)
    return grid


# --- Perturbation ---
# The environment randomly disturbs cells.
# This is "reality" pushing against persistence.

def perturb(grid, intensity=0.15):
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if random.random() < 0.2:   # 20% of cells get hit harder
                grid[y][x]['value'] += random.uniform(-intensity, intensity)
                grid[y][x]['value'] = max(0.0, min(1.0, grid[y][x]['value']))


# --- Neighbor Interaction ---
# Cells influence neighbors. Simple diffusion.
# But the key difference: seeing cells adjust HOW they interact
# based on their delta (trace vs current).

def get_neighbors(grid, x, y):
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
            neighbors.append(grid[ny][nx])
    return neighbors

def step(grid):
    new_values = [[0.0] * WIDTH for _ in range(HEIGHT)]

    for y in range(HEIGHT):
        for x in range(WIDTH):
            cell      = grid[y][x]
            neighbors = get_neighbors(grid, x, y)
            avg_neighbor = sum(n['value'] for n in neighbors) / len(neighbors)

            if cell['can_see']:
                # --- SEEING CELL ---
                # Compare current value to trace (previous value)
                delta = cell['value'] - cell['trace']

                # If delta is negative (I'm declining), weight neighbors MORE
                # (seek external input when declining)
                # If delta is positive (I'm growing), weight self MORE
                # (trust current trajectory when growing)
                if delta < -0.01:
                    # Declining: open up to neighbors
                    mix = 0.3
                    cell['adaptations'] += 1
                elif delta > 0.01:
                    # Growing: maintain course
                    mix = 0.05
                else:
                    # Stable: neutral
                    mix = 0.1

                new_values[y][x] = cell['value'] * (1 - mix) + avg_neighbor * mix

            else:
                # --- BLIND CELL ---
                # Fixed mixing ratio. Can't see delta. Just persists.
                mix = 0.1
                new_values[y][x] = cell['value'] * (1 - mix) + avg_neighbor * mix

    # Update all cells
    for y in range(HEIGHT):
        for x in range(WIDTH):
            cell         = grid[y][x]
            cell['trace'] = cell['value']   # save current as trace
            cell['value'] = max(0.0, min(1.0, new_values[y][x]))
            cell['age']  += 1


# --- Catastrophe ---
# Periodically, a big environmental shift hits.
# This is where blind spots kill you.

def catastrophe(grid, step_num):
    """Periodically, shift the energy landscape."""
    if step_num > 0 and step_num % CATASTROPHE_INTERVAL == 0:
        # Invert a large region
        cx, cy = random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1)
        radius = 8  # bigger blast
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if abs(x - cx) + abs(y - cy) < radius:
                    grid[y][x]['value'] = 1.0 - grid[y][x]['value']
        return True
    return False


# --- Measurement ---
# Don't measure success. Measure persistence and diversity.

def measure(grid):
    blind_cells  = []
    seeing_cells = []

    for y in range(HEIGHT):
        for x in range(WIDTH):
            cell = grid[y][x]
            if cell['can_see']:
                seeing_cells.append(cell)
            else:
                blind_cells.append(cell)

    def stats(cells):
        values   = [c['value'] for c in cells]
        avg      = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        avg_age  = sum(c['age'] for c in cells) / len(cells)
        return {
            'avg_value': avg,
            'variance':  variance,
            'diversity': variance,   # variance IS diversity here
            'avg_age':   avg_age,
        }

    blind_stats  = stats(blind_cells)
    seeing_stats = stats(seeing_cells)

    # Count adaptations (only seeing cells have these)
    total_adaptations = sum(c['adaptations'] for c in seeing_cells)

    return blind_stats, seeing_stats, total_adaptations


# --- Visualization (ASCII) ---

def render(grid, step_num, cat_happened=False):
    blocks = ' ░▒▓█'

    # Clear screen (works in most terminals and Pythonista)
    print('\033[2J\033[H', end='')

    header = f"Step {step_num}"
    if cat_happened:
        header += "  *** CATASTROPHE ***"
    print(header)
    print("=" * (WIDTH + 3))
    print(f"{'BLIND':^{WIDTH//2}} | {'SEEING':^{WIDTH//2}}")
    print("-" * (WIDTH + 3))

    for y in range(HEIGHT):
        line = ""
        for x in range(WIDTH):
            v   = grid[y][x]['value']
            idx = min(int(v * len(blocks)), len(blocks) - 1)
            line += blocks[idx]
            if x == WIDTH // 2 - 1:
                line += " | "
        print(line)

    # Stats
    blind_s, seeing_s, adaptations = measure(grid)
    print("-" * (WIDTH + 3))
    print(f"Avg value:   {blind_s['avg_value']:.3f}   |   {seeing_s['avg_value']:.3f}")
    print(f"Diversity:   {blind_s['diversity']:.4f}  |   {seeing_s['diversity']:.4f}")
    print(f"Adaptations: --      |   {adaptations}")
    print()


# --- Death and Renewal ---
# Cells that hit 0 or 1 (extremes) die and get replaced.
# This is the "small death" that creates space.

def death_and_renewal(grid):
    deaths_blind  = 0
    deaths_seeing = 0

    for y in range(HEIGHT):
        for x in range(WIDTH):
            cell = grid[y][x]
            # Death at extremes: fully depleted or fully saturated
            if cell['value'] < 0.05 or cell['value'] > 0.95:
                can_see = cell['can_see']
                if can_see:
                    deaths_seeing += 1
                else:
                    deaths_blind += 1
                # Renewal: new cell with random value
                grid[y][x] = make_cell(can_see=can_see)

    return deaths_blind, deaths_seeing


# --- Main Loop ---

def run():
    grid = make_grid()

    blind_deaths_total  = 0
    seeing_deaths_total = 0

    history = []

    for s in range(STEPS):
        # Perturb
        perturb(grid)

        # Maybe catastrophe
        cat = catastrophe(grid, s)

        # Step (interact + adapt)
        step(grid)

        # Death and renewal
        db, ds = death_and_renewal(grid)
        blind_deaths_total  += db
        seeing_deaths_total += ds

        # Render
        if s % SHOW_EVERY == 0:
            render(grid, s, cat)
            time.sleep(0.15)

        # Record history
        if s % 50 == 0:
            blind_s, seeing_s, adaptations = measure(grid)
            history.append({
                'step':             s,
                'blind_diversity':  blind_s['diversity'],
                'seeing_diversity': seeing_s['diversity'],
                'blind_deaths':     blind_deaths_total,
                'seeing_deaths':    seeing_deaths_total,
                'adaptations':      adaptations,
            })

    # --- Final Report ---
    print('\033[2J\033[H', end='')
    print("=" * 50)
    print("FINAL REPORT")
    print("=" * 50)
    print()
    print("The question: does the ability to see change")
    print("produce different behavior than blind persistence?")
    print()
    print(f"Total deaths (blind):   {blind_deaths_total}")
    print(f"Total deaths (seeing):  {seeing_deaths_total}")
    print()
    print("History:")
    print(f"{'Step':>6} {'Blind Div':>10} {'Seeing Div':>10} {'Adaptations':>12}")
    print("-" * 40)
    for h in history:
        print(f"{h['step']:>6} {h['blind_diversity']:>10.4f} "
              f"{h['seeing_diversity']:>10.4f} {h['adaptations']:>12}")
    print()
    print("What to look for:")
    print("- Does seeing side recover faster after catastrophe?")
    print("- Does seeing side maintain more diversity?")
    print("- Does seeing side have fewer deaths?")
    print("- Or does it make no difference at all?")
    print()
    print("This is step one. Observe. Then ask why.")


if __name__ == '__main__':
    run()
