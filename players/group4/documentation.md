## Group 4 Strategy Overview

Group 4 implements a cooperative helper AI that emphasizes coverage, rare-species prioritization, and simple signaling. Every helper shares the same code path, but diverges in behavior based on its helper index and the state observed in the simulator snapshots. The core components of the strategy are:

- **Safe-Zone Exploration:** Helpers never wander farther than 1008 manhattan steps from the Ark (the maximum distance that can be closed once the rain timer starts). We pre-compute a bounding box around the Ark as the "safe diamond."
- **Territory Assignment:** All helpers except Noah are assigned to pseudo-quadrants of the safe zone (computed by slicing the safe area into a grid whose width/height match the helper count). Each helper picks patrol targets within its region until the region is exhausted, then samples a random safe point elsewhere. This reduces redundant coverage.
- **Rarity-Based Priorities:** We build a stable ranking for each species using the known population counts. Animals belonging to rarer species are prioritized. Duplicates of the same species inside a helper's flock incur penalty points so we prefer collecting four unique species before returning.
- **Messaging:** Every helper broadcasts a single byte per turn. The first time a helper gets to act it broadcasts its region assignment (high-bit set). After that, it uses the top bit to signal "I'm returning" and the lower bits to share flock size, allowing neighbors to know roughly who is carrying animals or vacating an area.
- **Tracking & Contention Handling:** Helpers track sightings inside their five-kilometer vision radius. Once they chase a cell, they record the cell as "blocked" if someone else wins the obtain race. Blocked cells expire after a few turns, which prevents idle helpers from clumping together.
- **Return Policy:** Helpers return to the Ark when (a) the rain timer triggers, or (b) they own four unique species, or (c) they have a full flock with four unique species. During return travel, they drop all chasing behavior and instead path directly back to the Ark.

## Detailed Behavior Flow

### Snapshot Handling

Each turn, the simulator provides a `HelperSurroundingsSnapshot` containing:

1. Current position and rain flag.
2. A `Sight` object with all visible cells (including animals and helpers in each).
3. Flock contents and, if standing on the Ark, the Ark inventory view.

The helper immediately synchronizes its internal `position`, `flock`, and `sight` state, updates `species_on_ark` using the Ark view (to avoid retrieving duplicates), and checks whether a previously attempted Obtain succeeded. Failed obtains are logged in `blocked_cells` and `unavailable_animals` so we avoid repeatedly targeting animals already seized by other helpers.

### Messaging Protocol

- **Assignment Broadcast:** On the first turn, every helper publishes `0x80 | region_index` (region indices wrap around 0‒63) so peers know its coverage sector.
- **Return Status:** If `_should_return_to_ark()` is true, set bit `0x40` to signal intent to return home.
- **Flock Load Report:** Use the lower three bits to encode `min(len(flock), 7)`, giving a rough sense of load.

Incoming messages are decoded per turn to maintain `known_assignments` and a set of `helpers_returning`. In this baseline we do not alter our behavior based on peers' return signals, but the data structure is present for future coordination (e.g., reassigning regions when someone leaves early).

### Target Selection and Prioritization

- **Scoring Function:** `_score_animal` returns a tuple that sorts first by "duplicate species penalty" (are we already carrying this species?), then by species population (rarity), whether we still need the gender, duplicate counts, unknown gender penalty, and species ID. Lower tuples represent better animals.
- **Best Animal in Cell:** `_best_animal_in_cell` iterates visible animals in a cell, skipping ones we already have or that are marked unavailable. It picks the minimal score tuple.
- **Return Trigger:** `_should_return_to_ark` returns true when raining, when carrying ≥4 unique species, or when the flock is full *and* contains four unique species. This ensures we keep exploring until we can maximize scoring potential.
- **Patrol & Tracking:** `_update_tracking_cell` scans the sight radius for attractive cells and picks the best one. If none exists or if we've reached the target and it no longer has animals, we fall back to `_pick_new_patrol_target`, which chooses a random location inside our assigned region. The helper constantly moves toward either the tracking cell or the patrol target.
- **Blocked Cells:** If a tracked cell becomes contested (another helper is standing in the cell), we add it to `blocked_cells` with a short timeout to prevent tailgating.

### Movement and Obtain/Release Logic

1. **Get Action:**
   - If `_should_return_to_ark` is true ⇒ move directly towards the Ark.
   - Else, release the worst animal if a better one is present in our cell and the flock is full.
   - Else, attempt to obtain the best animal in our current cell (if any).
   - Else, chase the best tracking/patrol target.

2. **Random Jitter:** If `move_towards` results in less than ~0.05 km displacement (due to clamping), `_random_safe_step` triggers. This ensures helpers keep moving even when targets are too close or obstacles clamp their move.

3. **Release Heuristic:** `_maybe_release_for_priority` compares the worst-scoring animal in our flock against the best candidate in the current cell. If the flock animal is strictly worse, we release it to make space.

4. **Per-Cell Filtering:** `_select_animal_here` respects the flock capacity and uses `_best_animal_in_cell` to pick an animal in the current cell. `_prune_unavailable_animals` intersects the `unavailable_animals` set with the current cell to allow re-targeting once contested animals leave the area.

### Safety & Rain

- **SAFE_MANHATTAN_LIMIT:** By keeping every helper within 1008 manhattan steps, we guarantee that even if rain starts at the worst possible time, helpers can reach the Ark within the 1008-turn deadline.
- **Return-to-Ark Shortcut:** Once `_should_return_to_ark` is true, the helper clears `patrol_target` and `tracking_cell` to prevent them from reassigning new targets mid-retreat. The movement logic falls through to "walk straight home" each turn, ensuring animals are offloaded promptly.

## Future Improvements

- **Adaptive Territory Rebalancing:** Use incoming assignment broadcasts plus knowledge of who is returning to dynamically reassign coverage when distant helpers leave their quadrant.
- **Gender-aware Messaging:** Encode more sophisticated states (e.g., species IDs currently carried) so helpers can steer away from duplicates early.
- **Animal Density Estimation:** Maintain heat maps over time to focus effort on sparse regions when rare species counts are low.

This documentation should serve as a reference for how Player4 is architected and how each subsystem collaborates during the simulation. Refer to `player.py` for source-level details; the file is heavily commented with matching section headers.
