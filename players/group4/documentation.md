## Group 4 Strategy Overview

Group 4 implements a cooperative helper AI that emphasizes coverage, rare-species prioritization, and simple signaling. Every helper shares the same code path but adjusts behavior based on helper count and snapshot state. Core components:

- **Safe-Zone Exploration:** Helpers stay within the 1008-step Manhattan envelope around the Ark (shrinking dynamically once rain begins) so they can always make it home.
- **Territory Assignment:** Helpers slice the safe area into a grid by helper count and patrol their slice to reduce overlap. Noah ignores assignment.
- **Rarity-First Sweep (conditional):** When there are more than three helpers, early turns prioritize rarer species; when helper count is three or fewer, specialization is disabled and helpers grab any eligible animals immediately.
- **Messaging:** First message broadcasts region assignment; subsequent messages encode “returning” plus flock size in one byte so neighbors infer coverage and congestion.
- **Tracking & Contention:** Helpers track attractive cells; contested cells are temporarily blocked to avoid clumping. While returning, chasing is disabled (only in-cell upgrades allowed).
- **Return Policy:** Helpers head home on full flock or safety pressure (rain/limit). They resume the prior patrol target after unloading.

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

- **Scoring Function:** `_score_animal` penalizes duplicates and unknown genders, and ranks by rarity; lower tuples are better.
- **Best Animal in Cell:** `_best_animal_in_cell` filters out unavailable/duplicate/complete species. During the early sweep (only if >3 helpers) it prefers rare animals; if none are rare, it falls back to the best available.
- **Return Trigger:** `_should_return_to_ark` returns true on full flock or safety pressure (rain/safe-limit/endgame). Partial flocks keep exploring; with ≤3 helpers, specialization is off so they collect broadly.
- **Patrol & Tracking:** `_update_tracking_cell` picks the best visible cell unless returning; otherwise `_pick_new_patrol_target` chooses a point in the assigned region/safe area. Helpers resume their prior patrol anchor after unloading.
- **Blocked Cells:** Contested or emptied targets are blocked for a short timeout to avoid hovering.

### Movement and Obtain/Release Logic

1. **Get Action:**
   - If `_should_return_to_ark` is true ⇒ stop chasing, allow only in-cell upgrade (swap for rarer), and move to the Ark; after unloading, resume the saved patrol target.
   - Else, release the worst animal if a better one is present in our cell and the flock is full.
   - Else, attempt to obtain the best animal in our current cell (rare-first if sweep active, otherwise best available).
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
