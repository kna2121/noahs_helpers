import json
from pathlib import Path

# ---- knobs from the tournament spec ----

# Interpreting 1+1, 2+1, 8+1 as total helper counts: 2, 3, 9.
helpers_options = {
    "1p1": 2,
    "2p1": 3,
    "8p1": 9,
    "25": 25,
    "60": 60,
    "100": 100,
}

num_species_options = [1, 7, 14, 30, 100, 500, 1398]

# base animals per species for each pure distribution
distributions = {
    "unicorn": 2,       # Unicorn: 2 animals per species
    "rare": 6,          # Rare: 6 animals per species
    "intermediate": 20, # Intermediate: 20 animals per species
    "common": 100,      # Common: 100 animals per species
}

# three example ark positions (replace with Kristo's if needed)
ark_positions = [
    [150, 150],
    [500, 500],
    [850, 130],  # from your example config
]

# ---- output directory ----
out_dir = Path("tournament_config")
out_dir.mkdir(exist_ok=True, parents=True)

# ---- generate configs ----
for helpers_label, num_helpers in helpers_options.items():
    for num_species in num_species_options:
        for dist_name, per_species_count in distributions.items():
            # construct animals list: same base count per species
            animals = [per_species_count] * num_species

            for ark_idx, ark in enumerate(ark_positions, start=1):
                cfg = {
                    "num_helpers": num_helpers,
                    "animals": animals,
                    "ark": ark,
                }

                # descriptive filename
                filename = f"h{helpers_label}_s{num_species}_{dist_name}_ark{ark_idx}.json"
                out_path = out_dir / filename

                out_path.write_text(json.dumps(cfg, indent=2))

                print(f"wrote {out_path}")
