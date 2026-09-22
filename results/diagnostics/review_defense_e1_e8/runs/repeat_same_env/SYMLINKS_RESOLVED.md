The original run directory linked `audit/`, `config/`, `data/`, and `rendered/`
to the shared grounded package rather than holding its own copies. Those four
links were removed here to avoid duplicating ~1.1 GB of identical content.

Resolve them against the package copies:

| original link | replacement inside this package |
| --- | --- |
| `audit`    | `results/rq2_reliability/audit/` |
| `config`   | `config/protocols/protocol_rq2_rq3_grounded_3lang.json`, `config/prompts/` |
| `data`     | `data/pairs/rq2_pairs_3000_seed42.csv`, `data/render_metadata/` |
| `rendered` | `images/rq2_rendering_grid/`, `images/rq2_perturbation/` |

Only `inference/`, which holds this run's own outputs, is kept.
