# visualize/ (docs)

Helper tooling and docs for **audio → gesture** generation and the in-browser
3D viewer. Everything here is additive on top of the upstream GestureLSM repo;
you can ignore the `visualize/` directory entirely and the original pipeline
still works.

> Note: these docs were moved into `agents_info/`. The actual code/data files
> (`trim.py`, `output/visualize.html`, `input/`, `output/`) still live under
> `../visualize/` at the repo root.

## Documentation

| File | What it covers |
|---|---|
| [VISUALIZATION.md](./VISUALIZATION.md) | The end-to-end visualization pipeline. `gen.py` outputs, `visualize.html` viewer, NPZ schemas, caching, trim helper, browser limitations, common gotchas. **Start here.** |
| [omnicontrol_spatial_guidance.md](./omnicontrol_spatial_guidance.md) | Plan for adding OmniControl-style training-free spatial guidance at inference time. Architecture, integration points in `models/Diffusion.py`, chunked guidance scope, challenges. **Read before implementing control.** |

## Code (under `../visualize/`)

| File | Purpose |
|---|---|
| `../visualize/trim.py` | Trim `.npz` + `.wav` to a minimum-length test pair (default 150 frames → exactly one inference chunk). Run `python visualize/trim.py`. |
| `../visualize/output/visualize.html` | Static Plotly viewer. Drag-and-drop loader for `gen_position.npz` + `gt_position.npz` + `speech.wav`. Open in Chrome/Firefox (not VSCode Simple Browser). |

## Data folders (gitignored, under `../visualize/`)

| Path | Contents |
|---|---|
| `../visualize/input/` | Input audio + GT npz (e.g. `2_scott_0_1_1.wav` + `.npz`, and `_128f`-suffixed trimmed versions). |
| `../visualize/output/` | Per-run generation outputs: `gen_rotation.npz`, `gen_position.npz`, `gt_position.npz`, `speech.wav`. The viewer (`visualize.html`) lives here too. |
| `../visualize/output_128f/`, `output_*` | Alternate output dirs (e.g. for the trimmed one-chunk test). |

## Quick start

```bash
# generate (defaults: visualize/input/2_scott_0_1_1.wav → visualize/output/)
python ../gen.py        # from inside visualize/  OR  `python gen.py` from repo root

# open viewer in Chrome / Firefox, drag the visualize/output/ folder onto the drop zone
xdg-open ../visualize/output/visualize.html
```

See [VISUALIZATION.md](./VISUALIZATION.md) for the full story (CLI flags,
output schemas, cold/warm timing, dataloader hook, etc.).
