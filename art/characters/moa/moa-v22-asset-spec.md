# Moa v22 asset contract

## Decision

`moa-ungoo-benchmark-clips-v21` remains a technically valid delivery set but
is not the visual-production target. The v22 path replaces its source art;
palette cleanup or nearest scaling alone must not promote it.

## Runtime contract

- View: east-facing side view, 2D battle sprite.
- Final canvas: 244 x 247 RGBA.
- Foot anchor: x=123, y=243.
- Read at: 1x battle scale first; 4x nearest is inspection only.
- Core silhouette: low forward lean, visible boots, one dominant right-hand
  blade diagonal, secondary left blade.
- Afterimage: one short cyan ribbon attached behind the body with at most one
  small dark core. It must not read as a second actor.

## Production standard

- Render complete native pixel clusters for each final key pose; do not
  downsample an anime illustration as a production frame.
- Use 28-36 shared visible colors, navy/charcoal/silver dominant, with cyan and
  violet reserved for weapon/afterimage accents.
- Idle/walk retain a compact body envelope. Attack uses six authored poses:
  settle, load, first cut, third-beat cross cut, impact, recovery.
- No white or checkerboard pixels may be baked into the alpha background.

## Candidate record

`moa-v22-master-candidate-a.png` was generated from the v2 identity reference
for silhouette review. It is rejected as a production source because its
checkerboard background is baked (no alpha channel) and its separate drone
still competes with Moa's silhouette.

`moa-v22-idle-candidate-d-green.png` is the current idle identity candidate.
After nearest-neighbour 244 x 247 preparation and 64-colour reduction, its
actual battle envelope is 136 x 182 px. It passes the v22 canvas, palette, and
single-core/ribbon read, but is a one-frame source study only: it is not yet a
six-frame idle clip or a promoted runtime asset.

`moa-v22-attack-crosscut-candidate-e-green.png` establishes the third-beat
cross-cut body compression (148 x 162 px at 1x). It remains a rejected key-pose
study because its cyan X reaches too far from the body and turns the effect into
the primary read. The final impact frame must keep both diagonal trails inside
the compact body envelope.

`moa-v22-attack-crosscut-candidate-g-green.png` supersedes E as the accepted
cross-cut source study. Its 144 x 161 px/61-colour 1x output keeps the cyan X
at Moa's hip instead of extending across the canvas, while the detected foot is
translated to `(123,243)`. It remains a source study until its preceding and
recovery poses use the same compact blade lengths.

`moa-v22-attack-load-candidate-f-green.png` is the accepted load-pose source
study: it has a 104 x 151 px body envelope at 1x, 63 output colours, and keeps
both blades gathered across the torso. It establishes the v22 attack start
silhouette, but is not a runtime frame until the adjacent authored poses match
its stance and palette.

`moa-v22-logical-grid-master-candidate-h-green.png` and the derived 27-colour
244 x 247 study are rejected. The source proves that a 64px logical-grid brief
can reduce decorative noise, but its 2x enlargement loses facial, hand, and
weapon construction. Do not promote automatic grid reduction as a replacement
for pixel-lead redraw.

## Pipeline contract

`moa-v22-production-profile.json` preserves authored 244 x 247 placement,
keys flat green source backgrounds at tolerance 160, and rejects palettes above
36 colours. Candidate source preparation uses point resampling and a 36-colour
non-dithered reduction before AssetForge ingest; this prevents the benchmark
profile from enlarging a compact source to full-canvas scale.

Source studies must be translated before ingest so their detected foot is at
`(123, 243)`. The legacy proof set (`idle D`, `load F`, and `cross-cut E`) is
kept only for history; it must not be used to promote an attack clip.

## Pixel master lock

`v22-master/moa-v22-pixel-master-idle.png` is the only palette and silhouette
source for the first v22 pass. It is a real 244 x 247 RGBA file with 31 visible
colours, 136 x 182 px content, and foot `(123,243)` after AssetForge ingest.
`v22-master/moa-v22-palette.png` materializes that palette for review.

Every new attack pose starts by duplicating this master, then changes only the
following clusters: pelvis/torso lean, head angle, forearm and blade angle,
near leg overlap, and core/ribbon position. Do not introduce a colour that is
absent from the palette image, or a new costume micro-detail. The source master
is not automatically a runtime idle clip; the Codex-layer pipeline must first
simplify its hair, jacket, and weapon clusters before the six-pose set is
promoted.

## Codex key-pose proof set

`build/moa-v22-attack-arc/keyposes-final/` is the current six-key-pose proof
set. It contains `attack_00` through `attack_05` in this order: settle, load,
first cut, cross-cut, impact, recovery. All six are 244 x 247 RGBA with foot
`(123,243)` and 27-30 visible colours after the final shared ingest. The
contact sheet is `build/moa-v22-attack-arc/keyposes-final/_contact.png`.

The proof set is deliberately not a runtime promotion by itself: the production
profile requires a 14-frame attack clip. The current 14-frame candidate uses
real Codex-generated transitions, not duplicated holds or synthetic
interpolation. Its attack identity allowance is `0.65` height/width drift to
cover the intentional crouch-to-lunge envelope; release validation must still
pass the shared anchor, alpha, palette, and contact-sheet review.

## Promotion bar

No raw generated source, nearest-neighbour reduction, or palette quantisation
is a production promotion by itself. Codex must generate regulated layers from
the locked master, then AssetForge must extract, anchor, palette-lock, and
review: silhouette at 1x, readable blade hand, three-value material clusters,
one-frame turnaround, and the six-pose attack arc before exporting a v22
runtime set.
