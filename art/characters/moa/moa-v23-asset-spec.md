# Moa v23 asset contract

## Decision

v22 is retained as the current runtime/release set, but v23 is a clean visual
rebuild. The combat role remains a fast east-facing dual-blade striker; the
costume, silhouette, weapon construction, and palette are being re-authored
from a new Codex master rather than patched from v22.

## Runtime contract

- Final canvas: 244 x 247 RGBA.
- Direction: east-facing strict side view.
- Foot anchor: x=123, y=243.
- Read at 1x first; 4x nearest is inspection only.
- Main read: long segmented forward blade, compact off-hand blade, silver
  shoulder mantle, high-collar indigo coat, ash-silver hair with geometric
  clasp, silver boots.
- Exactly one short cyan hip ribbon; no cape and no second actor.

## Visual bar

- Native pixel clusters with crisp hard edges; no smooth illustration downsample
  as a final frame.
- 30-36 shared visible colors, with navy/charcoal/silver dominant and cyan /
  violet reserved for weapon nodes and energy accents.
- Three value steps per material, readable face/hand/boot construction, and a
  strong asymmetric silhouette comparable in density to the Starline reference
  characters.
- Idle and walk must stay inside a compact 160-190 px height envelope. Attack
  may extend horizontally through the main blade, but the body remains legible.

## Current candidate

`v23-master/moa-v23-master-c-clean.png` is the preferred master study.
It is a new Codex-generated restrained field-gear design with a cleaned alpha
background, prepared and ingested by AssetForge at 90% source scale: 244 x 247
canvas, 179 x 171 content box, foot `(123,243)`, and 36 visible colors.
Candidates A and B remain as alternate studies. The clean master still needs a
1x silhouette review while authoring the poses; idle, walk, hit, death, and the
full 14-frame attack must be authored from it before replacing v22.

## Promotion bar

Do not connect v23 to Starline or deploy it until the six-pose proof set passes
1x review, all required clips pass canvas/anchor/palette checks, and a real
Godot/Web battle capture confirms the new identity reads at runtime.
