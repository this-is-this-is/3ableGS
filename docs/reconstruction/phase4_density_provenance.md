# Base v4 density schedule restoration

The `combo_perceptual_02` configuration saved in
[E03](evidence/E03_combo_overlay.txt) changes only Background control settings:
`stop_split_at: 25000` and `stop_reset_at: 20000`. Its source is the historical
2026-10-12 session JSONL, line 120, recorded at 01:21:43 UTC.

The exact implementation is present in
[E04](evidence/E04_loss_implementation.txt), lines 360–386: the existing reset
condition gains `and self.opacity_reset_is_active()`, and that method returns
`stop_reset_at is None or self.step < stop_reset_at`. Its source is the same
session JSONL, line 135, recorded at 01:21:50 UTC. The restored implementation
copies this condition and helper into `models/gaussians/feature.py`.

The 3000-step interval and 100-step offset remain intact. Consequently the final
Background opacity reset is at step 18100; the otherwise scheduled reset at
21100 is skipped. Split/clone eligibility still uses the original modulo guard
even after opacity resets stop. Background splitting is permitted at 24900 and
disabled at 25000. Pruning continues under its original schedule beyond both
cutoffs. The optional `stop_cull_at` and GDAGS experiments also shown in E04 are
not part of this Base v4 reconstruction.

An absent or null `stop_reset_at` preserves Base v3 resets. The independent
RigidNodes refinement implementation and DeformableNodes subclass retain their
20000 split cutoff and original resets. SMPLNodes overrides the training
postprocess hook to update KNN only, bypassing inherited refinement entirely;
its inherited general split cutoff of 15000 is dormant. The Background overlay
does not change any of these behaviors.

`tests/test_base_v4_density.py` runs actual refinement calls on small CPU
populations and initialized Adam states. It verifies opacity caps, opacity
moment clearing/preservation, strict reset cutoff, Background split boundaries,
the retained modulo window, continued pruning, Rigid/Deformable split/reset
behavior, and SMPL's training-hook bypass of topology changes and resets.
