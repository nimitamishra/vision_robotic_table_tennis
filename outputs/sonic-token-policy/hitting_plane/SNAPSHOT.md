# Preserved simulation version: bracket-v4

Frozen on 2026-11-20. `final-version.json` records SHA-256 checksums of the local final run. The committed `runs/bracket-v4/policy.ts` is the final exported four-input, 64-output model, not an intermediate checkpoint.

The repository includes the weights, residual basis, metadata, complete evaluation summary, training progress and one state replay. Bulk videos, additional per-frame test replays and interactive recordings remain in the local final ZIP; regenerate videos with `render.py` and a fresh evaluation with `evaluate.py`. Checksums for those local assets are retained for provenance.

The frozen SONIC decoder, robot meshes and public motion-kit data are external dependencies. Paths in launchers and `decoder_env.py` reflect the development workstation; configure these paths when using another machine. Clone `https://github.com/dvij542/table-tennis-motion-kit` into repository `work/table-tennis-motion-kit` at commit `c6c4c061f8fcb24714fbe118a31e66c69adeb4bf`. The base prior and decoder seed are included explicitly despite the general run-file exclusions.

Validation before freezing: 5 pipeline tests, 7 legacy ball-return tests, G1 Studio offline test, and exported-model equivalence on 200 inputs passed. Fresh evaluation: 20/20 face contacts, 17/20 legal returns, mean deadline error 4.27 cm. These are narrow-range simulation results; hardware approval remains false.
