# docs

Key documents:

- [FRAMEWORK_DESCRIPTION.md](FRAMEWORK_DESCRIPTION.md): repo/code structure + data formats + workflows
- [Training.md](Training.md): concrete commands to run trainings + analyses
- [ED_v50_Architecture.md](ED_v50_Architecture.md), [ED_v58_Architecture.md](ED_v58_Architecture.md): ED model architecture notes
- [BestModels.dat](BestModels.dat): tracked best model runs

## Repository hygiene notes

- Keep only hand-maintained submit templates (`.sub`) under task `condor/` folders.
- Generated submit files (outside `channel_tagging/condor` and `electron_direction/condor`) should remain untracked.
- Condor runtime outputs should not be committed; only `.gitkeep` files are retained for `logs/`, `error/`, and `output/`.
