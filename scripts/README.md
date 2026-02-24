# Top-level scripts

These scripts are repository-level utilities:

- `init.sh`: bootstrap runtime environment (`PYTHONPATH`, LCG stack when available).
- `manage-submodules.sh`: helper for git submodule setup/update.
- `validate_job_config.py`: validate JSON job/training configuration files.

Task-specific training and HTCondor wrappers are kept under:

- `channel_tagging/scripts/`
- `electron_direction/scripts/`
