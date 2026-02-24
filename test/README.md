# test

This folder hosts lightweight end-to-end smoke tests.

These tests are designed to validate that:

- the **training code** for CT and ED runs on a tiny local sample
- the main **analysis (ana) apps** can consume those outputs and produce PDFs

## What runs

- `test/make_tiny_fixtures.py` generates tiny `.npz` fixtures under `test/artifacts/fixtures/`.
- `test/test_ct_training.sh` runs CT training on the fixtures and then runs `channel_tagging/ana/comprehensive_ct_analysis.py`.
- `test/test_ed_training.sh` runs ED training on the fixtures and then runs:
	- `electron_direction/ana/comprehensive_ed_analysis.py`
	- `electron_direction/ana/cnn_feature_interpretation.py`

## How to run

```bash
./test/testAllApps.sh
```

If you see environment-related failures, make sure you are running from a shell where `scripts/init.sh` works.

