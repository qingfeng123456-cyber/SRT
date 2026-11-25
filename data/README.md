# Data acquisition and preparation

Provider snapshot: https://github.com/chenditc/investment_data/releases/tag/2024-12-06

Obtain and extract the provider's China equity Qlib snapshot under its own terms. Supply the directory that directly contains `calendars/day.txt`, `instruments/csi300.txt` and `features/`:

```bash
python tools/prepare_data.py --source /path/to/qlib_cn
python tools/verify_data.py
```

The preparation utility reads the recorded binary prefixes through 2021-12-31, clips membership intervals, reproduces the text-file line endings and requires every output hash to match `expected_source_manifest.json`. It then installs `source_manifest.json` for the experiment loader. It never imports experiment modules or reads later quote payloads. The expected manifest records seven missing provider files. The financial arrays are not bundled in this repository.

The provider's effective membership intervals and corporate-action conventions define the input panel. The repository's MIT license covers the research code, not the separately acquired market data.
