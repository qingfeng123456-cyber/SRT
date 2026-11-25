# SRT: Student-Response Transport

Code and compact returned results for **From Revised Supervision to Deployable Forecasts: Student Responses and Delayed Transfer in Equity Prediction**, by Weijun Xia.

SRT follows the output response retained after changing a student's supervision. A shared causal forecasting architecture supplies three independent neural fits. Matched ridge heads form response dictionaries; recursive least squares, delayed state control, same-information controls and component ablations share the frozen forecasts.

## Repository

- `main.py`: single GPU experiment entry point.
- `src/`: forecasting, response dictionaries, delayed controllers, inference and portfolio proxy.
- `config.json` and `experiment_manifest.json`: recorded fixed protocol and comparison families.
- `tools/prepare_data.py`: public Qlib snapshot preparation and checksum matching.
- `tools/verify_data.py`: checksum-only verification of prepared data.
- `results/`: original aggregate exports and three saved training histories.
- `docs/`: protocol, data format and source provenance.

## Environment and run

Use a CUDA-enabled PyTorch environment on AutoDL or an equivalent GPU server. The returned experiment used an RTX 2080 Ti. Install the listed dependencies explicitly; the program never installs packages.

```bash
pip install -r requirements.txt
python tools/prepare_data.py --source /path/to/provider/qlib_cn
python tools/verify_data.py
python main.py
```

The source argument is the provider's Qlib directory containing `calendars`, `instruments` and `features`. See `data/README.md`. The financial input is closed at 2021-12-31. `python main.py --sanity-only` runs the fixed synthetic pipeline on the GPU server without financial fitting.

A full run executes the fixed controlled panels, three financial neural fits, frozen-head comparisons, delayed feedback analysis and the portfolio proxy. A returned `RUN-...-SRT-evidence.zip` preserves all outputs and statuses, including adverse and unidentified results.

## Recorded evidence

The returned run is `RUN-20261020-140616-102758-SRT`. Recurring alignment gives known-mean MSE 0.179715 for SRT versus 0.661727 for matched recursive least squares; an unseen reversal favors direct coefficient adaptation. In the two CSI300 development periods, return-MSE point estimates fall by 5.30% and 2.36% against the matched direct head, while ranking and the portfolio proxy favor the native direct benchmark. All recorded family-adjusted p values are at least 0.05. The compact exports retain the complete comparisons.

## License and citation

The research code is provided under the MIT license. Market data are obtained separately from the identified provider and follow that provider's terms. The manuscript's supplementary material is supplied through the journal submission. `CITATION.cff` supplies the author and manuscript citation.
