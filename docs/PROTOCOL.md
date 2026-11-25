# Recorded SRT protocol

The return target is `100*log(close[t+5]/close[t+1])`, using four close-to-close increments. Full feedback matures at origin plus five trading dates. Inputs comprise 64 historical dates with 26 values and 26 observation indicators. The causal encoder has width 48, kernel size 5 and dilations 1, 2 and 4, with origin-cohort context and prefix branches. Return and rank channels have separate state models and actions.

The A period is 2008–2013, B is 2014–2015 and C is 2016–2018. Chronological selection uses 2019 Q1, controller estimation uses 2019 Q2–Q4, and checking/warming use 2020 Q1/Q2. The reported development periods are 2020 H2 and 2021. The recorded data cutoff is 2021-12-31.

The sixteen financial methods share three neural fits and frozen forecast information. The four response-by-controller cells are R0C0, R1C0, R0C1 and SRT. The comparison families contain thirty RankIC and eighteen return-MSE slots. The ten controlled panels cover projection, recurrent and unseen alignment, covariate geometry, common market shifts, rank-mean mismatch and three null settings, with delays 5, 10 and 20.

The closed portfolio proxy selects the top decile by the rank head, uses four sleeves, enters at t+1, schedules exit at t+5 and charges ten basis points on each side. The code preserves its missing-mark and deferred-exit conventions.

The returned training histories select epochs 7, 6 and 9 for T0, TF and the student, stopping at epochs 12, 11 and 14 after five non-improving selection epochs. Each fit has 57,410 parameters. The controlled returns contain 927 completed and 33 completed-unidentified method-channel slots.
