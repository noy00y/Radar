# FMCW Radar / STAP Reference Pipeline — Milestone 1

This repository starts with a **synthetic, single-TX/single-RX FMCW IF simulator** and a range-Doppler reference processor. It is intentionally not yet a hardware decoder or a complete STAP implementation.

## Important
`configs/simulation.yaml` contains illustrative values, **not a verified IWR1843BOOST configuration**. Do not use it to configure the board. Before processing DCA1000 data, capture settings and ADC packing must be documented and validated.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

## Run

```bash
python scripts/run_simulation.py
pytest
```

The script prints target truth versus estimated peak locations and writes `range_doppler.png`.

## Current signal model
For each target, the simplified simulator uses:
- round-trip delay `tau = 2R/c`;
- monostatic Doppler `fd = 2v/lambda`;
- beat frequency approximately `fb = slope*tau - fd`;
- slow-time phase progression from Doppler.

Positive radial velocity is defined as receding. **The sign convention must be checked against the eventual TI capture/configuration convention.**

## Known limitations
- one receive channel only; `num_rx` is reserved for later work;
- no angle processing or antenna manifold yet;
- no phase noise, TX leakage, multipath, ADC quantization, or clipping;
- no TDM-MIMO modeling;
- ideal constant-velocity point targets;
- no CFAR or STAP yet;
- simple positive beat-frequency range axis;
- physical IWR1843/DCA1000 capture decoding not implemented.

## Next milestone
1. verify and document the actual radar chirp configuration;
2. add multichannel and clutter scene simulation;
3. implement a DCA1000 ADC decoder using a known capture format;
4. validate range/Doppler axes against real measurements.
