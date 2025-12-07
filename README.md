# HQTFIM Néel Cat – Hierarchical vs Standard TFIM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/AnassGaroual/hqtfim-neel-cat/actions/workflows/ci.yml/badge.svg)](https://github.com/AnassGaroual/hqtfim-neel-cat/actions/workflows/ci.yml)

This repository contains the numerical code and data pipeline for the paper:

> *Exact Properties of Néel Cat States and Constant Entanglement  
> at a Special Ordered Point of the Hierarchical Quantum Ising Model*  
> **Anass Garoual**, 2025.

Zenodo record (preprint PDF):  
**DOI:** [10.5281/zenodo.17849908](https://doi.org/10.5281/zenodo.17849908)

The code performs finite-size exact diagonalization of:
- the hierarchical quantum transverse-field Ising model (HQTFIM), and
- the standard 1D nearest-neighbour transverse-field Ising chain,

and compares:
- fidelity with an ideal Néel-cat state,
- half-chain entanglement entropy,
- staggered order parameter,
- effective exponents vs system size and hierarchy exponent α,
- the RG self-similarity field \(h^\star(\alpha)\) vs the true critical field \(h_c(\alpha)\).

## Install

```bash
git clone https://github.com/AnassGaroual/hqtfim-neel-cat.git
cd hqtfim-neel-cat

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
