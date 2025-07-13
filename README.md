# SGMT

**SGMT** (Slot-Guided Modular Transformer) is a PyTorch implementation of a Transformer architecture designed for **systematic compositional generalization**. SGMT breaks each input sequence into a small number of *slots* (using Slot Attention) and routes each slot through one of several tiny expert modules. This “slot + sparse module” design lets SGMT learn near-symbolic subroutines internally and achieve > 99 % token accuracy on challenging compositional benchmarks like SCAN, while using far fewer resources than large dense models.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![CI](https://github.com/rajbudigam/SGMT/actions/workflows/ci.yml/badge.svg)](https://github.com/rajbudigam/SGMT/actions)

---

## Features

- **Slot Attention Bottleneck**  
  Distills the encoder output into a fixed number of exchangeable slot vectors.  
- **Sparse Module Routing**  
  Routes each slot through one of *N* small MLP modules via hard Gumbel-Softmax.  
- **Minimal Footprint**  
  ~ 20 M parameters, ∼ 25 % fewer FLOPs than a dense baseline.  
- **State-of-the-Art Compositionality**  
  ≥ 99 % token accuracy on SCAN splits (length, productivity, novel primitives).  
- **Interpretable**  
  Slot–feature correlations and balanced module usage reveal clear emergent structure.  

---

## Installation

Clone and install:

```bash
git clone https://github.com/rajbudigam/SGMT.git
cd SGMT
pip install -r requirements.txt
pip install .
