# sparse-transformer-layers

[![Tests](https://github.com/mawright/sparse-transformer-layers/actions/workflows/tests.yml/badge.svg)](https://github.com/mawright/sparse-transformer-layers/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/mawright/sparse-transformer-layers/branch/main/graph/badge.svg)](https://codecov.io/gh/mawright/sparse-transformer-layers)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
[![License](https://img.shields.io/github/license/mawright/sparse-transformer-layers)](https://github.com/mawright/sparse-transformer-layers/blob/main/LICENSE)

Transformer layers for PyTorch sparse tensors

(Readme is a work in progress)

## Introduction

This repository contains several advanced Transformer-based layers meant for spatial data that may be large, spatially sparse, and/or irregularly structured.
In particular, they are oriented towards Transformer-based object detectors (i.e., DETRs) that operate on multi-level feature pyramids.
In particular, the repository contains implementations of:

- Multilevel self-attention
- Multilevel sparse neighborhood attention
- Sparse multi-scale deformable attention (MSDeformAttention)

The primary envisioned scenario for use is object detection and/or segmentation on large, spatially-sparse images or volumes.
In this setting, the standard implementations of spatial attention operations may not be applicable due to the size and irregularity of the data.
This repository builds on the related libraries [pytorch-sparse-utils](https://github.com/mawright/pytorch-sparse-utils) and [nd-rotary-encodings](https:///github.com/mawright/nd-rotary-encodings) for flexible, performant Transformer operations.