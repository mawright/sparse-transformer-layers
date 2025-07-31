# Sparse multi-scale deformable attention

## Overview

This implements a version of Multi-scale Deformable Attention (MSDeformAttention) adapted for sparse tensors. 

---

::: blocks.ms_deform_attn.SparseMSDeformableAttentionBlock
    options:
        members:
            - forward
            - reset_parameters
        show_root_heading: true
        show_root_toc_entry: true
        show_root_full_path: false

---

::: layers.sparse_ms_deform_attn.layer.SparseMSDeformableAttention
    options:
        members:
        - forward
        - reset_parameters
        show_root_heading: true
        show_root_toc_entry: true
        show_root_full_path: false

--- 

## Utilities

::: layers.sparse_ms_deform_attn.utils
    options:
        members:
            - sparse_split_heads
            - multilevel_sparse_bilinear_grid_sample
        show_root_heading: false
        show_root_toc_entry: false
        show_root_full_path: false
        heading_level: 3