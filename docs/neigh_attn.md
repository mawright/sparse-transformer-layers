# Multi-level sparse neighborhood attention

## Overview

The multi-level sparse neighborhood attention operation allows query points to attend to the small neighborhoods of nonzero points around their spatial position, one neighborhood for each feature level.
This is a potentially useful alternative or complement to multi-scale deformable attention, which can potentially try to sample from zero points on sparse tensors. The neighborhood attention operation, on the other hand, will always attend to all nonzero points within the given neighborhood sizes.

The neighborhood attention implementation makes use of a custom autograd operator that checkpoints the key and value projections of the neighborhood points and manually calculates the backward pass.
This checkpointing is essential for memory management, particularly for operations with many potential query points such as within a DETR encoder, or a DETR decoder with many object queries.

---

::: blocks.neighborhood_attn.SparseNeighborhoodAttentionBlock
    options:
        members:
            - forward
            - reset_parameters
        show_root_heading: true
        show_root_toc_entry: true
        show_root_full_path: false

---

::: blocks.neighborhood_attn
    options:
        members:
            - get_multilevel_neighborhoods
        show_root_heading: false
        show_root_toc_entry: false