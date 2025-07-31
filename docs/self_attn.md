# Multi-level sparse self-attention

## Overview

The self-attention implementation is intended for use with `torch.sparse_coo_tensor` multi-level feature maps. It uses [`RoPEEncodingND`](https://mawright.github.io/nd-rotary-encodings/layer/#position_encoding_layer.rope_encoding_layer.RoPEEncodingND) from [nd-rotary-encodings](https://github.com/mawright/nd-rotary-encodings) to encode the positions and feature levels of all input points.

---

::: blocks.self_attn.MultilevelSelfAttentionBlockWithRoPE
    options:
        members:
            - forward
            - reset_parameters
        show_root_heading: true
        show_root_toc_entry: true
        show_root_full_path: false