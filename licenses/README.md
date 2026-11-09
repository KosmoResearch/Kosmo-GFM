# Third-party licenses

Full license texts are retained here. See [NOTICE](../NOTICE) for the
corresponding components and modifications. Trailing whitespace has been
normalized; license wording is unchanged.

| License file | Source |
| --- | --- |
| [VGGT-LICENSE.txt](VGGT-LICENSE.txt) | [Upstream license](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt) |
| [DINOv2-LICENSE.txt](DINOv2-LICENSE.txt) | [Upstream license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) |
| [DPT-LICENSE.txt](DPT-LICENSE.txt) | [Upstream license](https://github.com/isl-org/DPT/blob/main/LICENSE) |
| [Depth-Anything-3-LICENSE.txt](Depth-Anything-3-LICENSE.txt) | [Upstream license](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/LICENSE) |
| [DeDoDe-LICENSE.txt](DeDoDe-LICENSE.txt) | [Upstream license](https://github.com/Parskatt/DeDoDe/blob/main/LICENSE) |
| [DaD-LICENSE.txt](DaD-LICENSE.txt) | [Upstream license](https://github.com/Parskatt/DaD/blob/main/LICENSE) |
| [LightGlue-LICENSE.txt](LightGlue-LICENSE.txt) | [Upstream license](https://github.com/cvg/LightGlue/blob/main/LICENSE) |
| [PyTorch3D-LICENSE.txt](PyTorch3D-LICENSE.txt) | [Upstream license](https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE) |

## Provenance checks required before release

The following gaps cannot be resolved from the current source tree alone:

- Confirm the direct source repository and revision for
  `gfm/models/heads/dpt_blocks.py`, `gfm/models/layers/conv_blocks.py` and
  `gfm/models/heads/depth_head.py`. The scratch, residual and fusion blocks
  share code with the DPT family and VGGT adaptations. The included Intel DPT
  MIT notice preserves that upstream attribution; it is not evidence that all
  intermediate adaptations are MIT or that these files are solely original
  Apache-2.0 code. Retain additional notices required by the actual source.
- Identify the author, source location and revision of the reference `4d_eval`
  library used in GFM4DBench, and verify its declared Apache-2.0 permission.
  Open-d4rt is the linked protocol reference; that link alone does not establish
  the provenance of every file in `eval4d`.

Record the confirmed revisions and modification scope in the relevant NOTICE
files. The VGGT license and its Acceptable Use Policy continue to apply to
VGGT-derived code. Model weights and datasets need their own release terms;
they are not covered by the source-code license notices.
