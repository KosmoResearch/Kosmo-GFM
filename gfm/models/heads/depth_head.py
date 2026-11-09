# Source provenance requires confirmation; see licenses/README.md.
"""Multi-scale depth and confidence head for GFM."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gfm.models.heads.dpt_blocks import Exp, Resnet, _make_fusion_block, _make_scratch


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        patch_size=14,
        features=256,
        features2=32,
        depth_channel=1,
        depth_act="sigmoid",
        interp_refinenet_cfg=None,
        pred_confidence=False,
        return_features=False,
        chunk_size=0,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.patch_size = patch_size
        self.features = features
        self.features_2 = features2
        self.depth_channel = depth_channel

        self.interp_refinenet_cfg = interp_refinenet_cfg

        self.return_features = return_features
        self.chunk_size = chunk_size

        self.build_act_postprocess()
        self.build_dpt_adapter()

        if depth_act == "sigmoid":
            self.depth_act = nn.Sigmoid()
        elif depth_act == "relu":
            self.depth_act = nn.ReLU(True)
        elif depth_act == "exp":
            self.depth_act = Exp()
        else:
            raise ValueError(f"Unsupported depth_act for inference: {depth_act}")

        self.output_conv1 = nn.Conv2d(
            self.features,
            self.features // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if self.interp_refinenet_cfg is not None:
            self.interp_refinenet_cfg["input_channel"] = self.features // 2
            cfg = {k: v for k, v in self.interp_refinenet_cfg.items() if k != "type"}
            self.interp_refinenet = Resnet(**cfg)

        if self.depth_channel > 0:
            self.output_conv2 = nn.Sequential(
                nn.Conv2d(
                    self.features // 2,
                    self.features_2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(True),
                nn.Conv2d(self.features_2, self.depth_channel, kernel_size=1, stride=1, padding=0),
                self.depth_act,
            )

        self.pred_confidence = pred_confidence
        if self.pred_confidence:
            self.confidence_conv = self.build_output_conv(self.features, self.features_2, 1)

    def build_act_postprocess(self):
        if isinstance(self.in_channels, (list, tuple)):
            self.projects = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                    for in_channel, out_channel in zip(self.in_channels, self.mid_channels)
                ]
            )
        else:
            self.projects = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=out_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                    for out_channel in self.mid_channels
                ]
            )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=self.mid_channels[0],
                    out_channels=self.mid_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=self.mid_channels[1],
                    out_channels=self.mid_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=self.mid_channels[3],
                    out_channels=self.mid_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

    def build_dpt_adapter(self):
        # Initialize the scratch layers used for feature refinement.
        self.scratch = _make_scratch(
            self.mid_channels,
            self.features,
            groups=1,
            expand=False,
        )

        # Optional component; not used here.
        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(
            self.features,
            use_bn=False,
            out_conv_cfg=self.interp_refinenet_cfg,
        )
        self.scratch.refinenet2 = _make_fusion_block(
            self.features,
            use_bn=False,
            out_conv_cfg=self.interp_refinenet_cfg,
        )
        self.scratch.refinenet3 = _make_fusion_block(
            self.features,
            use_bn=False,
            out_conv_cfg=self.interp_refinenet_cfg,
        )
        self.scratch.refinenet4 = _make_fusion_block(
            self.features,
            use_bn=False,
            out_conv_cfg=self.interp_refinenet_cfg,
        )

    def build_output_conv(self, ch1, ch2, outch):
        conv = nn.Sequential(
            nn.Conv2d(
                ch1 // 2,
                ch2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(ch2, outch, kernel_size=1, stride=1, padding=0),
        )
        return conv

    def _forward_impl(
        self,
        features,
        patch_h=None,
        patch_w=None,
        patch_start_idx=None,
        view_start_idx=None,
        view_end_idx=None,
    ):
        out = []
        for i, x in enumerate(features):
            if patch_start_idx is not None:
                if x.ndim == 4:
                    x = x[:, :, patch_start_idx:]
                elif x.ndim == 3:
                    x = x[:, patch_start_idx:]

            if view_start_idx is not None and view_end_idx is not None:
                x = x[:, view_start_idx:view_end_idx]

            if x.ndim == 4:
                x = x.reshape(-1, x.shape[-2], x.shape[-1])

            x = x.permute(0, 2, 1).contiguous().reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        # Apply the channel-reduction layers to the RGB features.
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.output_conv1(path_1)
        out = F.interpolate(
            out,
            (int(patch_h * self.patch_size), int(patch_w * self.patch_size)),
            mode="bilinear",
            align_corners=True,
        )

        if self.interp_refinenet_cfg is not None:
            out = self.interp_refinenet(out)

        if self.depth_channel == 1:
            depth = self.output_conv2(out)
            return_dict = dict(depth=depth)
        elif self.depth_channel == 3:
            points = self.output_conv2(out)
            return_dict = dict(points=points)
        elif self.depth_channel == 0:
            return_dict = dict()
        else:
            raise NotImplementedError

        if self.return_features:
            return_dict["features"] = out

        if self.pred_confidence:
            confidence = self.confidence_conv(out)
            return_dict["confidence"] = confidence

        return return_dict

    def forward(
        self,
        features,
        patch_h,
        patch_w,
        patch_start_idx=None,
        **kwargs,
    ):
        if self.chunk_size == 0:
            return self._forward_impl(
                features=features,
                patch_h=patch_h,
                patch_w=patch_w,
                patch_start_idx=patch_start_idx,
                view_start_idx=None,
                view_end_idx=None,
            )

        assert features[0].ndim == 4
        B, S = features[0].shape[:2]

        total_datas = {}
        for view_start_idx in range(0, S, self.chunk_size):
            view_end_idx = min(view_start_idx + self.chunk_size, S)

            outputs = self._forward_impl(
                features=features,
                patch_h=patch_h,
                patch_w=patch_w,
                patch_start_idx=patch_start_idx,
                view_start_idx=view_start_idx,
                view_end_idx=view_end_idx,
            )

            # [view_end_idx, view_start_idx)
            cur_S = view_end_idx - view_start_idx

            for key, val in outputs.items():
                if key not in total_datas:
                    total_datas[key] = []
                total_datas[key].append(val.view(B, cur_S, *val.shape[-3:]))

        for key, val in total_datas.items():
            total_datas[key] = torch.cat(val, dim=1)

        return total_datas
