import torch
import torch.nn as nn
import torch.nn.functional as F

from lidarnerf.encoding import get_encoder
from lidarnerf.activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(
        self,
        encoding="hashgrid",
        encoding_dir="frequency",
        multires=15,
        encoding_bg="hashgrid",
        desired_resolution=2048,
        log2_hashmap_size=19,
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        num_layers_bg=2,
        hidden_dim_bg=64,
        out_color_dim=3,
        out_lidar_color_dim=2,
        bound=1,
        **kwargs,
    ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.out_color_dim = out_color_dim
        self.out_lidar_color_dim = out_lidar_color_dim
        self.encoder, self.in_dim = get_encoder(
            encoding,
            desired_resolution=desired_resolution,
            log2_hashmap_size=log2_hashmap_size,
        )

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder("sphere_harmonics")

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = self.out_color_dim  # 3 rgb
            else:
                out_dim = hidden_dim_color

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # lidar color network
        self.encoder_lidar_dir, self.in_dim_dir = get_encoder("frequency", multires=12)

        lidar_color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = self.out_lidar_color_dim  # 2 rgb
            else:
                out_dim = hidden_dim_color

            lidar_color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.lidar_color_net = nn.ModuleList(lidar_color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(
                encoding_bg,
                input_dim=2,
                num_levels=4,
                log2_hashmap_size=19,
                desired_resolution=2048,
            )  # much smaller hashgrid

            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg

                if l == num_layers_bg - 1:
                    out_dim = 3  # 3 rgb
                else:
                    out_dim = hidden_dim_bg

                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        x = x.to(next(self.encoder.parameters()).device)
         # Ensure x is on the same device as the encoder
        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            "sigma": sigma,
            "geo_feat": geo_feat,
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x)  # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    # def color(self, x, d, cal_lidar_color=False, mask=None, geo_feat=None, **kwargs):
    #     # x: [N, 3] in [-bound, bound]
    #     # mask: [N,], bool, indicates where we actually needs to compute rgb.

    #     if mask is not None:
    #         rgbs = torch.zeros(
    #             mask.shape[0], self.out_dim, dtype=x.dtype, device=x.device
    #         )  # [N, 3]
    #         # in case of empty mask
    #         if not mask.any():
    #             return rgbs
    #         x = x[mask]
    #         d = d[mask]
    #         geo_feat = geo_feat[mask]

    #     if cal_lidar_color:
    #         d = self.encoder_lidar_dir(d)
    #         h = torch.cat([d, geo_feat], dim=-1)
    #         for l in range(self.num_layers_color):
    #             h = self.lidar_color_net[l](h)
    #             if l != self.num_layers_color - 1:
    #                 h = F.relu(h, inplace=True)
    #     else:
    #         d = self.encoder_dir(d)
    #         h = torch.cat([d, geo_feat], dim=-1)
    #         for l in range(self.num_layers_color):
    #             h = self.color_net[l](h)
    #             if l != self.num_layers_color - 1:
    #                 h = F.relu(h, inplace=True)

    #     # sigmoid activation for rgb
    #     h = torch.sigmoid(h)

    #     if mask is not None:
    #         rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
    #     else:
    #         rgbs = h

    #     return rgbs
    def color(self, x, d, cal_lidar_color=False, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(
                mask.shape[0], self.out_dim, dtype=x.dtype, device=x.device
            )  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask].to(x.device)  # optional, should already match
            d = d[mask].to(x.device)
            geo_feat = geo_feat[mask].to(x.device)

        if cal_lidar_color:
            d = self.encoder_lidar_dir(d)
            geo_feat = geo_feat.to(d.device)
            h = torch.cat([d, geo_feat], dim=-1)
            for l in range(self.num_layers_color):
                h = self.lidar_color_net[l](h)
                if l != self.num_layers_color - 1:
                    h = F.relu(h, inplace=True)
        else:
            d = self.encoder_dir(d)
            geo_feat = geo_feat.to(d.device)
            h = torch.cat([d, geo_feat], dim=-1)
            for l in range(self.num_layers_color):
                h = self.color_net[l](h)
                if l != self.num_layers_color - 1:
                    h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            mask = mask.to(rgbs.device)
            rgbs[mask] = h.to(rgbs.dtype).to(rgbs.device)
        else:
            rgbs = h

        return rgbs

    # optimizer utils
    def get_params(self, lr):
        params = [
            {"params": self.encoder.parameters(), "lr": lr},
            {"params": self.sigma_net.parameters(), "lr": lr},
            {"params": self.encoder_dir.parameters(), "lr": lr},
            {"params": self.color_net.parameters(), "lr": lr},
            {"params": self.encoder_lidar_dir.parameters(), "lr": lr},
            {"params": self.lidar_color_net.parameters(), "lr": lr},
        ]
        if self.bg_radius > 0:
            params.append({"params": self.encoder_bg.parameters(), "lr": lr})
            params.append({"params": self.bg_net.parameters(), "lr": lr})

        return params
