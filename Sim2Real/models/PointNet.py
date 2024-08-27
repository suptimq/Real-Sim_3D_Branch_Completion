from .build import MODELS

import torch.nn as nn

from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG


@MODELS.register_module()
class PointNetSSG(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[3, 64, 64, 128],
                use_xyz=self.config.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=self.config.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=self.config.use_xyz
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

        self.convert_relu_to_leakyrelu(self)

    def convert_relu_to_leakyrelu(self, model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, child_name, nn.LeakyReLU(0.2))
            else:
                self.convert_relu_to_leakyrelu(child)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else pc.transpose(1, 2).contiguous()

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))
    

@MODELS.register_module()
class PointNetMSG(PointNetSSG):
    def _build_model(self):
        super()._build_model()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[3, 32, 32, 64], [3, 64, 64, 128], [3, 64, 96, 128]],
                use_xyz=self.config.use_xyz,
            )
        )

        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=self.config.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, 1024],
                use_xyz=self.config.use_xyz,
            )
        )