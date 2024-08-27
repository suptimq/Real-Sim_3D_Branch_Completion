import torch
import torch.nn as nn
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2, DensityAwareChamferDistance
from losses.losses import RepulsionLoss, CPCLoss


@MODELS.register_module()
class PCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.number_fine = config.model.num_pred
        self.encoder_channel = config.model.encoder_channel
        self.grid_size = config.model.grid_size

        assert self.number_fine % self.grid_size**2 == 0
        self.number_coarse = self.number_fine // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,256,1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,self.encoder_channel,1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_channel,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,3*self.number_coarse)
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(self.encoder_channel+3+2,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,3,1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda() # 1 2 S
        
        self.repulsion_loss_func = None
        self.cpc_loss_func = None
        self.build_loss_func(config.loss)

    def build_loss_func(self, loss):
        if loss.consider_metric == 'CDL1':
            self.loss_func = ChamferDistanceL1()
        elif loss.consider_metric == 'CDL2':
            self.loss_func = ChamferDistanceL2()
        elif loss.consider_metric == 'DCD':
            self.loss_func = DensityAwareChamferDistance(**loss.kwargs)
        else:
            raise NotImplementedError

        if 'repulsion' in loss.additional_metrics.keys():
            self.repulsion_loss_func = RepulsionLoss(alpha=loss.additional_metrics['repulsion'])

        if 'cpc' in loss.additional_metrics.keys():
            self.cpc_loss_func = CPCLoss(alpha=loss.additional_metrics['cpc'])

    def get_loss(self, ret, gt, **kwargs):
        
        assert (~torch.isnan(ret[0])).all(), "Found NaN in Coarse Point"
        assert (~torch.isnan(ret[1])).all(), "Found NaN in FinePoint" 
        
        loss_dict = {}

        loss_dict['coarse_loss'] = self.loss_func(ret[0], gt)
        loss_dict['fine_loss'] = self.loss_func(ret[1], gt)

        if self.repulsion_loss_func is not None:
            loss_dict['repulsion_loss'] = self.repulsion_loss_func(ret[1])

        if self.cpc_loss_func is not None:
            centers = kwargs.get('centers', None)
            center_radii = kwargs.get('center_radii', None)
            assert centers is not None and center_radii is not None, 'Missing Segment GT Data'
            p2p_var1, p2s_mean1 = self.cpc_loss_func(ret[0], centers, center_radii)
            p2p_var2, p2s_mean2 = self.cpc_loss_func(ret[1], centers, center_radii)

            loss_dict['cpc_loss_p2p_var'] = p2p_var1 + p2p_var2
            loss_dict['cpc_loss_p2s_mean'] = p2s_mean1 + p2s_mean2

        return loss_dict

    def forward(self, xyz):
        bs , n , _ = xyz.shape
        # encoder
        feature = self.first_conv(xyz.transpose(2,1))  # B 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# B 512 n
        feature = self.second_conv(feature) # B 1024 n
        feature_global = torch.max(feature,dim=2,keepdim=False)[0] # B 1024
        # decoder
        coarse = self.mlp(feature_global).reshape(-1,self.number_coarse,3) # B M 3
        point_feat = coarse.unsqueeze(2).expand(-1,-1,self.grid_size**2,-1) # B M S 3
        point_feat = point_feat.reshape(-1,self.number_fine,3).transpose(2,1) # B 3 N

        seed = self.folding_seed.unsqueeze(2).expand(bs,-1,self.number_coarse, -1) # B 2 M S
        seed = seed.reshape(bs,-1,self.number_fine)  # B 2 N

        feature_global = feature_global.unsqueeze(2).expand(-1,-1,self.number_fine) # B 1024 N
        feat = torch.cat([feature_global, seed, point_feat], dim=1) # B C N
    
        fine = self.final_conv(feat) + point_feat   # B 3 N

        return (coarse.contiguous(), fine.transpose(1,2).contiguous())