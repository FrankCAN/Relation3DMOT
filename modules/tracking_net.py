from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .appear_net import AppearanceNet
from .motion_net import MotionNet
from .fusion_net import *  # noqa
from .gcn import affinity_module
from .new_end import *  # noqa
from .point_net import *  # noqa
from .score_net import *  # noqa

def feature_web(features):
    """

    :param objs: BxDxN
    :param dets: BxDxM
    :return: Bx2dxNxM
    """
    obj_mat = features.unsqueeze(-1).repeat(1, 1, 1, features.size(-1))  # BxDxNxM
    det_mat = dets.unsqueeze(-2).repeat(1, 1, objs.size(-1), 1)  # BxDxNxM
    related_pos = obj_mat - det_mat # (obj_mat - det_mat) / 2  # BxDxNxM
    return related_pos


class TrackingNet(nn.Module):

    def __init__(self,
                 seq_len,
                 appear_len=512,
                 appear_skippool=False,
                 appear_fpn=False,
                 score_arch='vgg',
                 score_fusion_arch='C',
                 appear_arch='vgg',
                 point_arch='v1',
                 point_len=512,
                 softmax_mode='single',
                 test_mode=0,
                 affinity_op='multiply',
                 dropblock=5,
                 end_arch='v2',
                 end_mode='avg',
                 without_reflectivity=True,
                 neg_threshold=0,
                 use_dropout=False):
        super(TrackingNet, self).__init__()
        self.seq_len = seq_len
        self.score_arch = score_arch
        self.neg_threshold = neg_threshold
        self.test_mode = test_mode  # 0:image;1:image;2:fusion
        point_in_channels = 4 - int(without_reflectivity)

        if point_len == 0:
            in_channels = appear_len
        else:
            in_channels = point_len

        in_channels = 1024

        self.fusion_module = None
        fusion = eval(f"fusion_module_{score_fusion_arch}")
        self.fusion_module = fusion(
            appear_len, point_len, out_channels=point_len)

        if appear_len == 0:
            print('No image appearance used')
            self.appearance = None
        else:
            self.appearance = AppearanceNet(
                appear_arch,
                appear_len,
                skippool=appear_skippool,
                fpn=appear_fpn,
                dropblock=dropblock)

        # build new end indicator
        if end_arch in ['v1', 'v2']:
            new_end = partial(
                eval("NewEndIndicator_%s" % end_arch),
                kernel_size=5,
                reduction=4,
                mode=end_mode)

        # build point net
        if point_len == 0:
            print("No point cloud used")
            self.point_net = None
        elif point_arch in ['v1']:
            point_net = eval("PointNet_%s" % point_arch)
            self.point_net = point_net(
                point_in_channels,
                out_channels=point_len,
                use_dropout=use_dropout)
        else:
            print("Not implemented!!")

        # build motion net
        self.motion_net = MotionNet(7, out_channels=512)

        # build affinity matrix module
        assert in_channels != 0
        self.w_link = affinity_module(
            in_channels, new_end=new_end, affinity_op=affinity_op)

        # build negative rejection module
        if score_arch in ['branch_cls', 'branch_reg']:
            self.w_det = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, 1, 1),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels, in_channels // 2, 1, 1),
                nn.BatchNorm1d(in_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels // 2, 1, 1, 1),
            )
        else:
            print("Not implement yet")

        self.softmax_mode = softmax_mode

        # self.relation_weights = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, 1, 1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels, in_channels, 1, 1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True)

        # )

        # self.bn_rsconv = nn.BatchNorm2d(in_channels)
        # self.activation = nn.ReLU(inplace=True)



    def associate(self, objs, dets):
        link_mat, new_score, end_score = self.w_link(objs, dets)

        link_score = link_mat

        # if self.softmax_mode == 'single':
        #     link_score = F.softmax(link_mat, dim=-1)
        # elif self.softmax_mode == 'dual':
        #     link_score_prev = F.softmax(link_mat, dim=-1)
        #     link_score_next = F.softmax(link_mat, dim=-2)
        #     link_score = link_score_prev.mul(link_score_next)
        # elif self.softmax_mode == 'dual_add':
        #     link_score_prev = F.softmax(link_mat, dim=-1)
        #     link_score_next = F.softmax(link_mat, dim=-2)
        #     link_score = (link_score_prev + link_score_next) / 2
        # elif self.softmax_mode == 'dual_max':
        #     link_score_prev = F.softmax(link_mat, dim=-1)
        #     link_score_next = F.softmax(link_mat, dim=-2)
        #     link_score = torch.max(link_score_prev, link_score_next)
        # else:
        #     link_score = link_mat

        return link_score, new_score, end_score

    def feature(self, dets, det_info):
        feats = []

        if self.appearance is not None:
            appear = self.appearance(dets)
            feats.append(appear)

        trans = None
        if self.point_net is not None:
            points, trans = self.point_net(
                det_info['points'].transpose(-1, -2),
                det_info['points_split'].long().squeeze(0))
            feats.append(points)

        motions = self.motion_net(torch.cat([det_info['bbox'].transpose(-1, -2), det_info['loc'].transpose(-1, -2)], dim=1))
        motions = torch.cat([motions, motions, motions], dim=0)
        # feats.append(motions)

        feats = torch.cat(feats, dim=-1).t().unsqueeze(0)  # LxD->1xDxL
        if self.fusion_module is not None:
            feats = self.fusion_module(feats)

            feats = torch.cat([feats, motions], dim=1)

            return feats, trans

        return feats, trans

    def determine_det(self, dets, feats):
        det_scores = self.w_det(feats).squeeze(1)  # Bx1xL -> BxL

        if not self.training:
            # add mask
            if 'cls' in self.score_arch:
                det_scores = det_scores.sigmoid()


#             print(det_scores[:, -1].size())
#             mask = det_scores[:, -1].lt(self.neg_threshold)
#             det_scores[:, -1] -= mask.float()
            mask = det_scores.lt(self.neg_threshold)
            det_scores -= mask.float()
        return det_scores

    def forward(self, dets, det_info, dets_split):
        feats, trans = self.feature(dets, det_info)
        det_scores = self.determine_det(dets, feats)

        start = 0
        link_scores = []
        new_scores = []
        end_scores = []

        prev_end = start + dets_split[0].item()
        end = prev_end + dets_split[1].item()

        feats_prev = feats[:, :, start:prev_end]
        feats_end = feats[:, :, prev_end:end]

        # B, C, M = feats_prev.size()
        # B, C, N = feats_end.size()
        #
        # mapping = torch.einsum('bci,bcj->bcij', feats_prev, feats_end)
        # feats_relation_weights = self.relation_weights(mapping)
        #
        # mapping_update = self.activation(self.bn_rsconv(torch.mul(feats_relation_weights, mapping)))
        #
        # feats_prev = F.max_pool2d(mapping_update, kernel_size=(1, N)).squeeze(3)
        # feats_end = F.max_pool2d(mapping_update, kernel_size=(M, 1)).squeeze(2)

        # B, C, N = feats_prev.size()
        # feats_prev = feats_prev.view(B, C, N, 1).repeat(1, 1, 1, N)
        # feats_prev = feats_prev - feats_prev.transpose(2, 3).contiguous() + torch.mul(feats_prev, torch.eye(N).view(1, 1, N, N).cuda())
        # feats_prev_weight = self.w_prev(feats_prev)
        # feats_prev = (feats_prev * feats_prev_weight).sum(-1)
        # # feats_prev = feats_prev.max(dim=-1, keepdim=False)[0]
        #
        # B, C, N = feats_end.size()
        # feats_end = feats_end.view(B, C, N, 1).repeat(1, 1, 1, N)
        # feats_end = feats_end - feats_end.transpose(2, 3).contiguous() + torch.mul(feats_end, torch.eye(N).view(1, 1, N, N).cuda())
        # feats_end_weight = self.w_end(feats_end)
        # feats_end = (feats_end * feats_end_weight).sum(-1)
        # feats_end = feats_end.max(dim=-1, keepdim=False)[0]

        # feats_prev = feats[:, :, start:prev_end]
        # feats_end = feats[:, :, prev_end:end]


        link_score, new_score, end_score = self.associate(feats_prev, feats_end)

        link_scores.append(link_score.squeeze(1))
        new_scores.append(new_score)
        end_scores.append(end_score)


        # for i in range(len(dets_split) - 1):
        #     prev_end = start + dets_split[i].item()
        #     end = prev_end + dets_split[i + 1].item()
        #     link_score, new_score, end_score = self.associate(
        #         feats[:, :, start:prev_end], feats[:, :, prev_end:end])
        #     link_scores.append(link_score.squeeze(1))
        #     new_scores.append(new_score)
        #     end_scores.append(end_score)
        #     start = prev_end

        if not self.training:
            fake_new = det_scores.new_zeros(
                (det_scores.size(0), link_scores[0].size(-2)))
            fake_end = det_scores.new_zeros(
                (det_scores.size(0), link_scores[-1].size(-1)))
            new_scores = torch.cat([fake_new] + new_scores, dim=1)
            end_scores = torch.cat(end_scores + [fake_end], dim=1)
        else:
            new_scores = torch.cat(new_scores, dim=1)
            end_scores = torch.cat(end_scores, dim=1)
        return det_scores, link_scores, new_scores, end_scores, trans
