"""Checkpoint-compatible LSTM-att+Sat model, independent of training scripts."""

import torch
from torch import nn
from torch.nn import functional as F

from .lstm_attention import LSTMTimeSeries


class Image_extractor(nn.Module):
    def __init__(self, input_dim, image_feature_dim):
        super(Image_extractor, self).__init__()
        self.image_feature_extractor = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, image_feature_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc_out = nn.Linear(image_feature_dim, 4)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.image_feature_extractor(x)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        x = x.reshape(b, t, -1)
        x = self.fc_out(x)
        return x


class MotionEncoder(nn.Module):
    def __init__(self, length):
        super(MotionEncoder, self).__init__()
        self.length = length
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ELU(),
        )
        self.spatial_down = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.temporal_net = nn.Sequential(
            TemporalConv3d(512, 256, stride=2), TemporalConv3d(256, 256, stride=2)
        )
        self.fusion = nn.Conv3d(256, 512, (1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.spatial_encoder(x)
        x = self.spatial_down(x)
        _, C, H, W = x.shape
        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        x = self.temporal_net(x)
        x = self.fusion(x)
        x = F.adaptive_avg_pool3d(x, (1, None, None))
        return x.squeeze(2)


class TemporalConv3d(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_dim, out_dim, (3, 1, 1), stride=(stride, 1, 1), padding=(1, 0, 0)
            ),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(),
        )
        self.residual = (
            nn.Conv3d(in_dim, out_dim, (1, 1, 1), stride=(stride, 1, 1))
            if in_dim != out_dim
            else None
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if self.residual:
            residual = self.residual(residual)
        return x + residual if self.residual else x


class Satellite_Memory(nn.Module):
    def __init__(self, memory_size, seq_len, out_len):
        super(Satellite_Memory, self).__init__()
        self.motion_matching_encoder = MotionEncoder(seq_len)
        self.motion_context_encoder = MotionEncoder(seq_len + out_len)
        self.embedder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(),
        )
        self.memory_shape = [memory_size, 512]
        self.memory_w = nn.init.normal_(
            torch.empty(self.memory_shape), mean=0.0, std=1.0
        )
        self.memory_w = nn.Parameter(self.memory_w, requires_grad=True)

    def forward(self, memory_x, phase):
        memory_x = memory_x.transpose(dim0=1, dim1=2)
        motion_encoder = (
            self.motion_context_encoder if phase == 1 else self.motion_matching_encoder
        )
        memory_query = torch.squeeze(motion_encoder(memory_x), dim=2)
        query_c, query_h, query_w = (
            memory_query.size()[1],
            memory_query.size()[2],
            memory_query.size()[3],
        )
        memory_query = memory_query.permute(0, 2, 3, 1)
        memory_query = torch.reshape(memory_query, (-1, query_c))
        query_norm = F.normalize(memory_query, dim=1)
        memory_norm = F.normalize(self.memory_w, dim=1)
        s = torch.mm(query_norm, memory_norm.transpose(dim0=0, dim1=1))
        addressing_vec = F.softmax(s, dim=1)
        memory_feature = torch.mm(addressing_vec, self.memory_w)
        memory_feature = torch.reshape(memory_feature, (-1, query_h, query_w, query_c))
        memory_feature = memory_feature.permute(0, 3, 1, 2)
        memory_feature = self.embedder(memory_feature)
        return memory_feature


class Solar_Predictor(nn.Module):
    def __init__(
        self,
        img_channel,
        img_size,
        vector_hidden_size,
        vector_output_size,
        input_dim,
        multi_input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        seq_len,
        out_len,
        if_multi_variable,
        use_satellite_memory,
    ):
        super(Solar_Predictor, self).__init__()
        self.if_multi_variable = if_multi_variable
        self.use_satellite_memory = use_satellite_memory
        if use_satellite_memory:
            self.image_extractor = Image_extractor(multi_input_dim, hidden_dim)
            self.satellite_memory = Satellite_Memory(128, seq_len, out_len)
            if self.if_multi_variable:
                self.point_predictor = LSTMTimeSeries(
                    input_dim + multi_input_dim + 4,
                    hidden_dim,
                    output_dim + multi_input_dim + 4,
                    num_layers,
                    use_satellite_memory,
                )
            else:
                self.point_predictor = LSTMTimeSeries(
                    input_dim + 4,
                    hidden_dim,
                    output_dim + 4,
                    num_layers,
                    use_satellite_memory,
                )
        elif self.if_multi_variable:
            self.point_predictor = LSTMTimeSeries(
                input_dim + multi_input_dim,
                hidden_dim,
                output_dim + multi_input_dim,
                num_layers,
                use_satellite_memory,
            )
        else:
            self.point_predictor = LSTMTimeSeries(
                input_dim, hidden_dim, output_dim, num_layers, use_satellite_memory
            )
        self.out_len = out_len
        self.output_dim = output_dim

    def forward(self, x, y=None, multi_input_seq=None, phase=2):
        image_seq = x[:, :, 0:4, :, :]
        point_seq = x[:, :, 4:5, 0, 0]
        short_seq = x[:, :, 0:4, :, :]
        if not self.use_satellite_memory:
            memorized_satellite_patterns = None
        elif phase == 1:
            if y is None:
                raise ValueError(
                    "Phase 1 requires future satellite frames; use phase=2 for inference."
                )
            long_seq = torch.cat((short_seq, y[:, :, 0:4, :, :]), 1)
            memorized_satellite_patterns = self.satellite_memory(long_seq, phase=1)
        else:
            memorized_satellite_patterns = self.satellite_memory(short_seq, phase=2)
        if self.use_satellite_memory:
            global_points = self.image_extractor(image_seq)
            if self.if_multi_variable:
                point_pred_seqs = self.point_predictor(
                    torch.cat((point_seq, multi_input_seq, global_points), 2),
                    memorized_satellite_patterns,
                )
            else:
                point_pred_seqs = self.point_predictor(
                    torch.cat((point_seq, global_points), 2),
                    memorized_satellite_patterns,
                )
        elif self.if_multi_variable:
            point_pred_seqs = self.point_predictor(
                torch.cat((point_seq, multi_input_seq), 2), memorized_satellite_patterns
            )
        else:
            point_pred_seqs = self.point_predictor(
                point_seq, memorized_satellite_patterns
            )
        return point_pred_seqs[:, :, 0 : self.output_dim]
