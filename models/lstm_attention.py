"""Attention LSTM used by the released checkpoints."""

import torch
from torch import nn
from torch.nn import functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.energy_layer = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_outputs, encoder_outputs):
        """
        Args:
            decoder_outputs: [batch_size, trg_seq_len, hidden_dim]
            encoder_outputs: [batch_size, src_seq_len, hidden_dim]

        Returns:
            context_vector: [batch_size, trg_seq_len, hidden_dim]
            attention_weights: [batch_size, trg_seq_len, src_seq_len]
        """
        query = self.query_layer(decoder_outputs)
        key = self.key_layer(encoder_outputs)
        query = query.unsqueeze(2)
        key = key.unsqueeze(1)
        energy = torch.tanh(query + key)
        energy = self.energy_layer(energy).squeeze(-1)
        attention_weights = F.softmax(energy, dim=-1)
        context_vector = torch.bmm(attention_weights, encoder_outputs)
        return (context_vector, attention_weights)


class SimpleSpatialTemporalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_w = nn.Linear(128, 128)
        self.feature_proj = nn.Conv2d(128, 128, 1)
        self.fusion = nn.Linear(256, 128)

    def forward(self, spatial_feat, seq_feat):
        """
        输入:
        spatial_feat: [1, 128, 32, 32]
        seq_feat:     [1, 12, 128]
        """
        projected = self.feature_proj(spatial_feat)
        B, C, H, W = projected.shape
        spatial_flat = projected.view(B, C, H * W)
        attn_scores = torch.matmul(self.q_w(seq_feat), spatial_flat)
        attn_weights = F.softmax(attn_scores, dim=-1)
        aggregated = torch.matmul(attn_weights, spatial_flat.permute(0, 2, 1))
        combined = torch.cat([aggregated, seq_feat], dim=-1)
        fused = self.fusion(combined)
        return fused


class LSTMTimeSeries(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, use_satellite_memory
    ):
        super(LSTMTimeSeries, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.encoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, batch_first=True
        )
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, batch_first=True
        )
        self.attention_layer = AdditiveAttention(hidden_dim)
        self.use_satellite_memory = use_satellite_memory
        if self.use_satellite_memory:
            self.final_attention = SimpleSpatialTemporalAttention()
        self.output_fc = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output_fc_1 = nn.Linear(hidden_dim, output_dim)
        self.simplefusion = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, conditional_vectors):
        batch_size, seq_len, _ = x.size()
        x = self.relu(self.input_fc(x))
        encoder_outputs, (hidden, cell) = self.encoder_lstm(x)
        decoder_input = torch.zeros(batch_size, seq_len, encoder_outputs.size(-1)).to(
            x.device
        )
        decoder_outputs, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        context_features, scores = self.attention_layer(
            encoder_outputs, decoder_outputs
        )
        outputs = self.output_fc(torch.cat((context_features, decoder_outputs), 2))
        if not self.use_satellite_memory:
            outputs = self.output_fc_1(outputs)
            return outputs
        outputs = self.final_attention(conditional_vectors, outputs)
        outputs = self.output_fc_1(outputs)
        return outputs
