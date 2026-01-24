import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.energy_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, decoder_outputs, encoder_outputs):
        trg_seq_len = decoder_outputs.size(1)
        src_seq_len = encoder_outputs.size(1)
        
        query = self.query_layer(decoder_outputs)  
        key = self.key_layer(encoder_outputs)     
        
        query = query.unsqueeze(2)  
        key = key.unsqueeze(1)      
        
        energy = torch.tanh(query + key)  
        energy = self.energy_layer(energy).squeeze(-1)  
        
        attention_weights = F.softmax(energy, dim=-1)  
        
        context_vector = torch.bmm(attention_weights, encoder_outputs)  
        
        return context_vector, attention_weights

class SimpleSpatialTemporalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.q_w = nn.Linear(128, 128)
        
        self.feature_proj = nn.Conv2d(128, 128, 1)
        
        self.fusion = nn.Linear(256, 128)  

    def forward(self, spatial_feat, seq_feat):
        projected = self.feature_proj(spatial_feat)  
        
        B, C, H, W = projected.shape
        spatial_flat = projected.view(B, C, H*W)     
        
        attn_scores = torch.matmul(self.q_w(seq_feat), spatial_flat) 
        attn_weights = F.softmax(attn_scores, dim=-1)           
        
        aggregated = torch.matmul(attn_weights, spatial_flat.permute(0,2,1)) 
        
        combined = torch.cat([aggregated, seq_feat], dim=-1)  

        fused = self.fusion(combined)  
        
        return fused

class EnhancedSpatioTemporalAttention(nn.Module):
    def __init__(self, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, groups=128),  
            nn.GELU(),
            nn.Conv2d(128, 128, 1),  
            nn.GroupNorm(8, 128)
        )
        
        self.q_proj = nn.Linear(128, 128)
        self.kv_proj = nn.Conv2d(128, 256, 1)  
        self.multihead_attn = nn.MultiheadAttention(embed_dim=128, num_heads=num_heads, batch_first=True)
        
        self.temporal_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.GLU(dim=-1),  
            nn.LayerNorm(128)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )

    def forward(self, spatial_feat, seq_feat):
        B, C, H, W = spatial_feat.shape
        
        spatial = self.spatial_proj(spatial_feat)  
        kv = self.kv_proj(spatial)  
        k, v = torch.chunk(kv, 2, dim=1) 
        
        k = k.view(B, 128, H*W).permute(0, 2, 1)  
        v = v.view(B, 128, H*W).permute(0, 2, 1)  
        
        q = self.temporal_proj(seq_feat)  
        
        attn_output, _ = self.multihead_attn(
            query=q,    
            key=k,      
            value=v,    
            need_weights=False
        )  
        
        combined = torch.cat([attn_output, seq_feat], dim=-1)  
        output = self.fusion(combined)  
        
        return output

class EfficientSpatioTemporalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.spatial_compress = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  
            nn.Flatten(start_dim=1)        
        )
        
        self.interaction = nn.Sequential(
            nn.Linear(128 + 128, 256),    
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.Sigmoid()                   
        )
        
        self.temporal_transform = nn.Linear(128, 128)
        
        self.final_fusion = nn.Linear(128 * 2, 128)

    def forward(self, spatial_feat, seq_feat):
        spatial_vec = self.spatial_compress(spatial_feat)  
        
        temporal_feat = self.temporal_transform(seq_feat)  
        
        expanded_spatial = spatial_vec.unsqueeze(1).expand(-1, 12, -1)  
        combined = torch.cat([expanded_spatial, temporal_feat], dim=-1) 
        gate = self.interaction(combined)  
        
        weighted_temporal = temporal_feat * gate  
        
        output = self.final_fusion(
            torch.cat([weighted_temporal, seq_feat], dim=-1)
        )  
        
        return output


class LSTMTimeSeries(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, use_satellite_memory):
        super(LSTMTimeSeries, self).__init__()
        
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)

        self.attention_layer = AdditiveAttention(hidden_dim)

        self.use_satellite_memory = use_satellite_memory
        if self.use_satellite_memory:
            self.final_attention = SimpleSpatialTemporalAttention()

        self.output_fc = nn.Linear(2*hidden_dim, hidden_dim)
        
        self.output_fc_1 = nn.Linear(hidden_dim, output_dim)

        self.simplefusion = nn.AdaptiveAvgPool2d((1, 1)) 
    
    def forward(self, x, conditional_vectors):
        batch_size, seq_len, _ = x.size()
        
        x = self.relu(self.input_fc(x))
        
        encoder_outputs, (hidden, cell) = self.encoder_lstm(x)  
        
        decoder_input = torch.zeros(batch_size, seq_len, encoder_outputs.size(-1)).to(x.device)  
        
        decoder_outputs, _ = self.decoder_lstm(decoder_input, (hidden, cell))  

        context_features, scores = self.attention_layer(encoder_outputs, decoder_outputs)

        outputs = self.output_fc(torch.cat((context_features, decoder_outputs),2))  
    
        if not self.use_satellite_memory:
            outputs = self.output_fc_1(outputs)
            return outputs

        outputs = self.final_attention(conditional_vectors, outputs)
        outputs = self.output_fc_1(outputs)
        return outputs