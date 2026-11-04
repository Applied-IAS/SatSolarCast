import torch
import torch.optim as optim
from dataloader import Solar_Insitu, Solar_Sat_Insitu, Solar_Sat_Insitu_Multi_Input
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
import os
import torch.nn.functional as F
import argparse
import copy
from lstm import LSTMTimeSeries

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='multi_input_combination_models_new_fusion_encoder_temporal_attention_dynamic_loss',
                    help='the path for saving models')
parser.add_argument('--results_dir', type=str, default='multi_input_combination_results_new_fusion_encoder_temporal_attention_dynamic_loss',
                    help='the path for output results')
parser.add_argument('--input_len', type=int, default=12,
                    help='the length of input sequence')
parser.add_argument('--output_len', type=int, default=12,
                    help='the length of output sequence')
parser.add_argument('--training_data_dir', type=str, default='solarcube_hybrid_sat_insitu_train.h5',
                    help='the path for training dataset')
parser.add_argument('--training_data_points_dir', type=str, default='solarcube_hybrid_sat_insitu_train_points.h5',
                    help='the path for points in training dataset')
parser.add_argument('--test_data_dir', type=str, default='solarcube_hybrid_sat_insitu_test.h5',
                    help='the path for test dataset')
parser.add_argument('--test_data_points_dir', type=str, default='solarcube_hybrid_sat_insitu_test_points.h5',
                    help='the path for poiints in test dataset')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help=' learning rate')     
parser.add_argument('--multi_channel_input', type=str, default='False',
                    help='use the satellite variables or not.')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='id for gpu')
parser.add_argument('--use_satellite_memory', type=int, default=1,
                    help='use the satellite memory or not.')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

class Image_extractor(nn.Module):
    def __init__(self, input_dim, image_feature_dim):
        super(Image_extractor, self).__init__()
        self.image_feature_extractor = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, image_feature_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc_out = nn.Linear(image_feature_dim, 4)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b*t, c, h, w)
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
            nn.ELU())

        self.spatial_down = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.temporal_net = nn.Sequential(
            TemporalConv3d(512, 256, stride=2),  
            TemporalConv3d(256, 256, stride=2)          
        )

        self.fusion = nn.Conv3d(256, 512, (1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        B, T, C, H, W = x.shape
        
        x = x.reshape(B*T, C, H, W)
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
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), 
                     stride=(stride, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(out_dim),
            nn.ReLU()
        )
        self.residual = nn.Conv3d(in_dim, out_dim, (1, 1, 1), 
                                stride=(stride, 1, 1)) if in_dim != out_dim else None
        
    def forward(self, x):
        residual = x
        x = self.conv(x)
        if self.residual:
            residual = self.residual(residual)
        return x + residual if self.residual else x

class SpatioTemporalAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv3d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_att = nn.Sequential(
            nn.Conv3d(channels, 1, (1, 3, 3), padding=(0, 1, 1)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        channel_att = self.channel_att(x)
        spatial_att = self.spatial_att(x)
        return x * channel_att * spatial_att

class Satellite_Memory(nn.Module):
    def __init__(self, memory_size, seq_len, out_len):
        super(Satellite_Memory, self).__init__()
        self.motion_matching_encoder = MotionEncoder(seq_len)
        self.motion_context_encoder = MotionEncoder(seq_len + out_len)

        self.embedder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.ReLU())

        self.memory_shape = [memory_size, 512]
        self.memory_w = nn.init.normal_(torch.empty(self.memory_shape), mean=0.0, std=1.0)
        self.memory_w = nn.Parameter(self.memory_w, requires_grad=True)

    def forward(self, memory_x, phase):
        memory_x = memory_x.transpose(dim0=1, dim1=2)
        motion_encoder = self.motion_context_encoder if phase == 1 else self.motion_matching_encoder
        memory_query = torch.squeeze(motion_encoder(memory_x), dim=2) 
        query_c, query_h, query_w = memory_query.size()[1], memory_query.size()[2], memory_query.size()[3]
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
    def __init__(self, img_channel, img_size, vector_hidden_size, vector_output_size, input_dim, multi_input_dim, hidden_dim, output_dim, num_layers, seq_len, out_len, if_multi_variable, use_satellite_memory):
        super(Solar_Predictor, self).__init__()
        self.if_multi_variable = if_multi_variable
        self.use_satellite_memory = use_satellite_memory
        if use_satellite_memory:
            self.image_extractor = Image_extractor(multi_input_dim, hidden_dim)
            self.satellite_memory = Satellite_Memory(128, seq_len, out_len)
            if self.if_multi_variable:
                self.point_predictor = LSTMTimeSeries(input_dim+ multi_input_dim+4, hidden_dim, output_dim+multi_input_dim+4, num_layers, use_satellite_memory)
            else:
                self.point_predictor = LSTMTimeSeries(input_dim+ 4, hidden_dim, output_dim+4, num_layers, use_satellite_memory)
        else:
            if self.if_multi_variable:
                self.point_predictor = LSTMTimeSeries(input_dim+ multi_input_dim, hidden_dim, output_dim+multi_input_dim, num_layers, use_satellite_memory)
            else:
                self.point_predictor = LSTMTimeSeries(input_dim, hidden_dim, output_dim, num_layers, use_satellite_memory)
        self.out_len = out_len
        self.output_dim = output_dim

    def forward(self, x, y, multi_input_seq, phase=1): 
        image_seq = x[:,:,0:4,:,:]
        point_seq = x[:,:,4:5,0,0]
        
        long_seq = torch.cat((x[:,:,0:4,:,:], y[:,:,0:4,:,:]), 1)
        short_seq = x[:,:,0:4,:,:]
        if not self.use_satellite_memory:
            memorized_satellite_patterns = None
        elif phase == 1:
            memorized_satellite_patterns = self.satellite_memory(long_seq, phase=1)
        else:
            memorized_satellite_patterns = self.satellite_memory(short_seq, phase=2)
        
        if self.use_satellite_memory:
            global_points = self.image_extractor(image_seq)
            if self.if_multi_variable:
                point_pred_seqs = self.point_predictor(torch.cat((point_seq, multi_input_seq, global_points), 2), memorized_satellite_patterns)
            else:
                point_pred_seqs = self.point_predictor(torch.cat((point_seq, global_points), 2), memorized_satellite_patterns)
        else:
            if self.if_multi_variable:
                point_pred_seqs = self.point_predictor(torch.cat((point_seq, multi_input_seq), 2), memorized_satellite_patterns)
            else:
                point_pred_seqs = self.point_predictor(point_seq, memorized_satellite_patterns)
        return point_pred_seqs[:,:,0:self.output_dim] 

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(args, model, train_loader, val_loader, criterion1, criterion2, optimizer, epochs, device):
    directory = args.model_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"directory '{directory}' have been constructed")
    else:
        print(f"directory '{directory}' already exist")
    early_stopping = EarlyStopping(patience=10, path=os.path.join(directory, 'best_model.pth'))
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch, multi_input_seq, points in train_loader:
            x_batch, y_batch, multi_input_seq = x_batch.to(device), y_batch.to(device), multi_input_seq.to(device)
            x_batch = x_batch.permute(0, 1, 4, 2, 3)
            y_batch = y_batch.permute(0, 1, 4, 2, 3)

            image_seq_label = y_batch[:,:,0:4,:,:]
            point_seq_label = y_batch[:,:,4:5,0,0]

            if args.use_satellite_memory:
                # Enable bank gradients before constructing the context-phase graph.
                model.satellite_memory.memory_w.requires_grad = True
                point_seq_pred = model(x_batch, y_batch, multi_input_seq, phase=1)
                loss = criterion2(point_seq_pred, point_seq_label)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                model.satellite_memory.memory_w.requires_grad = False
                point_seq_pred = model(x_batch, y_batch, multi_input_seq, phase=2)
                loss = criterion2(point_seq_pred, point_seq_label)
                # None gradients also prevent Adam momentum updates to the frozen bank.
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            else:
                point_seq_pred = model(x_batch, y_batch, multi_input_seq, phase=2)  
                loss = criterion2(point_seq_pred, point_seq_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, multi_input_seq, points in val_loader:
                x_batch, y_batch, multi_input_seq = x_batch.to(device), y_batch.to(device), multi_input_seq.to(device)

                x_batch = x_batch.permute(0, 1, 4, 2, 3)
                y_batch = y_batch.permute(0, 1, 4, 2, 3)

                image_seq_label = y_batch[:,:,0:4,:,:]
                point_seq_label = y_batch[:,:,4:5,0,0]
                
                point_seq_pred = model(x_batch, y_batch, multi_input_seq, phase=2)
                loss = criterion2(point_seq_pred, point_seq_label)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
    
    model.load_state_dict(torch.load(os.path.join(directory, 'best_model.pth')))
    return model

def test_model(args, model, val_dataloader):
    res = []
    label = []
    with torch.no_grad():
        for x_batch, y_batch, multi_input_seq, points in val_dataloader:
            x_batch, y_batch, multi_input_seq = x_batch.to(device), y_batch.to(device), multi_input_seq.to(device)

            x_batch = x_batch.permute(0, 1, 4, 2, 3)
            y_batch = y_batch.permute(0, 1, 4, 2, 3)

            image_seq_label = y_batch[:,:,0:4,:,:]
            point_seq_label = y_batch[:,:,4:5,0,0]
            
            point_seq_pred = model(x_batch, y_batch, multi_input_seq, phase=2)
            res.append(point_seq_pred.cpu())
            label.append(point_seq_label.cpu())
    res = torch.cat(res, dim=0)
    label = torch.cat(label, dim=0)
    directory = args.results_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"directory '{directory}' have been constructed")
    else:
        print(f"directory '{directory}' already exist")
    np.save(directory + '/' + 'new_combination_pred', res.cpu().numpy())
    np.save(directory + '/' + 'new_combination_gt', label.cpu().numpy())

def test_model_fine_grained(args, model):
    stations = [5, 7, 9, 18]
    for station in stations:
        index1 = args.test_data_dir.rfind('.h5')
        index2 = args.test_data_points_dir.rfind('points.h5')
        test_dataset = Solar_Sat_Insitu_Multi_Input(args.test_data_dir[:index1] + '_' + str(station) + '.h5', args.test_data_points_dir[:index2] + str(station) + '_points' + '.h5')
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1, drop_last=False)
        res = []
        label = []
        with torch.no_grad():
            for x_batch, y_batch, multi_input_seq, points in test_loader:
                x_batch, y_batch, multi_input_seq = x_batch.to(device), y_batch.to(device), multi_input_seq.to(device)
                x_batch = x_batch.permute(0, 1, 4, 2, 3)
                y_batch = y_batch.permute(0, 1, 4, 2, 3)

                image_seq_label = y_batch[:,:,0:4,:,:]
                point_seq_label = y_batch[:,:,4:5,0,0]

                point_seq_pred = model(x_batch, y_batch, multi_input_seq, phase=2)
                res.append(point_seq_pred.cpu())
                label.append(point_seq_label.cpu())
        res = torch.cat(res, dim=0)
        label = torch.cat(label, dim=0)
        directory = args.results_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"directory '{directory}' have been constructed")
        else:
            print(f"directory '{directory}' already exist")
        np.save(directory + '/' + 'new_combination_pred_' + str(station), res.cpu().numpy())
        np.save(directory + '/' + 'new_combination_gt_' + str(station), label.cpu().numpy())

if __name__ == "__main__":
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    image_size = 120
    img_channel = 4 
    img_size = 120
    vector_hidden_size = 4
    vector_output_size = 4
    input_dim = 1     
    multi_input_dim = 4
    hidden_dim = 128  
    output_dim = 1  
    num_layers = 4   
    seq_len = args.input_len     
    target_len = args.output_len   
    batch_size = args.batch_size  
    epochs = 50
    if_multi_variable = (args.multi_channel_input == 'True')
    use_satellite_memory = args.use_satellite_memory
    print(if_multi_variable)

    dataset = Solar_Sat_Insitu_Multi_Input(args.training_data_dir, args.training_data_points_dir)
    print('the length of training dataset', len(dataset))
    train_size = int(0.9 * len(dataset))  
    val_size = len(dataset) - train_size  
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    test_dataset = Solar_Sat_Insitu_Multi_Input(args.test_data_dir, args.test_data_points_dir)
    print('the length of test dataset', len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Solar_Predictor(img_channel, img_size, vector_hidden_size, vector_output_size, input_dim, multi_input_dim, hidden_dim, output_dim, num_layers, seq_len, target_len, if_multi_variable, use_satellite_memory).to(device)
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    input = (torch.randn((1, args.input_len, 5, img_size, img_size)).cuda(), torch.randn((1, args.output_len, 5, img_size, img_size)).cuda(), torch.randn(1, args.input_len, 4).cuda())
    flops = FlopCountAnalysis(model, input)
    print('flops', flops.total() / 1e9)
    total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print("Number of parameter: %.2fM" % (total/1e6))
    
    start_time = time.time()
    trained_model = train_model(args, model, train_loader, val_loader, criterion1, criterion2, optimizer, epochs=epochs, device=device)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    model = Solar_Predictor(img_channel, img_size, vector_hidden_size, vector_output_size, input_dim, multi_input_dim, hidden_dim, output_dim, num_layers, seq_len, target_len, if_multi_variable, use_satellite_memory).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_model.pth')))
    test_model(args, model, test_loader)
    test_model_fine_grained(args, model)