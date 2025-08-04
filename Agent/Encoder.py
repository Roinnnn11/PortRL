import torch.nn as nn
import torch

class Time2Vec(nn.Module):
    '''
    对时间进行编码：年/月/日/小时
    Time2Vec module: 将标量时间编码为向量（线性 + 正弦部分）
    '''
    def __init__(self, input_dim=1, out_dim=8):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # 线性部分
        self.freq = nn.Linear(input_dim, out_dim - 1)  # 正弦部分

    def forward(self, x):  # x: (batch, N, 1)
        """
        t: (B, N, 1) 表示时间，如天数/小时/时间戳
        return: (B, N, embed_dim)
        """
        linear_part = self.linear(x)  # (batch, N, 1)
        periodic_part = torch.sin(self.freq(x))  # (batch, N, out_dim - 1)
        return torch.cat([linear_part, periodic_part], dim=-1)  # (batch, N, out_dim)

class ContainerEncoder(nn.Module):
    '''Tranformer版，后续也可以考虑使用GAT'''
    def __init__(self, 
                 num_box_types,      # 集装箱种类数量（如进/出港+箱型）
                 input_dim, 
                 type_embed_dim=16,    
                 time_embed_dim=8,     
                 cont_dim=3,      # 连续特征数量，如[重量、体积、高度]
                 model_dim=64,         
                 num_layers=2,
                 nhead=4,
                 max_containers=960):
        super().__init__()

        # --类型嵌入--（例如：进/出港两种 + 箱类型 4种 → 总共 type_vocab_size）
        self.box_type_embed = nn.Embedding(num_box_types, 8)
        self.inout_embed = nn.Embedding(2, 4)  # 0: 入港, 1: 出港
        # --时间特征编码器--：入港时间、出港时间各自一个 Time2Vec
        self.time2vec_in = Time2Vec(input_dim=1, out_dim=time_embed_dim)
        self.time2vec_out = Time2Vec(input_dim=1, out_dim=time_embed_dim)

        # ----连续数值特征（如：重量、体积）线性映射----
        self.cont_linear = nn.Linear(cont_dim, 16)
        # --- 所有特征线性映射至 model_dim ---
        # total_input_dim = 8 + 4 + time_embed_dim * 2 + 16  # 拼接后的维度
        # self.input_proj = nn.Linear(total_input_dim, model_dim)
       
        self.input_proj = nn.Linear(input_dim, model_dim)  # 输入维度映射到 model_dim
        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_containers, model_dim))

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

       

    # def forward(self, box_type_ids, inout_flag,time_in, time_out, cont_feats):
    #     """
    #     对该批次箱子，做一个multi-head self-attention。
    #     输入：
    #         type_ids:     (batch, N)         -- 类型id，如 0=小箱, 1=大箱...
    #         input_flag:   (batch,N)          -- 进港/离港
    #         time_in/out:  (batch, N, 1)      -- 入/出港时间（天或小时 float 格式）
    #         cont_feats:   (batch, N, F)      -- 连续特征，如[重量, 体积, 高度]（待定）
    #     输出：
    #         encoded:(B,N,model_dim)
    #     """
    #      # --- 特征嵌入 ---
    #     box_emb = self.box_type_embed(box_type_ids)       # (B, N, 8)
    #     inout_emb = self.inout_embed(inout_flag)         # (B, N, 4)
    #     time_in_emb = self.time2vec_in(time_in)          # (B, N, 8)
    #     time_out_emb = self.time2vec_out(time_out)       # (B, N, 8)
    #     cont_emb = self.cont_linear(cont_feats)          # (B, N, 16)

    #     # --- 拼接所有特征 ---
    #     x = torch.cat([box_emb, inout_emb, time_in_emb, time_out_emb, cont_emb], dim=-1)  # (B, N, total_dim)

    #     # --- 投影到 Transformer 输入维度 ---
    #     x = self.input_proj(x)                           # (B, N, model_dim)

    #     # --- 加上位置编码 ---
    #     x = x + self.pos_embed[:, :x.size(1), :]

    #     # --- Transformer 编码 ---
    #     encoded = self.encoder(x)                        # (B, N, model_dim)

    #     return encoded
    def forward(self, x):  # x: Tensor of shape [B, T, 20]
        # 示例：使用一个线性层作为 embedding
        # 你也可以用 transformer encoder 或 GRU/LSTM 等
        # 映射到 model_dim
        x = self.input_proj(x)  # [B, T, model_dim]

        # 添加位置编码（截断到当前序列长度）
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]  # [B, T, model_dim]

        # Transformer Encoder
        x = self.encoder(x)  # [B, T, model_dim]
        return x


class RLDecoder(nn.Module):
    def __init__(self,model_dim,action_dim):
        '''用一个multi-head cross-attention，获取全局信息+context信息
        当前批次箱子为Query，历史箱子为key/value'''
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=4)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.actor = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, action_dim)
        )

    def forward(self, current_encoded, historical_encoded):  # both shape: (B, N, D)
        tgt = current_encoded.transpose(0, 1)     # (1, B, D)
        memory = historical_encoded.transpose(0, 1)  # (N, B, D)
        out = self.decoder(tgt, memory)           # (1, B, D)
        out = out.squeeze(0)                      # (B, D)
        # return out
        return self.actor(out)                    # (B, action_dim)

    
class Critic(nn.Module):
    '''用于估算当前状态的状态值 V(s)，即当前策略下，当前状态的“好坏”评估。'''
    def __init__(self, model_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 1)
        )

    def forward(self, encoded):  # encoded: (B, D)
        return self.net(encoded)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, model_dim, action_dim):
        super().__init__()
        self.encoder = ContainerEncoder(num_box_types=3,input_dim=20, model_dim=128)
        self.decoder = RLDecoder(model_dim, action_dim)
        self.critic = Critic(model_dim)

    def forward(self, current, history):
        history_encoded = self.encoder(history)
        current_encoded = self.encoder(current)[:, 0:1, :]  # 取当前箱子
        # print("current_encoded shape:", current_encoded.shape)
        # print("history_encoded shape:", history_encoded.shape)
        logits = self.decoder(current_encoded, history_encoded)
        value = self.critic(current_encoded.squeeze(1))
        return logits, value


# batch_size = 32
# num_containers = 100
# cont_feat_dim = 3

# type_ids = torch.randint(0, 3, (batch_size, num_containers))          # 6种类别
# inout_flags = torch.randint(0,1,(batch_size,num_containers))          # 进港/离港
# time_in = torch.rand(batch_size, num_containers, 1) * 3650            # 入港时间（10年）
# time_out = torch.rand(batch_size, num_containers, 1) * 3650           # 出港时间
# cont_feats = torch.rand(batch_size, num_containers, cont_feat_dim)    # 重量、体积等

# model = ContainerEncoder(num_box_types=3)
# output = model(type_ids,inout_flags, time_in, time_out, cont_feats)  # (batch, N, model_dim)
# print(output)

