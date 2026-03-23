import torch
import torch.nn as nn
import torch.nn.functional as F

class NanoWorldLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4, max_seq_len=256):
        """
        vocab_size: 词表大小（你的世界一共有多少个不同的字/词）
        d_model: 向量维度（脑容量，越大越聪明，但越慢）
        nhead: 注意力头数（同时思考多少个不同的上下文维度）
        num_layers: 神经网络的层数
        max_seq_len: 模型一次最多能看多长的句子
        """
        super().__init__()
        
        # 1. 词嵌入层：把 ID 变成高维向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. 位置嵌入层：告诉模型词的先后顺序（最简单的绝对位置编码）
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 3. 核心大脑：Transformer 层
        # batch_first=True 表示数据的形状是 [批次大小, 句子长度, 向量维度]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 输出层：把思考完的向量，重新映射回词表的概率分布
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz):
        """
        生成因果掩码 (Causal Mask)
        创造一个右上角全是负无穷、左下角全是 0 的矩阵。
        作用：强制模型在预测下一个词时，不能“偷看”未来的词。
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, idx, targets=None):
        """
        idx: 输入的词 ID 序列，形状 [B, T] (Batch, Time/Sequence_length)
        targets: 正确答案的词 ID 序列（训练时提供）
        """
        B, T = idx.shape
        
        # 获取设备 (CPU 或 GPU)
        device = idx.device
        
        # 生成 0 到 T-1 的位置 ID
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        
        # 将词向量和位置向量相加
        x = self.token_embedding(idx) + self.position_embedding(pos)
        
        # 生成掩码，防止偷看未来
        mask = self.generate_square_subsequent_mask(T).to(device)
        
        # 送入 Transformer 大脑进行深层思考
        # 注意：PyTorch 的 TransformerEncoder 默认需要掩码来表现得像一个 Decoder
        x = self.transformer(x, mask=mask, is_causal=True)
        
        # 输出每个位置预测下一个词的原始分数 (Logits)
        logits = self.lm_head(x)
        
        # 如果提供了正确的标签 (targets)，就计算损失 (Loss)
        loss = None
        if targets is not None:
            # CrossEntropyLoss 要求输入的形状是 [B*T, vocab_size]，所以要 reshape 一下
            B, T, C = logits.shape
            logits_view = logits.view(B * T, C)
            targets_view = targets.view(B * T)
            loss = F.cross_entropy(logits_view, targets_view)
            
        return logits, loss