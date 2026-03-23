import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from LLM import NanoWorldLLM

# ==========================================
# 1. 准备数据与“字典” (Tokenization)
# ==========================================
print("正在读取世界观语料...")
with open('world_lore_massive.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 找出文本中所有出现过的独立字符（这就是我们的“词表”）
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"语料总字数: {len(text)}")
print(f"词表大小 (独立字符数): {vocab_size}")

# 制作两个字典：字到数字(stoi)，数字到字(itos)
# 大模型不认识字，只认识数字 ID
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # 编码器：把字符串变成整数列表
decode = lambda l: ''.join([itos[i] for i in l]) # 解码器：把整数列表变回字符串

# 把整本小说全部转化为数字张量 (Tensor)
data = torch.tensor(encode(text), dtype=torch.long)

# ==========================================
# 2. 制作“喂食管” (Data Loader)
# ==========================================
# 划分训练集和验证集 (90% 训练，10% 考试)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split, batch_size, max_seq_len):
    """
    随机从语料中切出一小段作为输入 X，和往后移一位的正确答案 Y
    """
    data_source = train_data if split == 'train' else val_data
    # 随机生成 batch_size 个起始位置
    ix = torch.randint(len(data_source) - max_seq_len - 1, (batch_size,))
    
    # 拼凑出 X 和 Y
    x = torch.stack([data_source[i : i+max_seq_len] for i in ix])
    y = torch.stack([data_source[i+1 : i+max_seq_len+1] for i in ix])
    return x, y

# ==========================================
# 3. 点火！台式机显卡开始炼丹 (Training Loop)
# ==========================================
# 自动检测你的台式机有没有N卡(CUDA)或者Mac的M芯片(MPS)，都没有就用CPU
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"当前使用的计算设备: {device}")

# 超参数设置 (你可以根据你台式机的显存大小调整)
batch_size = 32      # 每次同时看 32 句话
max_seq_len = 64     # 每句话最长 64 个字
d_model = 128        # 模型的脑容量
learning_rate = 1e-3 # 学习率（步子迈多大）
max_iters = 3000     # 训练循环次数

# 实例化我们在上一步写的 Nano-LLM 模型
model = NanoWorldLLM(vocab_size=vocab_size, d_model=d_model, nhead=4, num_layers=4, max_seq_len=max_seq_len)
model = model.to(device) # 把模型扔进显卡！

# 使用 AdamW 优化器（大模型炼丹标配）
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("开始训练...")
for iter in range(max_iters):
    
    # 获取一批数据，并扔进显卡
    xb, yb = get_batch('train', batch_size, max_seq_len)
    xb, yb = xb.to(device), yb.to(device)
    
    # 前向传播：模型试着去猜，并计算出和正确答案的差距 (Loss)
    logits, loss = model(xb, targets=yb)
    
    # 反向传播：极其关键的清零、求导、更新权重三连击！
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # 每 300 步打印一次进度
    if iter % 300 == 0:
        print(f"步骤 {iter} | 训练集 Loss: {loss.item():.4f}")

# 训练结束后，把模型脑子里的权重保存下来！这就是我们要带去 MacBook 上的东西
torch.save(model.state_dict(), 'nano_world_weights.pth')
print("训练完成！模型权重已保存为 nano_world_weights.pth (这个文件只有几MB大小)")