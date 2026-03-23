import torch
import torch.nn.functional as F
from LLM import NanoWorldLLM

# ==========================================
# 1. 重建词表 (必须和训练时完全一致)
# ==========================================
# 在实际工程中，我们会把字典存成一个 json 文件。
# 这里为了方便演示，我们直接再次读取语料文件来生成相同的字典。
with open('world_lore_massive.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi.get(c, 0) for c in s] # 遇到没见过的字给个默认值
decode = lambda l: ''.join([itos[i] for i in l])

# ==========================================
# 2. 核心魔法：强制 CPU 加载与模型实例化
# ==========================================
# 你的老 Mac 没有 N 卡，所以这里是关键！
device = 'cpu' 
print(f"正在 {device} 上加载世界引擎...")

# 参数必须和训练时一模一样！
d_model = 128
max_seq_len = 64
model = NanoWorldLLM(vocab_size=vocab_size, d_model=d_model, nhead=4, num_layers=4, max_seq_len=max_seq_len)

# map_location='cpu' 是核心：把台式机 GPU 上训练的权重，强行映射到 Mac 的内存里
model.load_state_dict(torch.load('nano_world_weights.pth', map_location=device))
model.to(device)
model.eval() # 开启评估模式，关闭 Dropout 等训练特有机制，生成速度会变快！

# ==========================================
# 3. 编写生成循环 (自回归)
# ==========================================
def generate_lore(model, prompt, max_new_tokens, temperature=0.8, stop_word="[结束]"):
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_seq_len:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] 
            
            # === 删掉之前那个坑人的 forbidden_token_id 封印代码 ===
            # 让它自由发挥，我们用后处理来制裁它
            
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # 实时解码当前所有的文本
            current_text = decode(idx[0].tolist())
            
            # === 新的“完美刹车与剥离”逻辑 ===
            # 只看模型“续写”的那部分内容
            generated_part = current_text[len(prompt):]
            
            # 规则 1：如果它试图写新的标签 "["，立刻打断！
            if "[" in generated_part:
                clean_text = generated_part.split("[")[0].strip()
                return clean_text
            
            # 规则 2：为了兼容老习惯，如果它写出了 "结束" 俩字，也立刻打断！
            if "结束" in generated_part:
                clean_text = generated_part.split("结束")[0].strip()
                return clean_text
                
    # 如果耗尽了 token 还没停，就强行返回当前结果
    return current_text[len(prompt):].strip()

# ========================================== 
# 4. 开始运行！
# ==========================================
print("\n=== 世界记忆提取中 ===")
start_text = "[需求:热]"
generated_text = generate_lore(model, prompt=start_text, max_new_tokens=100, temperature=0.8)

print(f"\n{generated_text}\n")