from google import genai
import random
import time

# 1. 配置你的 Gemini API Key
GOOGLE_API_KEY = "AIzaSyBFJ2mpaLi-IY_Z4tQtP-QtW6t2zWkGSl0"
genai.configure(api_key=GOOGLE_API_KEY)

# 推荐使用 flash 模型，速度极快，用来生成这种短语料简直是大炮打蚊子
model = genai.GenerativeModel('gemini-1.5-flash')

# 2. 状态库
needs = ["渴", "饿", "热", "冷", "疲惫", "恐惧", "狂热", "繁衍", "生病"]
terrains = ["沙漠", "森林", "冰原", "沼泽", "高山", "废墟", "平原", "洞穴"]
morales = ["高昂", "崩溃", "麻木", "警惕", "悠闲"]

# 总共想生成多少条？每次让 Gemini 吐多少条？
TOTAL_SAMPLES = 1000
BATCH_SIZE = 50  # 每次请求生成 50 条，速度起飞

def generate_gemini_batch(batch_size):
    # 组装这 50 条的随机标签
    batch_tags = []
    for _ in range(batch_size):
        categories = [
            f"[需求:{random.choice(needs)}]",
            f"[地形:{random.choice(terrains)}]",
            f"[士气:{random.choice(morales)}]"
        ]
        num_tags = random.randint(1, 3)
        selected_tags = random.sample(categories, num_tags)
        random.shuffle(selected_tags)
        batch_tags.append("".join(selected_tags))
    
    # 将 50 个标签组合拼成一个带编号的列表发给大模型
    tags_list_str = "\n".join([f"{i+1}. {tag}" for i, tag in enumerate(batch_tags)])
    
    prompt = f"""
    你是一个奇幻生存游戏的文案生成器。
    我将给你 {batch_size} 组游戏状态标签。请为每一组标签写一句简短（30字以内）、极具画面感的生存描述。
    
    要求：
    1. 严格按照我给的标签顺序和内容来写。
    2. 每一行的输出格式必须是：标签内容 描述正文 [结束]
    3. 不要输出任何多余的开头、结尾、Markdown格式或空行！纯文本！
    
    这是标签列表：
    {tags_list_str}
    
    请直接输出这 {batch_size} 行结果：
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"API 请求失败: {e}")
        return ""

# 3. 开始批量轰炸
print(f"🚀 开始使用 Gemini 提速！目标：{TOTAL_SAMPLES} 条数据...")
training_data = ""

for i in range(0, TOTAL_SAMPLES, BATCH_SIZE):
    print(f"正在生成第 {i+1} 到 {i+BATCH_SIZE} 条...")
    
    batch_result = generate_gemini_batch(BATCH_SIZE)
    if batch_result:
        training_data += batch_result + "\n"
        
    # 稍微停顿一下，防止触发免费 API 的每分钟频率限制 (Rate Limit)
    time.sleep(2)

# 4. 写入文件，准备炼丹！
with open("world_lore_massive.txt", "w", encoding="utf-8") as f:
    f.write(training_data)

print(f"🎉 搞定！数据已全部保存到 world_lore_massive.txt")