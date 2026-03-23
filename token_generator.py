from google import genai
import random
import time

# 1. 配置你的 Gemini API Key
GOOGLE_API_KEY = "AIzaSyBFJ2mpaLi-IY_Z4tQtP-QtW6t2zWkGSl0"
client =  genai.Client(api_key=GOOGLE_API_KEY)

GEMINI_MODEL = "gemini-2.5-flash"  # 选择 Gemini 模型

# 2. 扩充状态库：加入决定文明灵魂的“风格”维度
styles = ["粗鄙野蛮", "文雅书卷", "狂热信仰", "抑郁悲观", "理智冷酷", "傲慢贵族"]
needs = ["渴", "饿", "热", "冷", "疲惫", "恐惧", "狂热", "繁衍", "生病"]
terrains = ["沙漠", "森林", "冰原", "沼泽", "高山", "废墟", "平原", "洞穴"]
morales = ["高昂", "崩溃", "麻木", "警惕", "悠闲"]

TOTAL_SAMPLES = 1000
BATCH_SIZE = 50
OUTPUT_FILE = "world_lore_styled.txt"

# 运行前先清空旧文件，或者创建一个空文件
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("")

def generate_gemini_batch(batch_size):
    batch_tags = []
    for _ in range(batch_size):
        # 强制带上“风格”标签，作为文明性格的基础
        style_tag = f"[风格:{random.choice(styles)}]"
        
        # 客观环境标签随机组合 (1到3个)
        categories = [
            f"[需求:{random.choice(needs)}]",
            f"[地形:{random.choice(terrains)}]",
            f"[士气:{random.choice(morales)}]"
        ]
        num_tags = random.randint(1, 3)
        selected_tags = random.sample(categories, num_tags)
        random.shuffle(selected_tags)
        
        # 拼接在一起，例如：[风格:粗鄙野蛮][地形:沼泽][需求:冷]
        full_tag = style_tag + "".join(selected_tags)
        batch_tags.append(full_tag)
    
    tags_list_str = "\n".join([f"{i+1}. {tag}" for i, tag in enumerate(batch_tags)])
    
    # 核心魔法：用 Prompt 强制大模型进行“风格化角色扮演”
    prompt = f"""
    你是一个奇幻生存游戏的世界观文案生成器。
    我将给你 {batch_size} 组游戏状态标签。每一组的第一个标签是【风格】（代表该文明的文化程度或性格），后面是他们面临的【客观环境】。
    
    请为每一组标签写一句简短（30字以内）的生存描述。
    核心要求：必须极其强烈地体现出【风格】标签所要求的语气和用词习惯！
    - 粗鄙野蛮：用词粗俗，暴躁，多用俚语或骂人的话。
    - 文雅书卷：使用半文言文、四字成语，语气克制悲悯。
    - 狂热信仰：言必称神明、试炼、罪人，情绪极端、神神叨叨。
    - 抑郁悲观：充满绝望、死气沉沉、放弃挣扎。
    - 理智冷酷：像机器一样客观分析生存概率，毫无感情。
    - 傲慢贵族：充满优越感，嫌弃环境脏乱，抱怨生活品质下降。
    
    输出格式必须是：标签组合 描述正文 [结束]
    不要有任何多余的开头、结尾、Markdown符号或空行！纯文本！
    
    这是标签列表：
    {tags_list_str}
    """
    
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            input=prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"API 请求失败: {e}")
        return ""

print(f"🚀 开始生成并实时写入！目标：{TOTAL_SAMPLES} 条数据...")

# 3. 边生成，边写入！
for i in range(0, TOTAL_SAMPLES, BATCH_SIZE):
    print(f"正在生成第 {i+1} 到 {i+BATCH_SIZE} 条...")
    
    batch_result = generate_gemini_batch(BATCH_SIZE)
    if batch_result:
        # 使用 "a" (append) 模式打开文件，将新获取的一批数据追加到末尾
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(batch_result + "\n")
        print(f"✅ 第 {i+1} 到 {i+BATCH_SIZE} 条已成功落盘！")
        
    # 稍微停顿一下，防止触发免费 API 的每分钟频率限制 (Rate Limit)
    time.sleep(2)

print(f"🎉 搞定！带有文明风格的数据已全部安全保存在 {OUTPUT_FILE}")