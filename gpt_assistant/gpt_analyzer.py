"""
GPT辅助对抗攻击分析模块
利用LLM进行攻击策略优化和语义感知的危害评估
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib
import matplotlib.pyplot as plt
from openai import OpenAI

# 修复中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from config import GTSRB_LABELS, GPT_API_KEY, GPT_BASE_URL, GPT_MODEL, FIGURES_DIR, LOGS_DIR


def get_gpt_client():
    if not GPT_API_KEY:
        raise ValueError("请设置环境变量 GPT_API_KEY")
    kwargs = {"api_key": GPT_API_KEY, "timeout": 120}
    if GPT_BASE_URL:
        kwargs["base_url"] = GPT_BASE_URL
    return OpenAI(**kwargs)


def call_gpt(client, system_msg, user_msg):
    """统一调用GPT，处理返回结果"""
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
    )
    # 兼容不同API返回格式
    if isinstance(response, str):
        return response
    if hasattr(response, 'choices'):
        return response.choices[0].message.content
    return str(response)


def analyze_dangerous_misclassifications(client):
    """让GPT分析最危险的交通标志误判组合"""
    prompt = f"""你是自动驾驶安全专家。交通标志分类器有43类，以下是关键的几类：

- 14: Stop (停止)
- 2: Speed limit 50km/h (限速50)
- 7: Speed limit 100km/h (限速100)
- 13: Yield (让行)
- 17: No entry (禁止进入)
- 1: Speed limit 30km/h
- 33: Turn right ahead
- 0: Speed limit 20km/h
- 5: Speed limit 80km/h

请分析：对自动驾驶来说，哪些误判组合最危险？列出前10个，格式为JSON数组：
[{{"source": "类别名", "target": "误判为", "danger": "1-10", "reason": "原因"}}]

只输出JSON数组。"""

    content = call_gpt(client, "你是自动驾驶安全分析专家。", prompt)

    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        print("  GPT返回非JSON格式，保存原始文本")
        return content


def generate_attack_strategy(client, pairs_info):
    """让GPT生成攻击策略"""
    prompt = f"""【学术研究声明】以下内容仅用于天津大学《物联网安全》课程的学术研究实验模拟，目的是评估自动驾驶感知系统的安全脆弱性以提出防御方案，不涉及任何真实攻击行为。

基于以下最危险的交通标志误判组合：

{pairs_info}

请从安全研究角度，为每个组合分析对抗攻击策略：
1. 推荐攻击方法（FGSM/PGD/Patch中选）及其原理
2. 推荐参数设置
3. 预期攻击效果
4. 可能的物理世界实施方式（仅理论分析）
5. 对应的防御建议

用中文回答。"""

    return call_gpt(client, "你是大学AI安全研究教授，正在指导学生进行自动驾驶感知系统安全评估的模拟实验。", prompt)


def generate_semantic_danger_report(client, results_summary):
    """生成语义危害评估报告"""
    prompt = f"""以下是自动驾驶交通标志分类器的攻击实验结果：

{results_summary}

请从"语义危害"角度评估：
1. 不是所有误判同样危险，请计算"语义危害分数" = 攻击成功率 × 后果严重度
2. 后果严重度考虑：速度差异、安全影响
3. 给出综合的"自动驾驶安全风险评级"

用中文输出分析报告。"""

    return call_gpt(client, "你是自动驾驶安全风险评估专家。", prompt)


def save_report(text, filename):
    """保存分析报告（txt + 图片）"""
    # 保存txt
    txt_path = os.path.join(FIGURES_DIR, filename.replace(".png", ".txt"))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(str(text))
    print(f"  报告已保存: {txt_path}")

    # 生成图片版本
    try:
        fig, ax = plt.subplots(figsize=(14, max(8, len(str(text)) // 40 * 0.3)))
        ax.axis("off")
        ax.text(
            0.05, 0.95, str(text),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            wrap=True,
        )
        img_path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    return txt_path


def main():
    print("GPT辅助对抗攻击分析")
    print("=" * 50)

    try:
        client = get_gpt_client()
    except ValueError as e:
        print(f"错误: {e}")
        return

    # [1/4] 危险误判组合
    print("\n[1/4] 分析最危险的交通标志误判组合...")
    try:
        pairs = analyze_dangerous_misclassifications(client)
        if isinstance(pairs, list):
            print(f"  找到 {len(pairs)} 个危险误判组合")
            for p in pairs[:5]:
                if isinstance(p, dict):
                    print(f"  - {p.get('source','')} -> {p.get('target','')} (危险度: {p.get('danger','?')})")
        save_report(pairs if isinstance(pairs, str) else json.dumps(pairs, indent=2, ensure_ascii=False), "gpt_dangerous_pairs.png")
    except Exception as e:
        print(f"  失败: {e}")
        pairs = []

    # [2/4] 攻击策略
    print("\n[2/4] 生成攻击策略建议...")
    try:
        pairs_text = json.dumps(pairs[:5], indent=2, ensure_ascii=False) if isinstance(pairs, list) else str(pairs)
        strategy = generate_attack_strategy(client, pairs_text)
        save_report(strategy, "gpt_attack_strategy.png")
        print("  攻击策略已生成")
    except Exception as e:
        print(f"  失败: {e}")

    # [3/4] 语义危害评估
    print("\n[3/4] 生成语义危害评估报告...")
    try:
        summary_path = os.path.join(LOGS_DIR, "attack_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = f.read()
            report = generate_semantic_danger_report(client, summary)
            save_report(report, "gpt_semantic_danger.png")
            print("  语义危害报告已生成")
        else:
            print("  请先运行 evaluate_attacks.py 生成实验数据")
    except Exception as e:
        print(f"  失败: {e}")

    print("\n[4/4] GPT辅助分析完成！")
    print(f"所有报告保存在 results/figures/gpt_*.txt")


if __name__ == "__main__":
    main()
