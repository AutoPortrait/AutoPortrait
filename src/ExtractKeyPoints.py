from LLM import LLMInstructional
from Prompts import Prompts
import warnings

prompts = Prompts()


async def extract_key_points(input: str) -> list[str]:
    while True:
        result = await LLMInstructional.process(prompts.prompt_extract_key_points, input)
        result = result.strip()
        if result.find("1.") != -1:
            warnings.warn(f"LLM output includes '1.'. Content:\n{result}\nRetry.")
            continue
        if result.find("**") != -1:
            warnings.warn(f"LLM output includes '**'. Content:\n{result}\nRetry.")
            continue
        if result.find("\n\n") != -1:
            warnings.warn(f"LLM output includes empty line. Content:\n{result}\nRetry.")
            continue
        if len(result.split("\n")) < 3:
            warnings.warn(f"LLM output is too short. Content:\n{result}\nRetry.")
            continue
        return [line.strip() for line in result.split("\n")]


def main():
    input = """
这一画像描绘了当代中国数学专业本科生群体，特别是在职业规划和个人生活选择方面面临的挑战和矛盾。他们通常在家庭期望和个人兴趣之间寻找平衡，并试图在快速变化的社会中找到自己的位置。

该群体具有以下特征：

1. **教育背景**：他们通常来自中国的小城镇或农村地区，通过国家专项计划等政策支持考入985或211高校，专业以数学为主，并可能对计算机科学、金融或人工智能等领域感兴趣。

2. **家庭期望**：家庭通常对他们抱有较高的期望，希望他们能够进入稳定的体制内工作，如公务员或教师。家庭成员可能对政治和权力有强烈的追求，并期望通过孩子的职业来提升家庭的社会地位。

3. **个人兴趣与职业规划**：尽管家庭有明确的期望，但他们个人可能更倾向于追求自由和创造性较强的职业，如进入互联网大厂工作。他们通常对体制内的文化和人情世故持批判态度，认为这种环境不利于个人技能的发挥和职业发展。

4. **工作稳定性与财务自由**：他们认识到工作稳定性对于个人和家庭的重要性，但同时也在考虑财务自由的可能性。他们认为财务自由虽然难以实现，但可能是实现个人自由和生活选择的重要途径。

5. **社会关系与个人空间**：他们对体制内的人情世故和权力斗争持批判态度，并试图在社会关系中保持个人空间和独立性。他们通常不喜欢被迫参与家庭安排的社交活动，特别是那些他们认为没有实际意义的活动。

6. **对未来的期望**：他们期望未来的生活能够有足够的自由和稳定性，能够让他们有时间追求个人兴趣和享受生活。他们可能更倾向于在小城市生活，以减少生活压力和房贷负担。

7. **个人特质**：他们通常是直率的人，喜欢表达自己的观点，但有时可能会因为过于直接而得罪人。他们对自己的专业有一定的认同感，但也认识到自己在某些领域需要进一步提升。

这一画像反映了中国当代大学生在个人成长和职业规划中的矛盾和挑战，他们需要在家庭期望和个人兴趣之间找到平衡，同时也要面对快速变化的社会环境和经济形势。
"""
    output = extract_key_points(input)
    for line in output:
        print(line)


if __name__ == "__main__":
    main()
