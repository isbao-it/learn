"""
阶段4练习：LangChain 调用 DeepSeek 分析爆款文案
场景：心理机制识别 Agent 的雏形（对应任务卡 04）
展示：普通文本输出 + 结构化 JSON 输出
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field


# ---------- 1. 读取项目 .env 配置（不硬编码 key） ----------
# 手动解析 .env 文件，读取以 # 开头的注释和包含 = 的键值对，将其存入字典。
# strip() 是字符串（str）对象的一个内置方法，用于移除字符串首尾（左右两端）指定的字符。
def load_env(path):
    env = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

# 配置好的模型参数和 API Key 都放在项目根目录的 .env 文件里，避免硬编码在代码中
env = load_env(r"C:\work\dev\projects\bao-kuan-xin-fa\.env")

# ---------- 2. 封装模型（一套代码接 DeepSeek） ----------
# 模型实例化
llm = ChatOpenAI(
    base_url=env["LLM_BASE_URL"],
    api_key=env["LLM_API_KEY"],
    model=env["LLM_MODEL"],
    temperature=0.3,#随机性
)

# 待分析的爆款文案
TEXT = "为什么你越努力越焦虑？三个原因拆穿你的情绪黑洞。第一，你一直在跟别人的进度条较劲。第二，你把努力当成了赎罪。第三，你从来没有问过自己真正想要什么。"


# ========== 演示1：普通文本输出（StrOutputParser） ==========
print("=" * 55)
print("演示1：自由分析（输出纯文本）")
print("=" * 55)

#提示词模板
prompt = PromptTemplate.from_template(
    "你是一位爆款视频分析师。请用 3 句话以内分析这段文案为什么容易引发共鸣：\n{text}"
)
# 前一个命令的输出，自动成为后一个命令的输入。
# 在 LangChain 中，LCEL 规定：只要一个对象实现了 invoke（同步）或 ainvoke（异步）方法，
# 它就可以成为管道中的一个节点（Runnable）。
chain = prompt | llm | StrOutputParser()   # ← LCEL 管道
result = chain.invoke({"text": TEXT})
print(result)
print()


# ========== 演示2：Pydantic 约束结构化 JSON 输出（PydanticOutputParser） ==========
print("=" * 55)
print("演示2：结构化输出（自动转成 Pydantic 模型）")
print("=" * 55)

# 定义 PsychologyResult 继承自 Pydantic 的 BaseModel，这相当于给大模型划定了一个严格的“数据填写模板”
class PsychologyResult(BaseModel):
    mechanisms: list[str] = Field(description="识别出的心理机制，3-5个")
    core_emotion: str = Field(description="最核心的情绪")
    suggestion: str = Field(description="给创作者的一条改进建议")
# 大模型是根据 description 来思考和输出的

# 演示2 专用：开启 DeepSeek JSON 模式（prompt 必须含 json 字样）
llm_json = ChatOpenAI(
    base_url=env["LLM_BASE_URL"],
    api_key=env["LLM_API_KEY"],
    model=env["LLM_MODEL"],
    temperature=0.3,
    model_kwargs={"response_format": {"type": "json_object"}},
)

parser = PydanticOutputParser(pydantic_object=PsychologyResult)

prompt2 = PromptTemplate.from_template(
    "你是一位爆款视频分析师。分析这段文案的心理机制。\n"
    "文案：{text}\n\n"
    "输出要求：直接输出 JSON 对象本身（不要用 properties 包裹），\n{format_instructions}"
)
chain2 = prompt2 | llm_json | parser           # ← 管道末端是解析器

raw = chain2.invoke({"text": TEXT, "format_instructions": parser.get_format_instructions()})
# get_format_instructions() 方法会生成一段自然语言提示，告诉大模型“你应该按照什么格式输出你的回答”
# aw: 接收链执行后的返回结果，类型是 PsychologyResult
print(f"类型: {type(raw).__name__}  ← 已经是 Pydantic 模型，不是字符串！")
print(f"心理机制: {raw.mechanisms}")
print(f"核心情绪: {raw.core_emotion}")
print(f"改进建议: {raw.suggestion}")
print()
print("直接序列化成 JSON 给前端：")
print(raw.model_dump_json())
