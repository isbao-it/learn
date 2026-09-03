"""
阶段5练习：LangGraph 多节点流水线
场景：爆款心法 Agent 流水线简化版（任务卡 03→04）
节点：拆叙事结构 → 识别心理机制 → 生成创作建议
"""

from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

# 手动解析本地 .env 文件，加载 LLM 的 API 地址、密钥和模型名称。
def load_env(path):
    env = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


env = load_env(r"C:\work\dev\projects\bao-kuan-xin-fa\.env")
# 普通大模型实例，用于生成自然语言文本。
llm = ChatOpenAI(base_url=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"],
                 model=env["LLM_MODEL"], temperature=0.3)
# 开启了 response_format={"type": "json_object"} 的大模型实例，强制其输出合法的 JSON 字符串。
llm_json = ChatOpenAI(base_url=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"],
                      model=env["LLM_MODEL"], temperature=0.3,
                      model_kwargs={"response_format": {"type": "json_object"}})

TEXT = "为什么你越努力越焦虑？三个原因拆穿你的情绪黑洞。第一，你一直在跟别人的进度条较劲。第二，你把努力当成了赎罪。第三，你从来没有问过自己真正想要什么。看完这条视频，你会重新理解努力。"

# ---------- 结构化输出模型 ----------
class Segments(BaseModel):
    """
    文案段落模型类，用于存储文案的各个段落
    继承自BaseModel，提供了数据验证和序列化功能
    """
    segments: list[str] = Field(description="文案段落，每段一句话")

class Psychology(BaseModel):
    """
    心理机制模型类，用于存储与文案段落对应的心理机制
    继承自BaseModel，提供了数据验证和序列化功能
    """
    mechanisms: list[str] = Field(description="每段对应的心理机制")


# ---------- 节点1：拆叙事结构 ----------

# 导入PydanticOutputParser用于解析Pydantic模型
seg_parser = PydanticOutputParser(pydantic_object=Segments)

# 创建一个提示模板模拟输入提示词，用于将文案拆分为叙事段落
prompt_seg = PromptTemplate.from_template(
    "你是一名叙事结构分析师。把下面文案拆成 2-4 个叙事段落（按逻辑断点）。"
    "文案：{text}"
    "直接输出 JSON 对象本身：{format_instructions}"
)

# 创建处理链，将提示模板、语言模型和解析器串联起来
chain_seg = prompt_seg | llm_json | seg_parser

# 添加节点
def deconstruct(state):
    result = chain_seg.invoke({"text": state["text"],
                               "format_instructions": seg_parser.get_format_instructions()})
    return {"segments": result.segments}


# ---------- 节点2：识别心理机制 ----------
psy_parser = PydanticOutputParser(pydantic_object=Psychology)
prompt_psy = PromptTemplate.from_template(
    "你是一名爆款心理学分析师。分析下面每个段落的心理机制（如损失厌恶、身份认同、社会比较）。"
    "段落：{segments}"
    "直接输出 JSON 对象本身：{format_instructions}"
)
chain_psy = prompt_psy | llm_json | psy_parser


def identify(state):
    result = chain_psy.invoke({"segments": state["segments"],
                               "format_instructions": psy_parser.get_format_instructions()})
    return {"psychology": result.mechanisms}


# ---------- 节点3：汇总报告 ----------
prompt_sum = PromptTemplate.from_template(
    "你是爆款视频总编。基于下面的叙事结构和心理机制，输出一份 3 句话的创作建议报告。"
    "叙事结构：{segments}"
    "心理机制：{psychology}"
)
chain_sum = prompt_sum | llm | StrOutputParser()


def summarize(state):
    report = chain_sum.invoke({"segments": state["segments"], "psychology": state["psychology"]})
    return {"report": report}


# ---------- 组装状态图 ----------
# 定义一个名为AnalysisState的类型字典类，继承自TypedDict
class AnalysisState(TypedDict):
    text: str          # 表示要分析的文本内容，类型为字符串
    segments: list     # 表示文本的分段结果，类型为列表
    psychology: list   # 表示心理学分析结果，类型为列表
    report: str        # 表示生成的分析报告，类型为字符串

# 组装为节点合边的状态图，定义节点之间的依赖关系和执行顺序
g = StateGraph(AnalysisState)
g.add_node("deconstruct", deconstruct)
g.add_node("identify", identify)
g.add_node("summarize", summarize)
g.add_edge("deconstruct", "identify")
g.add_edge("identify", "summarize")
g.set_entry_point("deconstruct")
g.set_finish_point("summarize")
# 编译状态图，生成可执行的流水线
app = g.compile()

# ---------- 执行流水线 ----------
result = app.invoke({"text": TEXT, "segments": [], "psychology": [], "report": ""})

print("=" * 55)
print("流水线执行结果")
print("=" * 55)
print("节点1 拆解出的段落:")
for i, s in enumerate(result["segments"], 1):
    print(f"   {i}. {s}")
print()
print("节点2 识别出的心理机制:")
for i, m in enumerate(result["psychology"], 1):
    print(f"   {i}. {m}")
print()
print("节点3 创作建议报告:")
print(result["report"])
