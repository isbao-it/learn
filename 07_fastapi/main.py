"""
阶段7练习：FastAPI 把 AI 流水线包成 REST API
场景：爆款心法后端（任务卡 10 的雏形）
接口：
  POST /analyze   输入文案 → 返回分析结果
  GET  /health    健康检查
"""

from typing import TypedDict
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

# ---------- 读取 DeepSeek 配置 ----------
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

# ---------- Pydantic：API 请求/响应模型（阶段1 的知识用上了） ----------
class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, description="视频文案")

class AnalyzeResponse(BaseModel):
    segments: list[str] = Field(description="叙事段落")
    psychology: list[str] = Field(description="心理机制")
    report: str = Field(description="创作建议")

# ---------- LangGraph 流水线（阶段5 的知识用上了） ----------
class Segments(BaseModel):
    segments: list[str] = Field(description="文案段落")

class Psychology(BaseModel):
    mechanisms: list[str] = Field(description="心理机制列表")

class AnalysisState(TypedDict):
    text: str
    segments: list
    psychology: list
    report: str


def build_pipeline():
    """
    构建一个用于文案分析的多阶段处理流水线
    该流水线包含三个主要步骤：叙事结构分析、心理机制分析和创作建议生成
    """
    # 初始化语言模型，使用OpenAI的Chat模型
    llm = ChatOpenAI(base_url=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"],
                     model=env["LLM_MODEL"], temperature=0.3)
    # 初始化专门用于JSON输出的语言模型
    llm_json = ChatOpenAI(base_url=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"],
                          model=env["LLM_MODEL"], temperature=0.3,
                          model_kwargs={"response_format": {"type": "json_object"}})



    # 创建叙事结构分析链
    seg_parser = PydanticOutputParser(pydantic_object=Segments)  # 定义输出解析器
    chain_seg = PromptTemplate.from_template(  # 创建提示模板
        "你是叙事结构分析师。把文案拆成 2-4 个叙事段落。文案：{text}"
        "直接输出 JSON 对象本身：{format_instructions}"
    ) | llm_json | seg_parser  # 组合模板、模型和解析器



    # 创建心理机制分析链
    psy_parser = PydanticOutputParser(pydantic_object=Psychology)  # 定义输出解析器
    chain_psy = PromptTemplate.from_template(  # 创建提示模板
        "你是爆款心理学分析师。分析每个段落的心理机制（损失厌恶、身份认同、社会比较等）。段落：{segments}"
        "直接输出 JSON 对象本身：{format_instructions}"
    ) | llm_json | psy_parser  # 组合模板、模型和解析器



    # 创建创作建议生成链
    chain_sum = PromptTemplate.from_template(  # 创建提示模板
        "你是爆款视频总编。基于叙事结构和心理机制，输出 3 句话创作建议。叙事结构：{segments} 心理机制：{psychology}"
    ) | llm | StrOutputParser()  # 组合模板、模型和字符串解析器

    # 定义三个处理函数
    def deconstruct(state):
        """叙事结构分析函数：将输入文案拆分为叙事段落"""
        r = chain_seg.invoke({"text": state["text"],
                              "format_instructions": seg_parser.get_format_instructions()})
        return {"segments": r.segments}

    def identify(state):
        """心理机制分析函数：分析每个段落的心理机制"""
        r = chain_psy.invoke({"segments": state["segments"],
                              "format_instructions": psy_parser.get_format_instructions()})
        return {"psychology": r.mechanisms}

    def summarize(state):
        """创作建议生成函数：基于分析结果生成创作建议"""
        return {"report": chain_sum.invoke({"segments": state["segments"],
                                            "psychology": state["psychology"]})}



    # 创建状态图并添加节点和边
    g = StateGraph(AnalysisState)  # 初始化状态图
    g.add_node("deconstruct", deconstruct)  # 添加叙事结构分析节点
    g.add_node("identify", identify)  # 添加心理机制分析节点
    g.add_node("summarize", summarize)  # 添加创作建议生成节点
    g.add_edge("deconstruct", "identify")  # 连接叙事结构分析到心理机制分析
    g.add_edge("identify", "summarize")  # 连接心理机制分析到创作建议生成  # 设置入口点为叙事结构分析
    g.set_entry_point("deconstruct")
    g.set_finish_point("summarize")
    return g.compile()

pipeline = build_pipeline()  # 服务启动时构建一次

# ---------- FastAPI 应用 ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("baokuan-api")

app = FastAPI(title="爆款心法 API", version="0.1.0")

# CORS：允许前端（Streamlit 等不同端口）跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 生产环境应限制为具体域名
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    logger.info("收到分析请求，文案长度: %d", len(req.text))
    try:
        result = pipeline.invoke({"text": req.text, "segments": [], "psychology": [], "report": ""})
    except Exception as e:
        logger.error("分析失败: %s", e)
        raise HTTPException(status_code=502, detail=f"AI 分析服务暂时不可用: {type(e).__name__}")
    return AnalyzeResponse(
        segments=result["segments"],
        psychology=result["psychology"],
        report=result["report"],
    )
