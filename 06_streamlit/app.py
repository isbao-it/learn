"""
阶段6练习：Streamlit 前端 —— 爆款分析器
整合：LangGraph 流水线 + DeepSeek + Streamlit 展示
对应项目：任务卡 11（前端页面）
"""

import streamlit as st  # 导入Streamlit库，用于构建Web应用界面
from typing import TypedDict  # 导入TypedDict，用于定义字典类型
from langgraph.graph import StateGraph  # 导入StateGraph，用于构建有向图工作流
from langchain_openai import ChatOpenAI  # 导入OpenAI聊天模型
from langchain_core.prompts import PromptTemplate  # 导入提示词模板
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser  # 导出输出解析器
from pydantic import BaseModel, Field  # 导入Pydantic基础模型和字段定义


# ---------- 读取配置 ----------
def load_env(path):  # 定义加载环境变量的函数
    env = {}  # 创建空字典存储环境变量
    with open(path, encoding="utf-8") as f:  # 打开文件
        for line in f:  # 逐行读取
            line = line.strip()  # 去除首尾空白
            if line and not line.startswith("#") and "=" in line:  # 过滤注释和空行
                k, v = line.split("=", 1)  # 分割键值对
                env[k.strip()] = v.strip()  # 去除空白并存储
    return env


env = load_env(r"C:\work\dev\projects\bao-kuan-xin-fa\.env")  # 加载环境变量文件

# ---------- 构建 LangGraph 流水线 ----------
class Segments(BaseModel):  # 定义文案段落数据模型
    segments: list[str] = Field(description="文案段落")  # 定义段落字段为字符串列表

class Psychology(BaseModel):  # 定义心理机制数据模型
    mechanisms: list[str] = Field(description="心理机制列表")  # 定义机制字段为字符串列表

class AnalysisState(TypedDict):  # 定义分析状态字典类型
    text: str  # 原始文本
    segments: list  # 文案段落
    psychology: list  # 心理机制
    report: str  # 分析报告


def build_pipeline():  # 构建分析流水线函数
    # 初始化语言模型
    llm = ChatOpenAI(base_url=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"],
                     model=env["LLM_MODEL"], temperature=0.3)  # 常规语言模型
    llm_json = ChatOpenAI(base_url=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"],
                          model=env["LLM_MODEL"], temperature=0.3,
                          model_kwargs={"response_format": {"type": "json_object"}})  # JSON输出语言模型



    # 初始化解析器
    seg_parser = PydanticOutputParser(pydantic_object=Segments)  # 段落解析器
    chain_seg = PromptTemplate.from_template(  # 构建段落分析链
        "你是叙事结构分析师。把文案拆成 2-4 个叙事段落。文案：{text}"
        "直接输出 JSON 对象本身：{format_instructions}"
    ) | llm_json | seg_parser  # 组合提示词、模型和解析器

    psy_parser = PydanticOutputParser(pydantic_object=Psychology)  # 心理机制解析器
    chain_psy = PromptTemplate.from_template(  # 构建心理分析链
        "你是爆款心理学分析师。分析每个段落的心理机制（损失厌恶、身份认同、社会比较等）。段落：{segments}"
        "直接输出 JSON 对象本身：{format_instructions}"
    ) | llm_json | psy_parser  # 组合提示词、模型和解析器

    chain_sum = PromptTemplate.from_template(  # 构建总结建议链
        "你是爆款视频总编。基于叙事结构和心理机制，输出 3 句话创作建议。叙事结构：{segments} 心理机制：{psychology}"
    ) | llm | StrOutputParser()  # 组合提示词、模型和字符串解析器



    # 定义状态处理函数
    def deconstruct(state):  # 文案拆解函数
        r = chain_seg.invoke({"text": state["text"],  # 调用段落分析链
                              "format_instructions": seg_parser.get_format_instructions()})
        return {"segments": r.segments}  # 返回段落结果

    def identify(state):  # 心理机制识别函数
        r = chain_psy.invoke({"segments": state["segments"],  # 调用心理分析链
                              "format_instructions": psy_parser.get_format_instructions()})
        return {"psychology": r.mechanisms}  # 返回心理机制结果

    def summarize(state):  # 总结建议函数
        return {"report": chain_sum.invoke({"segments": state["segments"],  # 调用总结建议链
                                            "psychology": state["psychology"]})}



    # 构建有向图工作流
    g = StateGraph(AnalysisState)  # 创建状态图
    g.add_node("deconstruct", deconstruct)  # 添加拆解节点
    g.add_node("identify", identify)  # 添加识别节点
    g.add_node("summarize", summarize)  # 添加总结节点
    g.add_edge("deconstruct", "identify")  # 连接拆解到识别
    g.add_edge("identify", "summarize")  # 连接识别到总结
    g.set_entry_point("deconstruct")  # 设置入口点
    g.set_finish_point("summarize")  # 设置结束点
    return g.compile()  # 编译并返回工作流


# ---------- 页面 ----------
st.set_page_config(page_title="爆款心法 · 分析器", page_icon="🔥")  # 设置页面配置
st.title("🔥 爆款心法 · 文案分析器")  # 设置页面标题
st.caption("输入视频文案 → AI 拆解叙事结构 + 识别心理机制 + 生成创作建议")  # 设置页面说明

text = st.text_area("📝 粘贴你的视频文案", height=140,  # 创建文本输入区域
                    placeholder="例：为什么你越努力越焦虑？三个原因拆穿你的情绪黑洞……")

if st.button("🚀 开始分析", type="primary"):  # 创建分析按钮
    if not text.strip():  # 检查输入是否为空
        st.warning("⚠️ 请先输入文案再分析")  # 显示警告
    else:
        with st.spinner("🤖 AI 分析中（拆结构 → 识心理 → 出建议）..."):  # 显示加载动画
            app = build_pipeline()  # 构建分析流水线
            result = app.invoke({"text": text, "segments": [], "psychology": [], "report": ""})  # 执行分析

        st.success("✅ 分析完成！")  # 显示成功消息

        st.subheader("📄 叙事结构拆解")  # 添加叙事结构标题
        for i, seg in enumerate(result["segments"], 1):  # 遍历段落
            st.write(f"**{i}.** {seg}")  # 显示段落

        st.subheader("🧠 心理机制识别")  # 添加心理机制标题
        mechs = result["psychology"]  # 获取心理机制列表
        cols = st.columns(min(len(mechs), 4))  # 创建最多4列
        for i, col in enumerate(cols):  # 遍历列
            if i < len(mechs):  # 确保不超出列表长度
                col.metric(label=f"机制 {i+1}", value=mechs[i])  # 显示机制指标

        st.subheader("💡 创作建议")  # 添加创作建议标题
        st.info(result["report"])  # 显示建议内容
