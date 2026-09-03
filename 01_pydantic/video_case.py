"""
阶段1练习：Pydantic 数据模型
场景：定义爆款心法项目里的"视频案例"模型
对应项目：任务卡 02（storage 数据层）的案例表结构
"""

from datetime import date
from enum import Enum
from pydantic import BaseModel, Field, field_validator


# 1️⃣ 枚举：平台只允许这两个值
class Platform(str, Enum):
    BILIBILI = "bilibili"
    DOUYIN = "douyin"


# 2️⃣ 定义视频案例模型（就是数据的"合同"）
class VideoCase(BaseModel):
    id: int
    title: str = Field(min_length=1, max_length=100)   # 标题 1~100 字符
    content: str                                       # 视频文案
    platform: Platform                                 # 平台（枚举校验）
    author: str = Field(min_length=1)
    views: int = Field(ge=0)                           # 播放量 >= 0
    likes: int = Field(ge=0)                           # 点赞量 >= 0
    publish_date: date                                 # 发布日期
    tags: list[str] = []                               # 标签列表，默认空

    # 3️⃣ 自定义校验：文案不能是空白
    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("文案不能是空白")
        return v.strip()


# ============ 运行演示 ============
if __name__ == "__main__":
    print("=" * 50)
    print("✅ 演示1：合法数据 → 正常通过")
    case = VideoCase(
        id=1,
        title="为什么你越努力越焦虑？",
        content="三个原因拆穿你的焦虑本质……",
        platform="bilibili",
        author="某UP主",
        views=1200000,
        likes=56000,
        publish_date="2026-07-01",   # 改成中文格式 → 看 Pydantic 怎么拒绝
        tags=["焦虑", "心理", "爆款"],
    )
    print("  标题:", case.title)
    print("  平台:", case.platform)
    print("  播放量:", case.views)
    print("  发布日期类型:", type(case.publish_date).__name__)

    print("\n✅ 演示2：转成 JSON（给 API 用）")
    print(case.model_dump_json())

    print("\n❌ 演示3：非法数据 → 自动拦截")
    try:
        VideoCase(
            id=2,
            title="测试",
            content="   ",          # 空白文案 → 触发自定义校验
            platform="youtube",      # 不在枚举里 → 报错
            views=-100,              # 负数播放量 → 报错
            likes=10,
            publish_date="2026-07-01",
        )
    except Exception as e:
        print("  拦截成功！错误信息：")
        for err in e.errors():
            print(f"    - 字段 [{err['loc'][0]}] 错误: {err['msg']}")
