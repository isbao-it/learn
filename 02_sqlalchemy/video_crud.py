"""
阶段2练习：SQLAlchemy 数据库操作
场景：把爆款视频案例存进 SQLite，实现增删改查
对应项目：任务卡 02（storage 数据层）的案例表 + CRUD
"""

from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

# 1️⃣ 连接数据库（SQLite 就是一个本地文件，无需安装服务器）
engine = create_engine("sqlite:///cases.db", echo=False)


# 2️⃣ 定义表结构（和阶段1的 Pydantic 模型长得像，但这次是"数据库的表"）
# 定义一个基础模型类，继承自DeclarativeBase
# 这个类将作为所有其他模型类的基类，提供SQLAlchemy ORM的基础功能
class Base(DeclarativeBase):
    pass


class VideoCase(Base):
    __tablename__ = "video_cases"          # 表名

    id: Mapped[int] = mapped_column(primary_key=True)   # 主键，自动递增
    title: Mapped[str] = mapped_column(nullable=False)
    content: Mapped[str]
    platform: Mapped[str]                  # 存 "bilibili" / "douyin"
    author: Mapped[str]
    views: Mapped[int] = mapped_column(default=0)
    likes: Mapped[int] = mapped_column(default=0)

    def __repr__(self):                    # 打印时显示什么
        return f"<VideoCase {self.id} {self.title} ({self.views}播放)>"


# 3️⃣ 建表（执行一次，生成 cases.db 文件）
Base.metadata.create_all(engine)


# ============ 增删改查演示 ============
if __name__ == "__main__":
    # --- 增：插入两条案例 ---
    with Session(engine) as session:
        case1 = VideoCase(
            title="为什么你越努力越焦虑？",
            content="三个原因拆穿你的焦虑本质……",
            platform="bilibili", author="某UP主",
            views=1200000, likes=56000,
        )
        case2 = VideoCase(
            title="月薪3千和3万的区别",
            content="不是努力，是选择……",
            platform="douyin", author="职场说",
            views=3500000, likes=210000,
        )
        session.add_all([case1, case2])    # 加入待写队列
        session.commit()                   # 提交 → 真正写入数据库
        print("✅ 增：插入 2 条案例成功")

    # --- 查：读出所有案例 ---
    with Session(engine) as session:
        cases = session.scalars(select(VideoCase)).all()
        print("\n📖 查：全部案例")
        for c in cases:
            print(f"   {c}")

    # --- 查：条件查询 + 排序（播放量最高的） ---
    with Session(engine) as session:
        top = session.scalars(
            select(VideoCase).order_by(VideoCase.views.desc()).limit(1)
        ).one()
        print(f"\n🏆 查：播放量最高的案例 → {top}")

    # --- 改：给第一条案例点赞量 +1000 ---
    with Session(engine) as session:
        case = session.scalars(select(VideoCase).limit(1)).one()   # 取第一条
        case.likes += 1000                 # 改内存里的对象
        session.commit()                   # 提交 → 写回数据库
        print(f"\n✏️  改：案例1 点赞量更新为 {case.likes}")

    # --- 删：删除播放量最低的案例 ---
    with Session(engine) as session:
        lowest = session.scalars(
            select(VideoCase).order_by(VideoCase.views).limit(1)
        ).one()
        print(f"\n🗑️  删：删除 {lowest.title}")
        session.delete(lowest)
        session.commit()

    # --- 最终确认 ---
    with Session(engine) as session:
        remain = session.scalars(select(VideoCase)).all()
        print(f"\n📊 数据库现有 {len(remain)} 条案例")
        for c in remain:
            print(f"   {c}")
