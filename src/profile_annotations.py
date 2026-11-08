from pathlib import Path
import pandas as pd

# ==================== 1. 路径配置 ====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "errors_merged.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "profiling"


# ==================== 2. 数据读取 ====================

def load_data(file_path: Path) -> pd.DataFrame:
    """读取 HSK 合并数据中的记录编号与检索原句。"""

    if not file_path.is_file():
        raise FileNotFoundError(f"数据文件不存在：{file_path}")

    df = pd.read_csv(
        file_path,
        usecols=["record_id", "检索原句"],
        dtype={"检索原句": "string"},
        encoding="utf-8-sig",
        low_memory=False,
    )

    print("偏误标注数据读取成功！")
    print(f"文件路径：{file_path}")

    return df


# ==================== 3. 标注结构检查 ====================

def inspect_brackets(text: str) -> dict:
    """
    检查文本中的标注括号结构。

    支持花括号、方括号及其嵌套。
    仅检查括号配对，不判断偏误编码是否合法。
    """

    stack = []
    pairs = {"}": "{", "]": "["}

    curly_count = 0
    square_count = 0
    has_mismatch = False
    has_nested = False

    for char in text:
        if char in "{[":
            if stack:
                has_nested = True

            stack.append(char)

        elif char in "}]":
            if not stack or stack[-1] != pairs[char]:
                has_mismatch = True
                continue

            opening = stack.pop()

            if opening == "{":
                curly_count += 1
            else:
                square_count += 1

    if stack:
        has_mismatch = True

    return {
        "has_annotation": curly_count + square_count > 0,
        "has_curly": curly_count > 0,
        "has_square": square_count > 0,
        "has_mixed": curly_count > 0 and square_count > 0,
        "has_nested": has_nested,
        "has_mismatch": has_mismatch,
        "bracket_pair_count": curly_count + square_count,
    }


# ==================== 4. 标注覆盖率分析 ====================

def annotation_coverage(df: pd.DataFrame) -> None:
    """统计显式标注的覆盖情况及括号结构异常。"""

    print("\n========== 1. 偏误标注覆盖情况 ==========")

    # 避免空值影响结构检查
    texts = df["检索原句"].fillna("")

    # 逐条检查括号结构
    results = pd.DataFrame(
        [inspect_brackets(text) for text in texts],
        index=df.index,
    )

    total = len(df)

    # 汇总各项统计
    metrics = {
        "总记录数": total,
        "含完整括号结构的记录数": int(
            results["has_annotation"].sum()
        ),
        "无完整括号结构的记录数": int(
            (~results["has_annotation"]).sum()
        ),
        "含花括号标注的记录数": int(
            results["has_curly"].sum()
        ),
        "含方括号标注的记录数": int(
            results["has_square"].sum()
        ),
        "同时包含两类括号的记录数": int(
            results["has_mixed"].sum()
        ),
        "存在嵌套括号的记录数": int(
            results["has_nested"].sum()
        ),
        "疑似括号不匹配的记录数": int(
            results["has_mismatch"].sum()
        ),
    }

    # 输出统计结果
    for name, count in metrics.items():
        rate = count / total * 100 if total else 0

        if name == "总记录数":
            print(f"{name}：{count:,}")
        else:
            print(f"{name}：{count:,}（{rate:.2f}%）")

    # 保存汇总报告
    summary = pd.DataFrame({
        "统计指标": list(metrics.keys()),
        "记录数": list(metrics.values()),
    })

    summary["占总记录比例(%)"] = (
        summary["记录数"] / total * 100
        if total else 0.0
    ).round(2)

    output_path = REPORT_DIR / "annotation_coverage.csv"

    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n标注覆盖报告已保存：{output_path}")



# ==================== 5. 篇章标注边界分析 ====================

def analyze_cp_mismatch(df: pd.DataFrame) -> None:
    """
    分析疑似括号不匹配记录与篇章标注边界的关系。

    HSK 语料库使用 {CP ... P} 表示篇章层面的标注，
    该结构可能跨越多个句子，因此单条记录中可能只出现
    起始标记 {CP 或结束标记 P}。

    本函数仅统计两种标记在疑似异常记录中的出现情况，
    不判断篇章标注是否完整或合法。
    """

    # ---------- 1. 获取疑似括号不匹配的记录 ----------

    # 将空值填充为空字符串，避免影响文本检查
    texts = df["检索原句"].fillna("")

    # 复用已有的括号结构检查函数
    results = pd.DataFrame(
        [inspect_brackets(text) for text in texts],
        index=df.index,
    )

    # 筛选此前被标记为括号不匹配的记录
    mismatch_texts = texts[results["has_mismatch"]]

    # ---------- 2. 检测篇章标注的起始与结束标记 ----------

    # 检查是否包含篇章标注起始标记 {CP
    has_cp_start = mismatch_texts.str.contains(
        "{CP",
        regex=False,
    )

    # 检查是否包含篇章标注结束标记 P}
    has_cp_end = mismatch_texts.str.contains(
        "P}",
        regex=False,
    )

    # ---------- 3. 统计四种互斥情况 ----------

    # 仅包含起始标记
    start_only = (has_cp_start & ~has_cp_end).sum()

    # 仅包含结束标记
    end_only = (~has_cp_start & has_cp_end).sum()

    # 同时包含起始和结束标记
    both = (has_cp_start & has_cp_end).sum()

    # 两种标记均不包含
    neither = (~has_cp_start & ~has_cp_end).sum()

    # ---------- 4. 输出篇章标注边界统计 ----------

    # 包含至少一种 CP 边界标记的疑似异常记录
    cp_related = start_only + end_only + both

    print("\n========== 2. 篇章标注边界分析 ==========")

    print(f"疑似括号不匹配记录数：{len(mismatch_texts):,}")
    print(f"包含 CP 边界标记：{cp_related:,}")
    print(f"不包含 CP 边界标记：{neither:,}")



# ==================== 6. 主程序入口 ====================


# ==================== 6. 主程序入口 ====================

def main():
    # ---------- 1. 创建报告目录 ----------
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 2. 读取 HSK 合并数据 ----------
    df = load_data(DATA_PATH)

    # ---------- 3. 阶段 1：标注覆盖情况 ----------
    annotation_coverage(df)

    # ---------- 4. 篇章标注边界分析 ----------
    analyze_cp_mismatch(df)


if __name__ == "__main__":
    main()
