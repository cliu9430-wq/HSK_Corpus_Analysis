from pathlib import Path
import pandas as pd

# ==================== 1. 路径配置 ====================

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 合并数据文件路径
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "errors_merged.csv"

# 数据画像报告目录
REPORT_DIR = PROJECT_ROOT / "reports" / "profiling"


# ==================== 2. 数据读取 ====================

def load_data(file_path: Path) -> pd.DataFrame:
    """读取 HSK 合并数据，保留原始记录。"""

    if not file_path.is_file():
        raise FileNotFoundError(f"数据文件不存在：{file_path}")

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        dtype={
            "作文编码": "string",
            "考试时间": "string",
            "证书": "string",
            "检索原句": "string",
            "source_file": "string",
        },
        low_memory=False,
    )

    print("数据读取成功！")
    print(f"文件路径：{file_path}")

    return df


# ==================== 3. 基础数据统计 ====================

def basic_profile(df: pd.DataFrame) -> None:
    """输出数据规模、字段信息和基本分布。"""

    print("\n========== 1. 数据基本规模 ==========")
    print(f"总记录数：{len(df):,}")
    print(f"总字段数：{len(df.columns)}")
    print(f"不同作文编码数量：{df['作文编码'].nunique():,}")

    print("\n========== 2. 字段信息 ==========")
    field_info = pd.DataFrame({
        "字段名称": df.columns,
        "数据类型": df.dtypes.astype(str).values,
    })
    print(field_info.to_string(index=False))

    print("\n========== 3. 前5条记录 ==========")
    print(df.head(5).to_string(index=False))

    print("\n========== 4. 证书等级分布 ==========")
    print(df["证书"].value_counts(dropna=False).to_string())

    print("\n========== 5. 国籍分布（前10名） ==========")
    print(df["国籍"].value_counts(dropna=False).head(10).to_string())

    print("\n========== 6. 考试时间范围 ==========")
    exam_time = df["考试时间"].dropna()
    if not exam_time.empty:
        print(f"最早考试时间：{exam_time.min()}")
        print(f"最晚考试时间：{exam_time.max()}")
    else:
        print("考试时间字段全部为空。")


# ==================== 4. 缺失值分析 ====================

def missing_profile(df: pd.DataFrame) -> None:
    """统计各字段的缺失数量与缺失比例，不修改原始数据。"""

    print("\n========== 7. 缺失值分析 ==========")

    # 统计每个字段的缺失数量
    missing_count = df.isna().sum()

    # 计算缺失比例（百分比）
    missing_rate = df.isna().mean() * 100

    # 构建缺失值统计表
    missing_stats = pd.DataFrame({
        "字段名称": df.columns,
        "缺失数量": missing_count.values,
        "缺失比例(%)": missing_rate.round(2).values,
    })

    # 按缺失数量降序排列
    missing_stats = missing_stats.sort_values(
        by="缺失数量",
        ascending=False,
    )

    print(missing_stats.to_string(index=False))

    # 统计至少存在一个缺失字段的记录数
    rows_with_missing = df.isna().any(axis=1).sum()

    print(f"\n存在缺失值的记录数：{rows_with_missing:,}")
    print(f"完整记录数：{len(df) - rows_with_missing:,}")

    # 保存统计报告
    output_path = REPORT_DIR / "missing_values.csv"

    missing_stats.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"缺失值报告已保存：{output_path}")




# ==================== 5. 重复记录与作文分布分析 ====================

def duplicate_profile(df: pd.DataFrame) -> None:
    """统计业务记录重复情况及作文对应的记录数分布。"""

    print("\n========== 8. 重复记录与作文分布分析 ==========")

    # 1. 排除合并时生成的技术字段
    technical_columns = [
        "record_id",
        "source_file",
        "source_row",
    ]

    business_columns = [
        col for col in df.columns
        if col not in technical_columns
    ]

    # 2. 业务字段完全重复
    # keep="first"：每组保留第一条，其余标记为重复
    business_duplicates = df.duplicated(
        subset=business_columns,
        keep="first",
    ).sum()

    # 3. 句子文本重复
    sentence_duplicates = df.duplicated(
        subset=["检索原句"],
        keep="first",
    ).sum()

    # 4. 统计每篇作文对应的记录数
    essay_counts = df["作文编码"].value_counts(
        dropna=False
    )

    # 5. 统计单条记录与多条记录的作文数量
    single_record_essays = (essay_counts == 1).sum()
    multiple_record_essays = (essay_counts > 1).sum()

    # 6. 输出重复记录检查结果
    print("\n--- 重复记录检查 ---")
    print(f"业务字段完全重复记录数：{business_duplicates:,}")
    print(f"句子文本重复记录数：{sentence_duplicates:,}")

    # 7. 输出作文记录分布
    print("\n--- 作文记录分布 ---")
    print(f"不同作文编码数量：{len(essay_counts):,}")
    print(f"仅包含1条记录的作文数：{single_record_essays:,}")
    print(f"包含多条记录的作文数：{multiple_record_essays:,}")

    print("\n每篇作文对应的记录数：")
    print(f"平均值：{essay_counts.mean():.2f}")
    print(f"最大值：{essay_counts.max():,}")
    print(f"最小值：{essay_counts.min():,}")

    # 8. 汇总统计结果
    duplicate_stats = pd.DataFrame({
        "统计指标": [
            "业务字段完全重复记录数",
            "句子文本重复记录数",
            "仅包含1条记录的作文数",
            "包含多条记录的作文数",
        ],
        "统计值": [
            business_duplicates,
            sentence_duplicates,
            single_record_essays,
            multiple_record_essays,
        ],
        "统计单位": [
            "记录",
            "记录",
            "作文",
            "作文",
        ],
    })

    # 9. 保存报告
    output_path = REPORT_DIR / "duplicate_summary.csv"

    duplicate_stats.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n重复记录与作文分布报告已保存：{output_path}")


# ==================== 6. 数据异常核验 ====================

def quality_check(df: pd.DataFrame) -> None:
    """核验评分缺失、考试时间异常和业务重复记录来源。"""

    print("\n========== 9. 数据异常核验 ==========")

    # ---------- 1. 评分字段缺失情况 ----------

    print("\n--- 1. 评分字段缺失核验 ---")

    oral_missing = df["口试分数"].isna()
    writing_missing = df["作文分数"].isna()

    # 两个字段同时缺失
    both_missing = oral_missing & writing_missing

    # 仅一个字段缺失
    oral_only = oral_missing & ~writing_missing
    writing_only = writing_missing & ~oral_missing

    # 任一评分字段缺失
    any_score_missing = oral_missing | writing_missing

    print(f"两个评分字段同时缺失：{both_missing.sum():,}")
    print(f"仅口试分数缺失：{oral_only.sum():,}")
    print(f"仅作文分数缺失：{writing_only.sum():,}")

    print(
        "评分缺失涉及的不同作文数："
        f"{df.loc[any_score_missing, '作文编码'].nunique():,}"
    )

    # ---------- 评分缺失与特殊考试时间的交集检查 ----------

    print("\n--- 评分缺失与199200时间编码的关联 ---")

    # 标记考试时间为199200的记录
    time_199200 = df["考试时间"] == "199200"

    # 同时满足两个条件
    print(
        f"199200且两个评分字段同时缺失："
        f"{(time_199200 & both_missing).sum():,}"
    )

    # 考试时间为199200，但评分没有同时缺失
    print(
        f"199200但评分未同时缺失："
        f"{(time_199200 & ~both_missing).sum():,}"
    )

    # 两个评分字段同时缺失，但考试时间不是199200
    print(
        f"评分同时缺失但时间不是199200："
        f"{(~time_199200 & both_missing).sum():,}"
    )

    # ---------- 2. 考试时间异常 ----------

    print("\n--- 2. 考试时间核验 ---")

    exam_time = df["考试时间"].astype("string")

    # 检查是否符合6位数字格式 YYYYMM
    valid_format = exam_time.str.fullmatch(
        r"\d{6}"
    ).fillna(False)

    # 提取月份并转换为数值
    month = pd.to_numeric(
        exam_time.str[-2:],
        errors="coerce",
    )

    # 识别异常月份
    month_zero = valid_format & (month == 0)
    invalid_month = valid_format & ~month.between(1, 12)

    # 识别格式异常
    invalid_format = ~valid_format

    print(f"月份为00的记录数：{month_zero.sum():,}")
    print(
        "其他非法月份的记录数："
        f"{(invalid_month & ~month_zero).sum():,}"
    )
    print(f"时间格式异常的记录数：{invalid_format.sum():,}")

    print("\n异常考试时间取值分布：")

    abnormal_time = (
        invalid_format | invalid_month
    )

    print(
        df.loc[abnormal_time, "考试时间"]
        .value_counts(dropna=False)
        .to_string()
    )

    # ---------- 3. 业务重复记录来源 ----------

    print("\n--- 3. 业务重复记录来源核验 ---")

    technical_columns = [
        "record_id",
        "source_file",
        "source_row",
    ]

    business_columns = [
        col for col in df.columns
        if col not in technical_columns
    ]

    # 标记所有属于重复组的记录，包括首次出现的记录
    duplicate_mask = df.duplicated(
        subset=business_columns,
        keep=False,
    )

    duplicate_rows = df.loc[duplicate_mask].copy()

    if duplicate_rows.empty:
        print("未发现业务字段完全重复记录。")
    else:
        # 为每个重复组编号
        duplicate_rows["duplicate_group"] = (
            duplicate_rows
            .groupby(
                business_columns,
                dropna=False,
                sort=False,
            )
            .ngroup()
        )

        # 统计每组涉及的不同源文件数量
        group_source_counts = (
            duplicate_rows
            .groupby("duplicate_group")["source_file"]
            .nunique()
        )

        # 统计重复组数量
        same_file_groups = (group_source_counts == 1).sum()
        cross_file_groups = (group_source_counts > 1).sum()

        print(f"涉及重复的记录总数：{len(duplicate_rows):,}")
        print(f"业务重复组数：{len(group_source_counts):,}")
        print(f"仅涉及同一文件的重复组数：{same_file_groups:,}")
        print(f"涉及多个文件的重复组数：{cross_file_groups:,}")

        print("\n重复记录来源文件分布：")
        print(
            duplicate_rows["source_file"]
            .value_counts()
            .to_string()
        )


# ==================== 7. 主程序入口 ====================

def main():
    # 创建报告目录
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取数据
    df = load_data(DATA_PATH)

    # 基础数据画像
    basic_profile(df)

    # 缺失值画像
    missing_profile(df)

    # 重复记录画像
    duplicate_profile(df)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)

    basic_profile(df)
    missing_profile(df)
    duplicate_profile(df)
    quality_check(df)


if __name__ == "__main__":
    main()

# AI辅助生成：GPT-6, 2026-11-9