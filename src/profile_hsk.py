from pathlib import Path
import pandas as pd

# ==================== 1. 路径配置 ====================

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 合并数据文件路径
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "errors_merged.csv"


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


# ==================== 4. 主程序入口 ====================

def main():
    df = load_data(DATA_PATH)
    basic_profile(df)


if __name__ == "__main__":
    main()
