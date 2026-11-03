from pathlib import Path
import pandas as pd

# 1. 项目路径
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "interim"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. 获取全部 CSV 文件
csv_files = sorted(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("未找到 CSV 文件")

# 3. 读取并检查数据
dfs = []

for file in csv_files:
    df = pd.read_csv(
        file,
        encoding="utf-8-sig",
        dtype=str
    )

    # 新增来源字段
    df["source_file"] = file.name
    df["source_row"] = range(2, len(df) + 2)

    print(f"{file.name}: {len(df)} 条")

    dfs.append(df)

# 4. 纵向合并
merged_df = pd.concat(dfs, ignore_index=True)

# 5. 新增内部记录 ID
merged_df.insert(
    0,
    "record_id",
    range(1, len(merged_df) + 1)
)

# 6. 基础检查
print("\n合并后总记录数：", len(merged_df))

# 排除新增的管理字段，检查原始数据是否完全重复
original_cols = [
    col for col in merged_df.columns
    if col not in ["record_id", "source_file", "source_row"]
]

duplicates = merged_df.duplicated(
    subset=original_cols
).sum()

print("原始字段完全重复的记录数：", duplicates)

# 7. 导出
output_file = OUTPUT_DIR / "errors_merged.csv"

merged_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n合并完成：{output_file}")

# AI辅助生成：GPT-6, 2026-11-3