
from pathlib import Path
from collections import Counter
from datetime import datetime
import json

import pandas as pd


# ==================== 1. 路径配置 ====================

# 脚本位于 HSK_Corpus_Analysis/src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data/raw"

ERRORS_PATH = RAW_DIR / "errors_CJba.csv"
TEXT_DIR = RAW_DIR / "compositions_CJba"
SCAN_DIR = RAW_DIR / "scans_CJba"

MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports/collection/scan_manifest_CJba.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data/processed"
REPORT_DIR = PROJECT_ROOT / "reports/integration"

ERRORS_OUTPUT = (
    OUTPUT_DIR / "errors_integrated_CJba.csv"
)

COMPOSITIONS_OUTPUT = (
    OUTPUT_DIR / "compositions_integrated_CJba.csv"
)

REPORT_OUTPUT = (
    REPORT_DIR / "integration_report_CJba.json"
)

ID_COLUMN = "作文编码"

EXPECTED_ERRORS = 570
EXPECTED_COMPOSITIONS = 529


# ==================== 2. 读取原始数据 ====================

def load_errors():
    """读取错句CSV，检查记录数及作文编码。"""

    df = pd.read_csv(
        ERRORS_PATH,
        dtype={ID_COLUMN: "string"},
        encoding="utf-8-sig"
    )

    if ID_COLUMN not in df.columns:
        raise ValueError(
            f"CSV缺少字段：{ID_COLUMN}"
        )

    if len(df) != EXPECTED_ERRORS:
        raise ValueError(
            f"错句数量异常：实际{len(df)}，"
            f"预期{EXPECTED_ERRORS}"
        )

    if df[ID_COLUMN].isna().any():
        raise ValueError("作文编码存在缺失值")

    df[ID_COLUMN] = df[ID_COLUMN].str.strip()

    if not df[ID_COLUMN].str.fullmatch(r"\d+").all():
        raise ValueError("存在格式异常的作文编码")

    unique_count = df[ID_COLUMN].nunique()

    if unique_count != EXPECTED_COMPOSITIONS:
        raise ValueError(
            f"作文数量异常：实际{unique_count}，"
            f"预期{EXPECTED_COMPOSITIONS}"
        )

    return df


def load_manifest():
    """读取扫描图片清单。"""

    with MANIFEST_PATH.open(
        "r", encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ValueError("扫描图片清单格式异常")

    return manifest


# ==================== 3. 构建作文表 ====================

def build_composition_table(composition_ids, manifest):
    """
    每篇作文生成一条记录。

    包含：
    - 作文编码
    - 完整标注文本
    - 标注文本路径
    - 扫描页数
    - 扫描图片路径
    """

    records = []

    for composition_id in sorted(composition_ids):

        # ---------- 读取完整标注 ----------

        text_path = (
            TEXT_DIR / f"{composition_id}.txt"
        )

        if not text_path.is_file():
            raise FileNotFoundError(
                f"缺少完整标注：{composition_id}"
            )

        full_text = text_path.read_text(
            encoding="utf-8"
        )

        if not full_text.strip():
            raise ValueError(
                f"完整标注为空：{composition_id}"
            )

        # ---------- 获取扫描图片清单 ----------

        images = manifest.get(composition_id)

        if not isinstance(images, list) or not images:
            raise ValueError(
                f"扫描图片清单缺失：{composition_id}"
            )

        scan_paths = []
        page_numbers = []

        for image in images:

            if not isinstance(image, dict):
                raise ValueError(
                    f"图片清单条目异常：{composition_id}"
                )

            page = image.get("page")
            relative_path = image.get("local_path")

            if (
                not isinstance(page, int)
                or isinstance(page, bool)
                or page < 1
                or not isinstance(relative_path, str)
            ):
                raise ValueError(
                    f"图片页码或路径异常：{composition_id}"
                )

            # ---------- 核验图片路径 ----------

            actual_path = (
                PROJECT_ROOT / relative_path
            ).resolve()

            expected_path = (
                SCAN_DIR
                / composition_id
                / f"page_{page:02d}.gif"
            ).resolve()

            if actual_path != expected_path:
                raise ValueError(
                    f"图片路径不匹配："
                    f"{composition_id}，第{page}页"
                )

            if not actual_path.is_file():
                raise FileNotFoundError(
                    f"扫描图片不存在：{actual_path}"
                )

            scan_paths.append(
                str(
                    actual_path.relative_to(
                        PROJECT_ROOT.resolve()
                    )
                )
            )

            page_numbers.append(page)

        # ---------- 核验页码 ----------

        if sorted(page_numbers) != list(
            range(1, len(images) + 1)
        ):
            raise ValueError(
                f"扫描图片页码不连续或重复："
                f"{composition_id}"
            )

        # 按页码排序，确保路径顺序正确
        sorted_paths = [
            path
            for _, path in sorted(
                zip(page_numbers, scan_paths)
            )
        ]

        records.append({
            ID_COLUMN: composition_id,
            "完整标注文本": full_text,
            "标注文本路径": str(
                text_path.relative_to(PROJECT_ROOT)
            ),
            "扫描页数": len(sorted_paths),
            "扫描图片路径": json.dumps(
                sorted_paths,
                ensure_ascii=False
            ),
        })

    compositions_df = pd.DataFrame(records)

    if compositions_df[ID_COLUMN].duplicated().any():
        raise ValueError("作文表存在重复作文编码")

    return compositions_df


# ==================== 4. 整合错句表 ====================

def integrate_errors(errors_df, compositions_df):
    """
    保留原始570条错句及其顺序，
    添加作文标注和扫描图片的关联字段。
    """

    new_columns = [
        "标注文本路径",
        "扫描页数",
        "扫描图片路径",
    ]

    conflicts = (
        set(new_columns)
        & set(errors_df.columns)
    )

    if conflicts:
        raise ValueError(
            f"原始CSV与新增字段重名：{conflicts}"
        )

    # ---------- 保存原始行号 ----------

    order_column = "__original_row_order__"

    if order_column in errors_df.columns:
        raise ValueError("临时行号字段名称冲突")

    errors_with_order = errors_df.copy()

    errors_with_order[order_column] = range(
        len(errors_with_order)
    )

    # ---------- 提取作文关联字段 ----------

    metadata = compositions_df[
        [ID_COLUMN] + new_columns
    ].copy()

    # 统一作文编码的数据类型
    metadata[ID_COLUMN] = (
        metadata[ID_COLUMN].astype("string")
    )

    # ---------- 多对一合并 ----------

    integrated = errors_with_order.merge(
        metadata,
        on=ID_COLUMN,
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False
    )

    # ---------- 检查关联完整性 ----------

    if not integrated["_merge"].eq("both").all():
        unmatched = integrated.loc[
            integrated["_merge"] != "both",
            ID_COLUMN
        ].tolist()

        raise ValueError(
            f"部分错句未关联到作文：{unmatched}"
        )

    if len(integrated) != len(errors_df):
        raise ValueError(
            "整合后错句记录数发生变化"
        )

    # ---------- 恢复原始行顺序 ----------

    integrated = (
        integrated
        .sort_values(order_column)
        .drop(
            columns=[order_column, "_merge"]
        )
        .reset_index(drop=True)
    )

    # ---------- 检查编码顺序 ----------

    original_ids = (
        errors_df[ID_COLUMN]
        .astype(str)
        .tolist()
    )

    integrated_ids = (
        integrated[ID_COLUMN]
        .astype(str)
        .tolist()
    )

    if integrated_ids != original_ids:
        raise ValueError(
            "整合后作文编码顺序发生变化"
        )

    # ---------- 检查原始字段是否保留 ----------

    original_columns = errors_df.columns.tolist()

    if integrated.columns[:len(original_columns)].tolist() != original_columns:
        raise ValueError(
            "整合后原始字段顺序发生变化"
        )

    for column in original_columns:
        if not integrated[column].equals(
            errors_df[column].reset_index(drop=True)
        ):
            raise ValueError(
                f"整合后原始字段内容发生变化：{column}"
            )

    return integrated


# ==================== 5. 输出整合结果 ====================

def save_results(
    errors_df,
    compositions_df,
    integrated_df
):
    """保存整合数据和统计报告。"""

    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    # ---------- 保存作文表 ----------

    compositions_df.to_csv(
        COMPOSITIONS_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    # ---------- 保存错句表 ----------

    integrated_df.to_csv(
        ERRORS_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    # ---------- 统计作文编码出现次数 ----------

    frequency = Counter(
        errors_df[ID_COLUMN]
        .value_counts()
        .tolist()
    )

    # ---------- 生成报告 ----------

    report = {
        "integration_time": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
        "error_records": len(integrated_df),
        "unique_compositions": len(compositions_df),
        "scan_pages_total": int(
            compositions_df["扫描页数"].sum()
        ),
        "error_frequency": {
            str(k): int(v)
            for k, v in sorted(frequency.items())
        },
        "output_files": [
            str(
                ERRORS_OUTPUT.relative_to(
                    PROJECT_ROOT
                )
            ),
            str(
                COMPOSITIONS_OUTPUT.relative_to(
                    PROJECT_ROOT
                )
            ),
        ],
        "integration_passed": True,
    }

    REPORT_OUTPUT.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return report


# ==================== 6. 主程序 ====================

def main():

    print("========== CJba数据整合 ==========")

    # 读取原始错句
    errors_df = load_errors()

    # 读取扫描图片清单
    manifest = load_manifest()

    composition_ids = set(
        errors_df[ID_COLUMN].tolist()
    )

    print(f"原始错句：{len(errors_df)}")
    print(f"不同作文：{len(composition_ids)}")

    # ---------- 检查清单编码 ----------

    manifest_ids = set(manifest.keys())

    if manifest_ids != composition_ids:
        missing = composition_ids - manifest_ids
        extra = manifest_ids - composition_ids

        raise ValueError(
            f"图片清单编码不一致："
            f"缺失{len(missing)}，"
            f"多余{len(extra)}"
        )

    # ---------- 构建作文表 ----------

    compositions_df = build_composition_table(
        composition_ids,
        manifest
    )

    print(
        f"完整作文：{len(compositions_df)}"
    )

    # ---------- 整合错句表 ----------

    integrated_df = integrate_errors(
        errors_df,
        compositions_df
    )

    print(
        f"关联错句：{len(integrated_df)}"
    )

    # ---------- 保存结果 ----------

    report = save_results(
        errors_df,
        compositions_df,
        integrated_df
    )

    # ---------- 输出汇总 ----------

    print("\n========== 整合结果 ==========")

    print(
        f"错句记录：{report['error_records']}"
    )

    print(
        f"作文记录：{report['unique_compositions']}"
    )

    print(
        f"扫描图片：{report['scan_pages_total']}张"
    )

    print(
        f"编码频次分布：{report['error_frequency']}"
    )

    print("整合完成，原始文件未修改")

    print(
        f"错句表："
        f"{ERRORS_OUTPUT.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"作文表："
        f"{COMPOSITIONS_OUTPUT.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"整合报告："
        f"{REPORT_OUTPUT.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
