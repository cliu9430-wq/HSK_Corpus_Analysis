
from pathlib import Path
from collections import Counter
from datetime import datetime
import json
import re

import pandas as pd
from PIL import Image, UnidentifiedImageError


# ==================== 1. 路径配置 ====================

# 当前脚本位于 src/validate/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data/raw"
TEXT_DIR = RAW_DIR / "compositions_CJba"
SCAN_DIR = RAW_DIR / "scans_CJba"

ERRORS_PATH = RAW_DIR / "errors_CJba.csv"

COLLECTION_DIR = PROJECT_ROOT / "reports/collection"
STATUS_PATH = COLLECTION_DIR / "collection_status_CJba.csv"
MANIFEST_PATH = COLLECTION_DIR / "scan_manifest_CJba.json"

REPORT_DIR = PROJECT_ROOT / "reports/validation"
SUMMARY_PATH = REPORT_DIR / "validation_summary_CJba.json"
ISSUES_PATH = REPORT_DIR / "validation_issues_CJba.csv"

ID_COLUMN = "作文编码"

EXPECTED_ERRORS = 570
EXPECTED_COMPOSITIONS = 529

ISSUE_COLUMNS = [
    "composition_id",
    "category",
    "issue",
    "detail",
]

issues = []


# ==================== 2. 辅助函数 ====================

def add_issue(composition_id, category, issue, detail=""):
    issues.append({
        "composition_id": str(composition_id),
        "category": category,
        "issue": issue,
        "detail": str(detail),
    })


def valid_text(path):
    """检查文本是否存在、可解码且非空。"""
    if not path.is_file():
        return False, "文件不存在"

    try:
        text = path.read_text(encoding="utf-8")

        if not text.strip():
            return False, "文本为空"

        return True, ""

    except (OSError, UnicodeError) as e:
        return False, str(e)


def valid_gif(path):
    """检查GIF是否能够完整解码。"""
    if not path.is_file():
        return False, "文件不存在"

    try:
        with Image.open(path) as img:
            if img.format != "GIF":
                return False, "文件格式不是GIF"

            img.verify()

        return True, ""

    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
    ) as e:
        return False, str(e)


def safe_local_path(relative_path):
    """防止清单中的异常路径指向项目外部。"""
    if not isinstance(relative_path, str):
        raise ValueError("local_path不是字符串")

    path = Path(relative_path)

    if path.is_absolute():
        raise ValueError("不允许绝对路径")

    resolved = (PROJECT_ROOT / path).resolve()
    scan_root = SCAN_DIR.resolve()

    if not resolved.is_relative_to(scan_root):
        raise ValueError("图片路径超出扫描件目录")

    return resolved


# ==================== 3. 错句数据验收 ====================

def validate_errors():
    print("\n[1/4] 检查错句CSV")

    df = pd.read_csv(
        ERRORS_PATH,
        dtype={ID_COLUMN: "string"},
        encoding="utf-8-sig"
    )

    if ID_COLUMN not in df.columns:
        raise ValueError(f"缺少字段：{ID_COLUMN}")

    raw_ids = df[ID_COLUMN]
    ids = raw_ids.dropna().str.strip()

    invalid_ids = ids[
        ~ids.str.fullmatch(r"\d+")
    ]

    if raw_ids.isna().any():
        add_issue("", "errors", "作文编码缺失",
                  raw_ids.isna().sum())

    if (ids == "").any():
        add_issue("", "errors", "作文编码为空字符串",
                  (ids == "").sum())

    if not invalid_ids.empty:
        add_issue("", "errors", "作文编码格式异常",
                  invalid_ids.tolist()[:10])

    valid_ids = ids[
        ids.str.fullmatch(r"\d+")
    ]

    composition_ids = set(valid_ids.tolist())
    counts = Counter(valid_ids.tolist())

    if len(df) != EXPECTED_ERRORS:
        add_issue("", "errors", "错句数量异常",
                  f"实际{len(df)}，预期{EXPECTED_ERRORS}")

    if len(composition_ids) != EXPECTED_COMPOSITIONS:
        add_issue("", "errors", "作文数量异常",
                  f"实际{len(composition_ids)}，"
                  f"预期{EXPECTED_COMPOSITIONS}")

    frequency = Counter(counts.values())

    print(f"  错句记录：{len(df)}")
    print(f"  不同作文：{len(composition_ids)}")
    print(f"  编码出现次数分布：{dict(sorted(frequency.items()))}")

    return composition_ids, {
        "error_records": len(df),
        "unique_compositions": len(composition_ids),
        "id_frequency": dict(sorted(frequency.items())),
    }


# ==================== 4. 标注文本验收 ====================

def validate_texts(composition_ids):
    print("\n[2/4] 检查完整标注文本")

    success = 0
    lengths = []

    for composition_id in sorted(composition_ids):
        path = TEXT_DIR / f"{composition_id}.txt"

        valid, detail = valid_text(path)

        if valid:
            success += 1
            lengths.append(
                len(path.read_text(encoding="utf-8"))
            )
        else:
            add_issue(
                composition_id,
                "text",
                "完整标注异常",
                detail
            )

    actual_files = {
        p.stem
        for p in TEXT_DIR.glob("*.txt")
        if p.is_file()
    }

    extra_files = actual_files - composition_ids

    for composition_id in sorted(extra_files):
        add_issue(
            composition_id,
            "text",
            "存在多余的标注文件"
        )

    print(f"  预期文本：{len(composition_ids)}")
    print(f"  有效文本：{success}")
    print(f"  多余文本：{len(extra_files)}")

    return {
        "expected": len(composition_ids),
        "valid": success,
        "extra": len(extra_files),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


# ==================== 5. 扫描图片验收 ====================

def validate_scans(composition_ids, manifest):
    print("\n[3/4] 检查扫描图片")

    valid_compositions = 0
    expected_pages_total = 0
    valid_pages_total = 0
    expected_file_paths = set()

    for composition_id in sorted(composition_ids):
        images = manifest.get(composition_id)

        if not isinstance(images, list) or not images:
            add_issue(
                composition_id,
                "scan",
                "图片清单缺失或为空"
            )
            continue

        composition_valid = True
        page_numbers = []

        for image in images:
            if not isinstance(image, dict):
                add_issue(
                    composition_id, "scan",
                    "图片清单条目格式异常"
                )
                composition_valid = False
                continue

            page = image.get("page")
            relative_path = image.get("local_path")

            expected_pages_total += 1
            page_numbers.append(page)

            try:
                path = safe_local_path(relative_path)

                expected_path = (
                    SCAN_DIR / composition_id
                    / f"page_{page:02d}.gif"
                ).resolve()

                if (
                    not isinstance(page, int)
                    or isinstance(page, bool)
                    or page < 1
                    or path != expected_path
                ):
                    raise ValueError("页码或本地路径不匹配")

                if path in expected_file_paths:
                    raise ValueError("图片路径重复")

                expected_file_paths.add(path)

                valid, detail = valid_gif(path)

                if valid:
                    valid_pages_total += 1
                else:
                    composition_valid = False
                    add_issue(
                        composition_id,
                        "scan",
                        f"第{page}页图片异常",
                        detail
                    )

            except (ValueError, TypeError) as e:
                composition_valid = False
                add_issue(
                    composition_id,
                    "scan",
                    "图片清单路径异常",
                    e
                )

        expected_sequence = list(
            range(1, len(images) + 1)
        )

        if sorted(
            p for p in page_numbers if isinstance(p, int)
        ) != expected_sequence:
            composition_valid = False
            add_issue(
                composition_id,
                "scan",
                "扫描页码不连续或重复"
            )

        if composition_valid:
            valid_compositions += 1

    # 检查是否存在未登记的GIF
    actual_paths = {
        p.resolve()
        for p in SCAN_DIR.rglob("*.gif")
        if p.is_file()
    }

    extra_paths = actual_paths - expected_file_paths

    for path in sorted(extra_paths):
        add_issue(
            "",
            "scan",
            "存在未登记的GIF",
            path.relative_to(PROJECT_ROOT)
        )

    extra_ids = set(manifest) - composition_ids

    for composition_id in sorted(extra_ids):
        add_issue(
            composition_id,
            "scan",
            "图片清单包含非目标作文"
        )

    print(f"  图片完整的作文：{valid_compositions}")
    print(f"  清单登记图片：{expected_pages_total}")
    print(f"  有效图片：{valid_pages_total}")
    print(f"  未登记GIF：{len(extra_paths)}")

    return {
        "valid_compositions": valid_compositions,
        "expected_pages": expected_pages_total,
        "valid_pages": valid_pages_total,
        "extra_gifs": len(extra_paths),
    }


# ==================== 6. 状态文件验收 ====================

def validate_status(composition_ids, manifest):
    print("\n[4/4] 检查采集状态文件")

    df = pd.read_csv(
        STATUS_PATH,
        dtype={"composition_id": "string"},
        keep_default_na=False
    )

    required_columns = {
        "composition_id",
        "text_status",
        "scan_status",
        "expected_pages",
        "downloaded_pages",
        "collection_status",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError("采集状态CSV缺少必要字段")

    if df["composition_id"].duplicated().any():
        add_issue("", "status", "状态文件存在重复作文编码")

    status_ids = set(df["composition_id"])

    for composition_id in sorted(composition_ids - status_ids):
        add_issue(
            composition_id,
            "status",
            "缺少采集状态记录"
        )

    for composition_id in sorted(status_ids - composition_ids):
        add_issue(
            composition_id,
            "status",
            "存在非目标作文状态"
        )

    complete_count = 0

    for _, row in df.iterrows():
        composition_id = row["composition_id"]

        if composition_id not in composition_ids:
            continue

        if (
            row["text_status"] == "success"
            and row["scan_status"] == "success"
            and row["collection_status"] == "complete"
        ):
            complete_count += 1
        else:
            add_issue(
                composition_id,
                "status",
                "采集状态未完成",
                row["collection_status"]
            )

        images = manifest.get(composition_id, [])

        if isinstance(images, list):
            try:
                expected = int(row["expected_pages"])
                downloaded = int(row["downloaded_pages"])

                if (
                    expected != len(images)
                    or downloaded != len(images)
                ):
                    add_issue(
                        composition_id,
                        "status",
                        "扫描页数与清单不一致"
                    )

            except (ValueError, TypeError):
                add_issue(
                    composition_id,
                    "status",
                    "扫描页数不是有效整数"
                )

    print(f"  状态记录数：{len(df)}")
    print(f"  标记完成：{complete_count}")

    return {
        "status_records": len(df),
        "marked_complete": complete_count,
    }


# ==================== 7. 主程序 ====================

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    composition_ids, errors_result = validate_errors()

    texts_result = validate_texts(composition_ids)

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(manifest, dict):
        raise ValueError("扫描图片清单不是JSON对象")

    scans_result = validate_scans(
        composition_ids, manifest
    )

    status_result = validate_status(
        composition_ids, manifest
    )

    summary = {
        "validation_time": datetime.now().isoformat(
            timespec="seconds"
        ),
        "errors": errors_result,
        "texts": texts_result,
        "scans": scans_result,
        "collection_status": status_result,
        "issue_count": len(issues),
        "validation_passed": len(issues) == 0,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary, ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )

    pd.DataFrame(
        issues, columns=ISSUE_COLUMNS
    ).to_csv(
        ISSUES_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== 验收结果 ==========")
    print(f"发现问题：{len(issues)}")

    if summary["validation_passed"]:
        print("验收通过：未发现文件级异常")
    else:
        print("验收未通过：请查看问题明细CSV")

    print(
        f"汇总报告："
        f"{SUMMARY_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"问题明细："
        f"{ISSUES_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()

# AI辅助生成：GPT-6, 2026-11-17