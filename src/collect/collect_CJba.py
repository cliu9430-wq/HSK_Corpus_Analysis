from pathlib import Path
from urllib.parse import urljoin, urlparse
from datetime import datetime
import csv
import json
import os
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ==================== 1. 基础配置 ====================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data/raw/errors_CJba.csv"
TEXT_DIR = PROJECT_ROOT / "data/raw/compositions_CJba"
SCAN_DIR = PROJECT_ROOT / "data/raw/scans_CJba"
REPORT_DIR = PROJECT_ROOT / "reports/collection"

STATUS_PATH = REPORT_DIR / "collection_status_CJba.csv"
MANIFEST_PATH = REPORT_DIR / "scan_manifest_CJba.json"

ID_COLUMN = "作文编码"

BASE_URL = "https://hsk.blcu.edu.cn/"
TEXT_API = BASE_URL + "api/v1/resource/text/"
DOC_API = BASE_URL + "api/v1/resource/doc/"

REQUEST_INTERVAL = 1.5
TIMEOUT = 20

# 先设为5进行正式版验证，确认后改为None
MAX_COMPOSITIONS = None

STATUS_FIELDS = [
    "composition_id",
    "text_status",
    "scan_status",
    "expected_pages",
    "downloaded_pages",
    "collection_status",
    "error",
    "updated_at",
]


# ==================== 2. 网络请求 ====================

session = requests.Session()

retry = Retry(
    total=2,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET"],
)

session.mount(
    "https://",
    HTTPAdapter(max_retries=retry)
)

session.headers.update({
    "User-Agent": "HSK-Corpus-Research/1.0"
})

last_request_time = 0.0


class AccessRestricted(Exception):
    """访问被限制时停止整个采集任务。"""


def get_response(url):
    global last_request_time

    elapsed = time.monotonic() - last_request_time
    if elapsed < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - elapsed)

    last_request_time = time.monotonic()

    response = session.get(url, timeout=TIMEOUT)

    if response.status_code in (401, 403, 429):
        raise AccessRestricted(
            f"HTTP {response.status_code}，停止采集：{url}"
        )

    response.raise_for_status()
    return response


def get_api_data(url):
    result = get_response(url).json()

    if not isinstance(result, dict):
        raise ValueError("接口返回的不是JSON对象")

    if result.get("code") != 0:
        raise ValueError(f"API异常：{result.get('msg')}")

    return result.get("data")


# ==================== 3. 文件操作 ====================

def atomic_write_bytes(path, content):
    """先写临时文件，再替换正式文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")

    try:
        with open(temp_path, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_write_json(path, data):
    content = json.dumps(
        data, ensure_ascii=False, indent=2
    ).encode("utf-8")

    atomic_write_bytes(path, content)


def valid_text(path):
    if not path.is_file():
        return False

    try:
        return bool(path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeError):
        return False


def valid_gif(path):
    if not path.is_file():
        return False

    try:
        if path.stat().st_size <= 6:
            return False

        with path.open("rb") as f:
            return f.read(6) in (b"GIF87a", b"GIF89a")
    except OSError:
        return False


# ==================== 4. 读取任务 ====================

def load_composition_ids():
    df = pd.read_csv(
        CSV_PATH,
        dtype={ID_COLUMN: "string"},
        encoding="utf-8-sig"
    )

    if ID_COLUMN not in df.columns:
        raise ValueError(f"缺少字段：{ID_COLUMN}")

    if df[ID_COLUMN].isna().any():
        raise ValueError("作文编码存在缺失值")

    ids = (
        df[ID_COLUMN]
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    if any(not x.isdecimal() for x in ids):
        raise ValueError("存在格式异常的作文编码")

    print(f"错句记录数：{len(df)}")
    print(f"不同作文数：{len(ids)}")

    if len(df) != 570 or len(ids) != 529:
        raise ValueError(
            "数据规模与此前核验结果不一致，请先检查CSV"
        )

    return ids


# ==================== 5. 完整标注采集 ====================

def collect_text(composition_id):
    output_path = TEXT_DIR / f"{composition_id}.txt"

    if valid_text(output_path):
        return "skipped"

    text = get_api_data(TEXT_API + composition_id)

    if not isinstance(text, str) or not text.strip():
        raise ValueError("完整标注为空或格式异常")

    atomic_write_bytes(
        output_path,
        text.encode("utf-8")
    )

    return "downloaded"


# ==================== 6. 扫描图片采集 ====================

def get_image_url(relative_path):
    if not isinstance(relative_path, str):
        raise ValueError("图片路径不是字符串")

    url = urljoin(BASE_URL, relative_path)
    parsed = urlparse(url)

    if (
        parsed.scheme != "https"
        or parsed.netloc != "hsk.blcu.edu.cn"
        or not parsed.path.startswith("/resource/zt/")
        or not parsed.path.lower().endswith(".gif")
    ):
        raise ValueError(f"非法图片路径：{relative_path}")

    return url


def collect_scans(composition_id):
    paths = get_api_data(DOC_API + composition_id)

    if not isinstance(paths, list) or not paths:
        raise ValueError("扫描图片路径列表异常")

    output_dir = SCAN_DIR / composition_id
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []

    for page, relative_path in enumerate(paths, start=1):
        url = get_image_url(relative_path)
        output_path = output_dir / f"page_{page:02d}.gif"

        if valid_gif(output_path):
            action = "skipped"
        else:
            response = get_response(url)

            if not response.content.startswith(
                (b"GIF87a", b"GIF89a")
            ):
                raise ValueError(
                    f"第{page}页不是有效GIF"
                )

            atomic_write_bytes(
                output_path,
                response.content
            )
            action = "downloaded"

        images.append({
            "page": page,
            "source_path": relative_path,
            "local_path": str(
                output_path.relative_to(PROJECT_ROOT)
            ),
            "action": action,
        })

    valid_count = sum(
        valid_gif(output_dir / f"page_{p:02d}.gif")
        for p in range(1, len(paths) + 1)
    )

    if valid_count != len(paths):
        raise ValueError("扫描图片数量核验失败")

    return len(paths), valid_count, images


# ==================== 7. 采集状态管理 ====================

def save_status(records):
    STATUS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    temp_path = STATUS_PATH.with_name(
        STATUS_PATH.name + ".tmp"
    )

    try:
        with temp_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=STATUS_FIELDS
            )
            writer.writeheader()
            writer.writerows(records)

        temp_path.replace(STATUS_PATH)
    finally:
        temp_path.unlink(missing_ok=True)


def load_status():
    if not STATUS_PATH.exists():
        return {}

    df = pd.read_csv(
        STATUS_PATH,
        dtype={"composition_id": "string"},
        keep_default_na=False
    )

    return {
        row["composition_id"]: row.to_dict()
        for _, row in df.iterrows()
    }


def load_manifest():
    if not MANIFEST_PATH.exists():
        return {}

    return json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )


def scans_complete(composition_id, images):
    if not isinstance(images, list) or not images:
        return False

    for image in images:
        path = PROJECT_ROOT / image["local_path"]
        if not valid_gif(path):
            return False

    return True


# ==================== 8. 主程序 ====================

def main():
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_ids = load_composition_ids()

    ids = (
        all_ids[:MAX_COMPOSITIONS]
        if MAX_COMPOSITIONS is not None
        else all_ids
    )

    previous_status = load_status()
    manifest = load_manifest()

    records_by_id = dict(previous_status)

    try:
        for index, composition_id in enumerate(ids, start=1):
            print(
                f"\n[{index}/{len(ids)}] "
                f"作文编码：{composition_id}"
            )

            text_path = TEXT_DIR / f"{composition_id}.txt"

            previous_images = manifest.get(
                composition_id, []
            )

            if (
                valid_text(text_path)
                and scans_complete(
                    composition_id, previous_images
                )
            ):
                print("  已完成，跳过")
                continue

            record = {
                "composition_id": composition_id,
                "text_status": "failed",
                "scan_status": "failed",
                "expected_pages": "",
                "downloaded_pages": "",
                "collection_status": "incomplete",
                "error": "",
                "updated_at": "",
            }

            errors = []

            # 采集完整标注
            try:
                action = collect_text(composition_id)
                record["text_status"] = "success"
                print(f"  标注文本：{action}")

            except AccessRestricted:
                raise

            except (requests.RequestException,
                    ValueError, OSError) as e:
                errors.append(f"text: {e}")
                print(f"  标注文本失败：{e}")

            # 采集扫描图片
            try:
                expected, downloaded, images = (
                    collect_scans(composition_id)
                )

                record["scan_status"] = "success"
                record["expected_pages"] = expected
                record["downloaded_pages"] = downloaded

                manifest[composition_id] = images
                atomic_write_json(
                    MANIFEST_PATH, manifest
                )

                print(f"  扫描图片：{downloaded}/{expected}页")

            except AccessRestricted:
                raise

            except (requests.RequestException,
                    ValueError, OSError) as e:
                errors.append(f"scans: {e}")
                print(f"  扫描图片失败：{e}")

            if (
                record["text_status"] == "success"
                and record["scan_status"] == "success"
            ):
                record["collection_status"] = "complete"

            record["error"] = " | ".join(errors)
            record["updated_at"] = (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            )

            records_by_id[composition_id] = record
            save_status(list(records_by_id.values()))

    except AccessRestricted as e:
        print(f"\n访问受限，已停止：{e}")

    except KeyboardInterrupt:
        print("\n用户中断采集，已有文件保留")

    finally:
        save_status(list(records_by_id.values()))
        atomic_write_json(MANIFEST_PATH, manifest)
        session.close()

        completed = sum(
            valid_text(TEXT_DIR / f"{cid}.txt")
            and scans_complete(
                cid, manifest.get(cid, [])
            )
            for cid in ids
        )

        print("\n========== 采集汇总 ==========")
        print(f"本次任务作文数：{len(ids)}")
        print(f"已完整采集：{completed}")
        print(f"尚未完成：{len(ids) - completed}")
        print(f"状态文件：{STATUS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

# AI辅助生成：GPT-6, 2026-11-17