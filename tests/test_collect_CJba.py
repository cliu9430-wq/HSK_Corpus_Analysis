
from pathlib import Path
from urllib.parse import urljoin, urlparse
import json
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ==================== 1. 配置 ====================

# 脚本位置：项目根目录/src/test/test_collect_CJba.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "data/raw/errors_CJba.csv"
TEXT_DIR = PROJECT_ROOT / "data/raw/compositions"
SCAN_DIR = PROJECT_ROOT / "data/raw/scans"
REPORT_PATH = PROJECT_ROOT / "reports/collection/test_CJba.json"

ID_COLUMN = "作文编码"
TEST_LIMIT = 3

BASE_URL = "https://hsk.blcu.edu.cn/"
TEXT_API = BASE_URL + "api/v1/resource/text/"
DOC_API = BASE_URL + "api/v1/resource/doc/"

REQUEST_INTERVAL = 1.5
TIMEOUT = 20


# ==================== 2. 请求配置 ====================

session = requests.Session()

retry = Retry(
    total=2,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET"],
)

adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

session.headers.update({
    "User-Agent": "HSK-Corpus-Research-Test/1.0"
})


def get_response(url):
    """发送请求并控制访问频率。"""
    time.sleep(REQUEST_INTERVAL)
    response = session.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response


def get_api_data(url):
    """获取并检查官网 JSON 接口数据。"""
    result = get_response(url).json()

    if not isinstance(result, dict):
        raise ValueError("接口响应不是 JSON 对象")

    if result.get("code") != 0:
        raise ValueError(f"接口返回异常：{result.get('msg')}")

    return result.get("data")


# ==================== 3. 读取作文编码 ====================

def load_composition_ids():
    df = pd.read_csv(
        CSV_PATH,
        dtype={ID_COLUMN: "string"},
        encoding="utf-8-sig"
    )

    if ID_COLUMN not in df.columns:
        raise ValueError(
            f"未找到字段 {ID_COLUMN}，实际字段：{df.columns.tolist()}"
        )

    ids = (
        df[ID_COLUMN]
        .dropna()
        .str.strip()
        .drop_duplicates()
        .head(TEST_LIMIT)
        .tolist()
    )

    # 作文编码只允许数字，避免构造异常 URL 或文件路径
    for composition_id in ids:
        if not composition_id.isdecimal():
            raise ValueError(f"作文编码格式异常：{composition_id}")

    print(f"CSV 记录数：{len(df)}")
    print(f"本次测试作文数：{len(ids)}")

    return ids


# ==================== 4. 采集完整标注 ====================

def collect_text(composition_id):
    url = TEXT_API + composition_id
    text = get_api_data(url)

    if not isinstance(text, str) or not text.strip():
        raise ValueError("完整标注文本为空或格式异常")

    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TEXT_DIR / f"{composition_id}.txt"
    output_path.write_text(text, encoding="utf-8")

    return {
        "text_path": str(output_path.relative_to(PROJECT_ROOT)),
        "text_length": len(text),
    }


# ==================== 5. 采集扫描图片 ====================

def collect_scans(composition_id):
    paths = get_api_data(DOC_API + composition_id)

    if not isinstance(paths, list) or not paths:
        raise ValueError("扫描图片路径列表为空或格式异常")

    output_dir = SCAN_DIR / composition_id
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    for page, relative_path in enumerate(paths, start=1):
        if not isinstance(relative_path, str):
            raise ValueError("图片路径不是字符串")

        image_url = urljoin(BASE_URL, relative_path)
        parsed = urlparse(image_url)

        # 仅允许访问官网域名下的 GIF 资源
        if (
            parsed.scheme != "https"
            or parsed.netloc != "hsk.blcu.edu.cn"
            or not parsed.path.startswith("/resource/zt/")
            or not parsed.path.lower().endswith(".gif")
        ):
            raise ValueError(f"图片路径不符合预期：{relative_path}")

        response = get_response(image_url)

        # GIF 文件头校验
        if not response.content.startswith((b"GIF87a", b"GIF89a")):
            raise ValueError(f"返回内容不是有效 GIF：{image_url}")

        output_path = output_dir / f"page_{page:02d}.gif"
        output_path.write_bytes(response.content)

        downloaded.append({
            "page": page,
            "source_path": relative_path,
            "local_path": str(output_path.relative_to(PROJECT_ROOT)),
            "size_bytes": len(response.content),
        })

    return {
        "expected_pages": len(paths),
        "downloaded_pages": len(downloaded),
        "images": downloaded,
    }


# ==================== 6. 主程序 ====================

def main():
    composition_ids = load_composition_ids()
    results = []

    for index, composition_id in enumerate(composition_ids, start=1):
        print(f"\n[{index}/{len(composition_ids)}] {composition_id}")

        record = {
            "composition_id": composition_id,
            "text_status": "failed",
            "scan_status": "failed",
        }

        try:
            text_info = collect_text(composition_id)
            record.update(text_info)
            record["text_status"] = "success"
            print(f"  标注文本：成功，{text_info['text_length']} 字符")
        except (requests.RequestException, ValueError, OSError) as e:
            record["text_error"] = str(e)
            print(f"  标注文本：失败，{e}")

        try:
            scan_info = collect_scans(composition_id)
            record.update(scan_info)
            record["scan_status"] = "success"
            print(f"  扫描图片：成功，{scan_info['downloaded_pages']} 页")
        except (requests.RequestException, ValueError, OSError) as e:
            record["scan_error"] = str(e)
            print(f"  扫描图片：失败，{e}")

        results.append(record)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    text_success = sum(r["text_status"] == "success" for r in results)
    scan_success = sum(r["scan_status"] == "success" for r in results)

    print("\n========== 测试结果 ==========")
    print(f"测试作文数：{len(results)}")
    print(f"标注采集成功：{text_success}")
    print(f"扫描采集成功：{scan_success}")
    print(f"采集报告：{REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

# AI辅助生成：GPT-6, 2026-11-17