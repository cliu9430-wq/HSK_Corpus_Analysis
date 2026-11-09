"""Validate the documented code inventory; does not validate corpus annotations."""
import csv
import re
import sys
from pathlib import Path

FIELDS = ['code', 'level', 'description', 'code_type', 'example', 'validation_status']
# Numeric lexical envelope only: the source does not define range or digit width.
RULES = {
    'CC<n>': r'CC[0-9]+',
    'W<n>': r'W[0-9]+',
    'CJ-<component>': r'CJ-(?:zhuy|wy|sy|by|buy|dy|zy|zxy)',
    'CJ+<component>': r'CJ\+(?:zhuy|wy|sy|by|buy|dy|zy|zxy)',
}

def validate(path):
    errors = []
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            errors.append(f'字段必须为 {FIELDS}')
        rows = list(reader)
    seen = set()
    for line, row in enumerate(rows, 2):
        for field in FIELDS:
            # Several sentence types are listed without an official example.
            if field != 'example' and not (row.get(field) or '').strip():
                errors.append(f'第{line}行缺少 {field}')
        if None in row or any(row.get(f) is None for f in FIELDS):
            errors.append(f'第{line}行列数不完整')
        code = row.get('code', '')
        if code in seen:
            errors.append(f'重复编码: {code}')
        seen.add(code)
        if row.get('validation_status') not in ('confirmed', 'pending'):
            errors.append(f'第{line}行状态值非法: {row.get("validation_status")}')
        kind = row.get('code_type')
        if kind not in ('固定编码', '参数化编码'):
            errors.append(f'未知编码类型: {kind}')
        if (kind == '参数化编码') != (code in RULES):
            errors.append(f'规则与类型冲突: {code}')
        if row.get('level') not in ('字', '词', '句', '篇章', '标点符号'):
            errors.append(f'未知层级: {code}')
        if kind == '固定编码':
            for rule, pattern in RULES.items():
                if re.fullmatch(pattern, code):
                    errors.append(f'固定编码与参数规则重叠: {code} / {rule}')
    if set(RULES) - seen:
        errors.append('缺少参数化规则')

    return rows, errors

def confirmed_codes(rows):
    """供后续匹配程序使用：仅返回已确认条目，排除待核验编码。"""
    return [row for row in rows if row.get('validation_status') == 'confirmed']


if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'config/hsk_annotation_codes.csv'
    rows, errors = validate(path)
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        sys.exit(1)
    fixed = sum(r['code_type'] == '固定编码' for r in rows)
    print(f'校验通过：{len(rows)} 条，固定编码 {fixed} 条，参数化规则 {len(rows)-fixed} 条。')

    confirmed = len(confirmed_codes(rows))
    pending = sum(r['validation_status'] == 'pending' for r in rows)
    print(f'confirmed {confirmed} 条，pending {pending} 条；有效编码统计应仅使用 confirmed 条目。')

# AI辅助生成：GPT-6, 2026-11-8