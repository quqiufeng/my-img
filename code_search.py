#!/usr/bin/env python3
"""
Code Search Tool - 代码索引查询工具

【使用方法】

命令行参数:
  python3 code_search.py <索引文件.bin> [选项]

选项:
  --stats, -s              显示索引统计信息
  --find <name>, -f        精确查找符号名称
  --search <keyword>       模糊搜索关键字
  --file <filename>        获取指定文件中的所有函数
  --regex <pattern>        正则表达式搜索
  --fuzzy <keyword>        模糊匹配搜索
  --batch <symbols>        批量查找符号（逗号分隔）
  --json                   输出 JSON 格式
  --indent <n>             JSON 缩进（默认 2）
  --no-code                不显示代码片段（只显示元数据）
  --full-code              显示完整代码（不裁剪）
  --limit <n>              限制返回结果数量（默认 20）

【使用示例】

1. 查看索引统计信息:
   $ python3 code_search.py stable-diffusion-cpp.bin --stats

2. 精确查找函数（JSON 输出）:
   $ python3 code_search.py stable-diffusion-cpp.bin --find generate_image --json

3. 正则搜索:
   $ python3 code_search.py stable-diffusion-cpp.bin --regex "^ggml_.*" --json

4. 模糊搜索（拼写容错）:
   $ python3 code_search.py stable-diffusion-cpp.bin --fuzzy "gen_img" --json

5. 批量查找:
   $ python3 code_search.py stable-diffusion-cpp.bin --batch "func1,func2,func3" --json

【索引格式】
- 支持 V3 格式索引文件（由 code_index.py 生成）
- 包含符号名称、签名、文件路径、行号、代码片段
- 包含文件上下文：头文件依赖(includes)、宏定义(macros)、类型定义(typedefs)

【性能】
- 查询速度：毫秒级（内存映射 + 二进制索引）
- 内存占用：只加载查询结果，不占用AI上下文

【依赖】
- Python 3.8+
- 需要 code_index 项目生成的 .bin 索引文件
"""

import os
import sys
import json
import struct
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime


class SymbolKind(Enum):
    """符号类型"""

    FUNCTION = 0
    CLASS = 1
    STRUCT = 2
    ENUM = 3
    NAMESPACE = 4
    TYPEDEF = 5
    VARIABLE = 6
    MACRO = 7


@dataclass
class Symbol:
    """符号数据类"""

    name: str
    signature: str
    file_path: str
    line: int
    kind: int
    code: str
    includes: List[str]
    macros: Dict[str, str]
    typedefs: Dict[str, str]


class IndexReader:
    """索引读取器 - 支持 V3 格式（带上下文）"""

    MAGIC = 0x53594458  # "SYDX"
    VERSION = 3
    HEADER_SIZE = 128
    SYMBOL_SIZE = 80

    def __init__(self, index_path: str):
        self.index_path = index_path
        self.file = None
        self.header = {}
        self.string_table = b""
        self.code_table = b""
        self.context_table = b""
        self.num_symbols = 0
        self._symbol_cache = {}  # 名称 -> 偏移缓存

    def open(self) -> bool:
        """打开索引文件"""
        try:
            self.file = open(self.index_path, "rb")

            # 读取 Header
            header_data = struct.unpack("=32I", self.file.read(self.HEADER_SIZE))

            self.header = {
                "magic": header_data[0],
                "version": header_data[1],
                "num_symbols": header_data[2],
                "string_table_size": header_data[5],
                "string_table_offset": header_data[6],
                "symbols_offset": header_data[7],
                "code_table_size": header_data[9],
                "code_table_offset": header_data[10],
                "context_table_size": header_data[11],
                "context_table_offset": header_data[12],
            }

            if self.header["magic"] != self.MAGIC:
                return False

            if self.header["version"] < 3:
                return False

            self.num_symbols = self.header["num_symbols"]

            # 读取 String table
            self.file.seek(self.header["string_table_offset"])
            self.string_table = self.file.read(self.header["string_table_size"])

            # 读取 Code table
            self.file.seek(self.header["code_table_offset"])
            self.code_table = self.file.read(self.header["code_table_size"])

            # 读取 Context table
            if self.header.get("context_table_offset", 0) > 0:
                self.file.seek(self.header["context_table_offset"])
                self.context_table = self.file.read(self.header["context_table_size"])

            # 构建符号缓存
            self._build_symbol_cache()

            return True
        except Exception as e:
            return False

    def _build_symbol_cache(self):
        """构建符号名称到文件偏移的缓存"""
        self.file.seek(self.header["symbols_offset"])
        for i in range(self.num_symbols):
            offset = self.file.tell()
            data = struct.unpack("=20I", self.file.read(self.SYMBOL_SIZE))
            name = self._get_string(data[0])
            if name:
                self._symbol_cache[name] = offset

    def close(self):
        """关闭索引文件"""
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def _get_string(self, offset: int) -> str:
        """从 string table 获取字符串"""
        if offset == 0 or offset >= len(self.string_table):
            return ""
        end = offset
        while end < len(self.string_table) and self.string_table[end] != 0:
            end += 1
        return self.string_table[offset:end].decode("utf-8", errors="ignore")

    def _get_code(self, offset: int, length: int) -> str:
        """从 code table 获取代码"""
        if offset >= len(self.code_table) or length == 0:
            return ""
        end = min(offset + length, len(self.code_table))
        return self.code_table[offset:end].decode("utf-8", errors="ignore")

    def _get_context(self, offset: int, length: int) -> Dict:
        """从 context table 获取上下文"""
        if offset >= len(self.context_table) or length == 0:
            return {}
        end = min(offset + length, len(self.context_table))
        context_json = self.context_table[offset:end].decode("utf-8", errors="ignore")
        try:
            return json.loads(context_json)
        except json.JSONDecodeError:
            return {}

    def _read_symbol_at_offset(self, offset: int) -> Optional[Symbol]:
        """从指定偏移读取符号"""
        try:
            self.file.seek(offset)
            data = struct.unpack("=20I", self.file.read(self.SYMBOL_SIZE))

            name_offset = data[0]
            signature_offset = data[3]
            file_path_offset = data[5]
            line = data[8]
            code_offset = data[11]
            code_length = data[12]
            context_offset = data[13]
            context_length = data[14]
            kind = data[19]

            context = self._get_context(context_offset, context_length)

            return Symbol(
                name=self._get_string(name_offset),
                signature=self._get_string(signature_offset),
                file_path=self._get_string(file_path_offset),
                line=line,
                kind=kind,
                code=self._get_code(code_offset, code_length),
                includes=context.get("includes", []),
                macros=context.get("macros", {}),
                typedefs=context.get("typedefs", {}),
            )
        except Exception:
            return None

    def find_symbol(self, name: str) -> Optional[Symbol]:
        """精确查找符号（O(1)）"""
        if name in self._symbol_cache:
            return self._read_symbol_at_offset(self._symbol_cache[name])
        return None

    def find_symbols(self, names: List[str]) -> Dict[str, Optional[Symbol]]:
        """批量查找符号"""
        return {name: self.find_symbol(name) for name in names}

    def search_symbols(self, keyword: str, max_results: int = 20) -> List[Symbol]:
        """搜索包含关键字的符号"""
        results = []
        keyword_lower = keyword.lower()

        for name, offset in self._symbol_cache.items():
            if keyword_lower in name.lower():
                symbol = self._read_symbol_at_offset(offset)
                if symbol:
                    results.append(symbol)
                    if len(results) >= max_results:
                        break

        return results

    def search_regex(self, pattern: str, max_results: int = 20) -> List[Symbol]:
        """正则表达式搜索"""
        results = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            for name, offset in self._symbol_cache.items():
                if regex.search(name):
                    symbol = self._read_symbol_at_offset(offset)
                    if symbol:
                        results.append(symbol)
                        if len(results) >= max_results:
                            break
        except re.error:
            pass
        return results

    def search_fuzzy(
        self, keyword: str, threshold: int = 3, max_results: int = 20
    ) -> List[tuple]:
        """模糊匹配搜索（Levenshtein 距离）"""
        results = []
        keyword_lower = keyword.lower()

        for name, offset in self._symbol_cache.items():
            name_lower = name.lower()
            # 前缀匹配
            if name_lower.startswith(keyword_lower):
                score = 0
            else:
                # 计算 Levenshtein 距离
                score = self._levenshtein_distance(keyword_lower, name_lower)

            if score <= threshold:
                symbol = self._read_symbol_at_offset(offset)
                if symbol:
                    results.append((score, symbol))

        # 按分数排序
        results.sort(key=lambda x: x[0])
        return [(score, sym) for score, sym in results[:max_results]]

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算 Levenshtein 距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def get_functions_in_file(self, file_name: str) -> List[Symbol]:
        """获取指定文件中的所有函数"""
        results = []

        for name, offset in self._symbol_cache.items():
            symbol = self._read_symbol_at_offset(offset)
            if symbol and symbol.kind == 0 and file_name in symbol.file_path:
                results.append(symbol)

        return results

    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        return {
            "total_symbols": self.num_symbols,
            "version": self.header.get("version"),
            "string_table_size": len(self.string_table),
            "code_table_size": len(self.code_table),
            "context_table_size": len(self.context_table),
        }


def symbol_to_dict(symbol: Symbol, full_code: bool = False) -> Dict[str, Any]:
    """将 Symbol 转换为字典（用于 JSON 序列化）"""
    kind_names = {
        0: "function",
        1: "class",
        2: "struct",
        3: "enum",
        4: "namespace",
        5: "typedef",
        6: "variable",
        7: "macro",
    }

    # 智能裁剪代码片段
    code_snippet = symbol.code
    if not full_code and code_snippet:
        code_snippet = smart_truncate_code(code_snippet)

    return {
        "name": symbol.name,
        "kind": kind_names.get(symbol.kind, "unknown"),
        "signature": symbol.signature,
        "location": {"file": symbol.file_path, "line": symbol.line, "column": 0},
        "code_snippet": code_snippet,
        "context": {
            "includes": symbol.includes[:10] if symbol.includes else [],  # 限制数量
            "macros": symbol.macros,
            "typedefs": symbol.typedefs,
        },
    }


def smart_truncate_code(code: str, max_lines: int = 50) -> str:
    """智能裁剪代码片段

    策略：
    1. 如果代码 <= max_lines，返回完整代码
    2. 如果代码 > max_lines，提取：
       - 函数签名和前 20 行
       - 省略中间部分
       - 最后 10 行（通常是返回语句和结束括号）
    """
    lines = code.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines:
        return code

    # 提取关键部分
    header_lines = lines[:20]
    footer_lines = lines[-10:]

    truncated = (
        header_lines
        + ["    ...", f"    // ... ({total_lines - 30} lines omitted) ...", "    ..."]
        + footer_lines
    )
    return "\n".join(truncated)


def format_output(data: Any, json_output: bool = False, indent: int = 2) -> str:
    """格式化输出"""
    if json_output:
        return json.dumps(data, ensure_ascii=False, indent=indent)
    else:
        return str(data)


def print_symbol(symbol: Symbol, show_code: bool = True):
    """打印符号信息（人类可读格式）"""
    kind_names = {
        0: "函数",
        1: "类",
        2: "结构体",
        3: "枚举",
        4: "命名空间",
        5: "类型定义",
        6: "变量",
        7: "宏",
    }
    kind_name = kind_names.get(symbol.kind, "未知")

    print(f"\n{'=' * 60}")
    print(f"【{kind_name}】{symbol.name}")
    print(f"{'=' * 60}")
    print(f"文件: {symbol.file_path}:{symbol.line}")
    if symbol.signature:
        print(f"签名: {symbol.signature}")
    if symbol.includes:
        print(
            f"头文件: {', '.join(symbol.includes[:5])}{'...' if len(symbol.includes) > 5 else ''}"
        )

    if show_code and symbol.code:
        print(f"\n代码:")
        print("-" * 60)
        lines = symbol.code.split("\n")[:30]
        for i, line in enumerate(lines, 1):
            print(f"{i:3d}: {line}")
        total_lines = len(symbol.code.split("\n"))
        if total_lines > 30:
            remaining = total_lines - 30
            print(f"... (还有 {remaining} 行)")
        print("-" * 60)


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Code Index 查询工具")
    parser.add_argument("index", help="索引文件路径 (.bin)")
    parser.add_argument("--find", "-f", help="精确查找符号")
    parser.add_argument("--search", "-s", help="搜索关键字")
    parser.add_argument("--regex", "-r", help="正则表达式搜索")
    parser.add_argument("--fuzzy", help="模糊匹配搜索")
    parser.add_argument("--batch", "-b", help="批量查找符号（逗号分隔）")
    parser.add_argument("--file", help="获取指定文件中的所有函数")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    parser.add_argument("--json", "-j", action="store_true", help="输出 JSON 格式")
    parser.add_argument("--indent", type=int, default=2, help="JSON 缩进（默认 2）")
    parser.add_argument("--no-code", action="store_true", help="不显示代码")
    parser.add_argument(
        "--full-code", action="store_true", help="显示完整代码（不裁剪）"
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=20, help="限制返回结果数量（默认 20）"
    )

    args = parser.parse_args()

    if not os.path.exists(args.index):
        error_response = {
            "error": True,
            "message": f"索引文件不存在: {args.index}",
            "timestamp": datetime.now().isoformat(),
        }
        print(format_output(error_response, args.json))
        sys.exit(1)

    with IndexReader(args.index) as reader:
        if not reader.file:
            error_response = {
                "error": True,
                "message": "无法打开索引文件",
                "timestamp": datetime.now().isoformat(),
            }
            print(format_output(error_response, args.json))
            sys.exit(1)

        # 构建响应
        response = {
            "query": {
                "index_file": args.index,
                "timestamp": datetime.now().isoformat(),
            },
            "result": {},
        }

        if args.stats:
            response["query"]["type"] = "stats"
            stats = reader.get_stats()
            response["result"] = {"stats": stats}

            if not args.json:
                print(f"\n已加载索引: {args.index}")
                print(f"\n【索引统计】")
                print(f"  总符号数: {stats['total_symbols']:,}")
                print(f"  版本: {stats['version']}")
                print(f"  字符串表: {stats['string_table_size']:,} bytes")
                print(f"  代码表: {stats['code_table_size']:,} bytes")
                print(f"  上下文表: {stats['context_table_size']:,} bytes")

        if args.find:
            response["query"]["type"] = "find"
            response["query"]["keyword"] = args.find

            symbol = reader.find_symbol(args.find)
            if symbol:
                response["result"]["found"] = True
                response["result"]["symbol"] = symbol_to_dict(symbol, args.full_code)
                if not args.json:
                    print_symbol(symbol, not args.no_code)
            else:
                response["result"]["found"] = False
                response["result"]["message"] = f"未找到: {args.find}"
                if not args.json:
                    print(f"\n查找: {args.find}")
                    print(f"未找到: {args.find}")

        if args.batch:
            response["query"]["type"] = "batch"
            response["query"]["keywords"] = args.batch.split(",")

            symbols = reader.find_symbols(args.batch.split(","))
            response["result"]["symbols"] = {
                name: symbol_to_dict(sym, args.full_code) if sym else None
                for name, sym in symbols.items()
            }

            if not args.json:
                print(f"\n批量查找: {args.batch}")
                for name, sym in symbols.items():
                    if sym:
                        print(f"  ✓ {name}")
                    else:
                        print(f"  ✗ {name} (未找到)")

        if args.search:
            response["query"]["type"] = "search"
            response["query"]["keyword"] = args.search

            results = reader.search_symbols(args.search, args.limit)
            response["result"]["count"] = len(results)
            response["result"]["symbols"] = [
                symbol_to_dict(sym, args.full_code) for sym in results
            ]

            if not args.json:
                print(f"\n搜索: {args.search}")
                print(f"找到 {len(results)} 个结果:")
                kind_names = {
                    0: "函数",
                    1: "类",
                    2: "结构体",
                    3: "枚举",
                    4: "命名空间",
                    5: "类型定义",
                    6: "变量",
                    7: "宏",
                }
                for i, sym in enumerate(results[:10], 1):
                    kind_name = kind_names.get(sym.kind, "未知")
                    print(
                        f"  {i}. [{kind_name}] {sym.name} @ {sym.file_path}:{sym.line}"
                    )

        if args.regex:
            response["query"]["type"] = "regex"
            response["query"]["pattern"] = args.regex

            results = reader.search_regex(args.regex, args.limit)
            response["result"]["count"] = len(results)
            response["result"]["symbols"] = [
                symbol_to_dict(sym, args.full_code) for sym in results
            ]

            if not args.json:
                print(f"\n正则搜索: {args.regex}")
                print(f"找到 {len(results)} 个结果")

        if args.fuzzy:
            response["query"]["type"] = "fuzzy"
            response["query"]["keyword"] = args.fuzzy

            results = reader.search_fuzzy(args.fuzzy, max_results=args.limit)
            response["result"]["count"] = len(results)
            response["result"]["symbols"] = [
                {"distance": score, **symbol_to_dict(sym, args.full_code)}
                for score, sym in results
            ]

            if not args.json:
                print(f"\n模糊搜索: {args.fuzzy}")
                print(f"找到 {len(results)} 个结果")

        if args.file:
            response["query"]["type"] = "file"
            response["query"]["filename"] = args.file

            results = reader.get_functions_in_file(args.file)
            response["result"]["count"] = len(results)
            response["result"]["symbols"] = [
                symbol_to_dict(sym, args.full_code) for sym in results[: args.limit]
            ]

            if not args.json:
                print(f"\n获取文件: {args.file}")
                print(f"找到 {len(results)} 个函数:")
                for i, sym in enumerate(results[:20], 1):
                    print(f"  {i}. {sym.name} (line {sym.line})")

        # 输出 JSON（如果指定了 --json）
        if args.json:
            print(format_output(response, True, args.indent))


if __name__ == "__main__":
    main()
