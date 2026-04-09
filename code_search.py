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
  --no-code                不显示代码片段（只显示元数据）

【使用示例】

1. 查看索引统计信息:
   $ python3 code_search.py stable-diffusion-cpp.bin --stats

   输出:
   【索引统计】
     总符号数: 88,750
     版本: 3
     字符串表: 2,000,863 bytes
     代码表: 14,435,037 bytes
     上下文表: 85,693,877 bytes

2. 精确查找函数:
   $ python3 code_search.py stable-diffusion-cpp.bin --find generate_image

   输出:
   【函数】generate_image
   文件: /path/to/stable-diffusion.cpp:3585
   签名: sd_image_t* generate_image(...)
   头文件: ggml_extend.hpp, model.h, ...
   [代码片段...]

3. 模糊搜索关键字:
   $ python3 code_search.py stable-diffusion-cpp.bin --search upscale

   输出:
   找到 20 个结果:
     1. [变量] upscale_factor @ ...
     2. [函数] ggml_compute_forward_upscale_f32 @ ...

4. 获取文件中的所有函数:
   $ python3 code_search.py stable-diffusion-cpp.bin --file stable-diffusion.cpp

   输出:
   找到 156 个函数:
     1. generate_image (line 3585)
     2. img2img (line 3680)
     ...

5. 只显示元数据（不显示代码）:
   $ python3 code_search.py stable-diffusion-cpp.bin --find main --no-code

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

【作者】
基于 codemine 项目的 SymbolIndexReader 实现
"""

import os
import sys
import json
import struct
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


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
                print(f"错误: 无效的 magic number: {self.header['magic']:#x}")
                return False

            if self.header["version"] < 3:
                print(f"错误: 只支持 V3+ 索引，当前版本: {self.header['version']}")
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

            return True
        except Exception as e:
            print(f"错误: 无法打开索引文件: {e}")
            return False

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

    def find_symbol(self, name: str) -> Optional[Symbol]:
        """精确查找符号"""
        self.file.seek(self.header["symbols_offset"])

        for _ in range(self.num_symbols):
            data = struct.unpack("=20I", self.file.read(self.SYMBOL_SIZE))

            name_offset = data[0]
            symbol_name = self._get_string(name_offset)

            if symbol_name == name:
                # 找到匹配
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
                    name=symbol_name,
                    signature=self._get_string(signature_offset),
                    file_path=self._get_string(file_path_offset),
                    line=line,
                    kind=kind,
                    code=self._get_code(code_offset, code_length),
                    includes=context.get("includes", []),
                    macros=context.get("macros", {}),
                    typedefs=context.get("typedefs", {}),
                )

        return None

    def search_symbols(self, keyword: str, max_results: int = 20) -> List[Symbol]:
        """搜索包含关键字的符号"""
        results = []
        self.file.seek(self.header["symbols_offset"])

        for _ in range(self.num_symbols):
            data = struct.unpack("=20I", self.file.read(self.SYMBOL_SIZE))

            name_offset = data[0]
            symbol_name = self._get_string(name_offset)

            if keyword.lower() in symbol_name.lower():
                signature_offset = data[3]
                file_path_offset = data[5]
                line = data[8]
                code_offset = data[11]
                code_length = data[12]
                context_offset = data[13]
                context_length = data[14]
                kind = data[19]

                context = self._get_context(context_offset, context_length)

                results.append(
                    Symbol(
                        name=symbol_name,
                        signature=self._get_string(signature_offset),
                        file_path=self._get_string(file_path_offset),
                        line=line,
                        kind=kind,
                        code=self._get_code(code_offset, code_length),
                        includes=context.get("includes", []),
                        macros=context.get("macros", {}),
                        typedefs=context.get("typedefs", {}),
                    )
                )

                if len(results) >= max_results:
                    break

        return results

    def get_functions_in_file(self, file_name: str) -> List[Symbol]:
        """获取指定文件中的所有函数"""
        results = []
        self.file.seek(self.header["symbols_offset"])

        for _ in range(self.num_symbols):
            data = struct.unpack("=20I", self.file.read(self.SYMBOL_SIZE))

            kind = data[19]
            if kind != 0:  # 只取函数
                continue

            file_path_offset = data[5]
            file_path = self._get_string(file_path_offset)

            if file_name in file_path:
                name_offset = data[0]
                signature_offset = data[3]
                line = data[8]
                code_offset = data[11]
                code_length = data[12]
                context_offset = data[13]
                context_length = data[14]

                context = self._get_context(context_offset, context_length)

                results.append(
                    Symbol(
                        name=self._get_string(name_offset),
                        signature=self._get_string(signature_offset),
                        file_path=file_path,
                        line=line,
                        kind=kind,
                        code=self._get_code(code_offset, code_length),
                        includes=context.get("includes", []),
                        macros=context.get("macros", {}),
                        typedefs=context.get("typedefs", {}),
                    )
                )

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


def print_symbol(symbol: Symbol, show_code: bool = True):
    """打印符号信息"""
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
        # 只显示前30行
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
    parser.add_argument("--file", help="获取指定文件中的所有函数")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    parser.add_argument("--no-code", action="store_true", help="不显示代码")

    args = parser.parse_args()

    if not os.path.exists(args.index):
        print(f"错误: 索引文件不存在: {args.index}")
        sys.exit(1)

    with IndexReader(args.index) as reader:
        if not reader.file:
            sys.exit(1)

        print(f"\n已加载索引: {args.index}")

        if args.stats:
            stats = reader.get_stats()
            print(f"\n【索引统计】")
            print(f"  总符号数: {stats['total_symbols']:,}")
            print(f"  版本: {stats['version']}")
            print(f"  字符串表: {stats['string_table_size']:,} bytes")
            print(f"  代码表: {stats['code_table_size']:,} bytes")
            print(f"  上下文表: {stats['context_table_size']:,} bytes")

        if args.find:
            print(f"\n查找: {args.find}")
            symbol = reader.find_symbol(args.find)
            if symbol:
                print_symbol(symbol, not args.no_code)
            else:
                print(f"未找到: {args.find}")

        if args.search:
            print(f"\n搜索: {args.search}")
            results = reader.search_symbols(args.search)
            print(f"找到 {len(results)} 个结果:")
            for i, sym in enumerate(results[:10], 1):
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
                kind_name = kind_names.get(sym.kind, "未知")
                print(f"  {i}. [{kind_name}] {sym.name} @ {sym.file_path}:{sym.line}")

            if len(results) > 0 and not args.no_code:
                print_symbol(results[0], True)

        if args.file:
            print(f"\n获取文件: {args.file}")
            results = reader.get_functions_in_file(args.file)
            print(f"找到 {len(results)} 个函数:")
            for i, sym in enumerate(results[:20], 1):
                print(f"  {i}. {sym.name} (line {sym.line})")

            if len(results) > 0 and not args.no_code:
                print(f"\n显示第一个函数的代码:")
                print_symbol(results[0], True)


if __name__ == "__main__":
    main()
