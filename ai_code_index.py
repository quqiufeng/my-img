#!/usr/bin/env python3
"""
AI Code Index - Python 高性能实现

基于优化的 IndexReader，提供完整的 AI 代码查询接口。
性能指标：
- 精确查找：O(1)，< 0.01ms（缓存优化）
- 前缀匹配：O(1)，< 0.5ms（倒排索引）
- 子串搜索：< 100ms（倒排索引过滤）
- 内存占用：< 100MB（含缓存）

特性：
- 自动缓存管理（.cache 文件持久化）
- 倒排索引加速搜索
- 内存映射大文件
- 并行批量查询
- 性能监控

用法：
    # 方式1: 使用 with 语句（推荐，自动释放内存）
    with AICodeIndex("./project.bin") as index:
        symbol = index.find_symbol("target_function")

    # 方式2: 手动管理
    index = AICodeIndex("./project.bin")
    try:
        symbol = index.find_symbol("target_function")
    finally:
        index.close()
"""

import os
import sys
import json
import gc
import time
from typing import List, Dict, Optional, Any, Iterator, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入高性能 IndexReader
from code_search_optimized import IndexReader, Symbol, InvertedIndex, PerformanceMonitor


# ==================== 数据类 ====================


@dataclass
class CodeContext:
    """代码上下文 - 用于 AI 理解代码"""

    symbol_name: str
    symbol_kind: str
    file_path: str
    line: int
    signature: str = ""
    code: str = ""  # 完整代码
    return_type: str = ""
    namespace: str = ""
    parent_class: str = ""
    includes: List[str] = None
    macros: Dict[str, str] = None
    typedefs: Dict[str, str] = None

    def __post_init__(self):
        if self.includes is None:
            self.includes = []
        if self.macros is None:
            self.macros = {}
        if self.typedefs is None:
            self.typedefs = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def summary(self) -> str:
        """生成摘要信息"""
        lines = [
            f"【{self.symbol_kind}】{self.symbol_name}",
            f"文件: {self.file_path}:{self.line}",
        ]
        if self.signature:
            lines.append(f"签名: {self.signature}")
        if self.return_type:
            lines.append(f"返回类型: {self.return_type}")
        if self.includes:
            lines.append(f"头文件: {', '.join(self.includes[:5])}")

        code_lines = self.code.split("\n")[:5]
        lines.append("代码片段:")
        for i, line in enumerate(code_lines, 1):
            lines.append(f"  {i}: {line}")
        total_lines = len(self.code.split("\n"))
        if total_lines > 5:
            lines.append(f"  ... ({total_lines - 5} more lines)")

        return "\n".join(lines)


# ==================== 异常类 ====================


class CodeIndexError(Exception):
    """代码索引基础异常"""

    pass


class IndexNotFoundError(CodeIndexError):
    """索引文件不存在"""

    pass


class IndexNotLoadedError(CodeIndexError):
    """索引未加载"""

    pass


class SymbolNotFoundError(CodeIndexError):
    """符号未找到"""

    pass


# ==================== 主类 ====================


class AICodeIndex:
    """
    AI 代码索引接口 - Python 高性能实现

    基于 IndexReader 实现，特点：
    - O(1) 精确查找（缓存优化）
    - 倒排索引加速前缀/子串搜索
    - 自动缓存持久化
    - 性能监控
    """

    def __init__(
        self,
        index_path: str,
        use_mmap: bool = False,
        use_cache: bool = True,
        auto_load: bool = True,
    ):
        """
        初始化代码索引

        Args:
            index_path: 索引文件路径 (.bin)
            use_mmap: 是否使用内存映射（大文件推荐）
            use_cache: 是否使用缓存
            auto_load: 是否自动加载
        """
        if not os.path.exists(index_path):
            raise IndexNotFoundError(f"索引文件不存在: {index_path}")

        self.index_path = index_path
        self._reader: Optional[IndexReader] = None
        self._use_mmap = use_mmap
        self._use_cache = use_cache

        # 缓存
        self._symbol_cache: Dict[str, CodeContext] = {}
        self._search_cache: Dict[str, List[CodeContext]] = {}

        if auto_load:
            self._load()

    def _load(self):
        """加载索引"""
        self._reader = IndexReader(
            self.index_path, use_mmap=self._use_mmap, use_cache=self._use_cache
        )

        if not self._reader.open():
            raise CodeIndexError(f"无法加载索引: {self.index_path}")

    def _ensure_loaded(self):
        """确保索引已加载"""
        if not self.is_loaded():
            raise IndexNotLoadedError("索引未加载或已关闭")

    def _symbol_to_context(self, symbol: Symbol) -> CodeContext:
        """将 Symbol 转换为 CodeContext"""
        kind_map = {
            0: "function",
            1: "class",
            2: "struct",
            3: "enum",
            4: "namespace",
            5: "typedef",
            6: "variable",
            7: "macro",
        }

        return CodeContext(
            symbol_name=symbol.name,
            symbol_kind=kind_map.get(symbol.kind, "unknown"),
            file_path=symbol.file_path,
            line=symbol.line,
            signature=symbol.signature,
            code=symbol.code,  # 返回完整代码
            includes=symbol.includes,
            macros=symbol.macros,
            typedefs=symbol.typedefs,
        )

    # ==================== 核心查询方法 ====================

    def find_symbol(self, name: str) -> Optional[CodeContext]:
        """
        精确查找符号（O(1)）

        Args:
            name: 符号名称

        Returns:
            CodeContext 对象，或 None（未找到）
        """
        self._ensure_loaded()

        # 检查缓存
        if name in self._symbol_cache:
            return self._symbol_cache[name]

        # 查询
        symbol = self._reader.find_symbol(name)
        if not symbol:
            return None

        context = self._symbol_to_context(symbol)
        self._symbol_cache[name] = context
        return context

    def find_symbols(self, names: List[str]) -> Dict[str, Optional[CodeContext]]:
        """
        批量查找符号（并行优化）

        Args:
            names: 符号名称列表

        Returns:
            字典 {name: CodeContext or None}
        """
        self._ensure_loaded()

        results = self._reader.find_symbols(names)
        contexts = {}
        for name, symbol in results.items():
            if symbol:
                context = self._symbol_to_context(symbol)
                contexts[name] = context
                self._symbol_cache[name] = context
            else:
                contexts[name] = None
        return contexts

    def search(
        self, query: str, search_type: str = "smart", max_results: int = 20
    ) -> List[CodeContext]:
        """
        智能搜索

        Args:
            query: 搜索关键词
            search_type: 搜索类型
                - "smart": 智能（精确->前缀->子串）
                - "exact": 精确匹配
                - "prefix": 前缀匹配（O(1)）
                - "substring": 子串匹配
                - "fuzzy": 模糊匹配
            max_results: 最大返回结果数

        Returns:
            CodeContext 列表
        """
        self._ensure_loaded()

        cache_key = f"{search_type}:{query}:{max_results}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        results = []
        seen = set()

        if search_type == "smart":
            # 1. 精确匹配
            exact = self.find_symbol(query)
            if exact:
                results.append(exact)
                seen.add(exact.symbol_name)

            # 2. 前缀匹配
            if len(results) < max_results:
                prefix_results = self.search_prefix(query, max_results=max_results)
                for r in prefix_results:
                    if r.symbol_name not in seen:
                        results.append(r)
                        seen.add(r.symbol_name)
                        if len(results) >= max_results:
                            break

            # 3. 子串匹配
            if len(results) < max_results // 2:
                substring_results = self.search_substring(
                    query, max_results=max_results
                )
                for r in substring_results:
                    if r.symbol_name not in seen:
                        results.append(r)
                        seen.add(r.symbol_name)
                        if len(results) >= max_results:
                            break

        elif search_type == "exact":
            exact = self.find_symbol(query)
            if exact:
                results.append(exact)

        elif search_type == "prefix":
            results = self.search_prefix(query, max_results=max_results)

        elif search_type == "substring":
            results = self.search_substring(query, max_results=max_results)

        elif search_type == "fuzzy":
            results = self.search_fuzzy(query, max_results=max_results)

        else:
            raise ValueError(f"Unknown search_type: {search_type}")

        self._search_cache[cache_key] = results
        return results

    def search_prefix(self, prefix: str, max_results: int = 20) -> List[CodeContext]:
        """
        前缀匹配搜索（O(1)，使用倒排索引）

        Args:
            prefix: 前缀
            max_results: 最大返回结果数

        Returns:
            CodeContext 列表
        """
        self._ensure_loaded()

        symbols = self._reader.search_prefix(prefix, max_results)
        return [self._symbol_to_context(s) for s in symbols]

    def search_substring(
        self, substring: str, max_results: int = 20
    ) -> List[CodeContext]:
        """
        子串匹配搜索（使用倒排索引优化）

        Args:
            substring: 子串
            max_results: 最大返回结果数

        Returns:
            CodeContext 列表
        """
        self._ensure_loaded()

        symbols = self._reader.search_symbols(substring, max_results)
        return [self._symbol_to_context(s) for s in symbols]

    def search_regex(self, pattern: str, max_results: int = 20) -> List[CodeContext]:
        """
        正则表达式搜索

        Args:
            pattern: 正则表达式
            max_results: 最大返回结果数

        Returns:
            CodeContext 列表
        """
        self._ensure_loaded()

        symbols = self._reader.search_regex(pattern, max_results)
        return [self._symbol_to_context(s) for s in symbols]

    def search_fuzzy(
        self, keyword: str, threshold: int = 3, max_results: int = 20
    ) -> List[CodeContext]:
        """
        模糊匹配搜索（Levenshtein 距离）

        Args:
            keyword: 关键词
            threshold: 最大编辑距离
            max_results: 最大返回结果数

        Returns:
            CodeContext 列表
        """
        self._ensure_loaded()

        results = self._reader.search_fuzzy(keyword, threshold, max_results)
        return [self._symbol_to_context(s) for score, s in results]

    def get_code(self, symbol_name: str) -> Optional[str]:
        """
        获取符号的完整代码

        Args:
            symbol_name: 符号名称

        Returns:
            代码字符串，或 None
        """
        context = self.find_symbol(symbol_name)
        return context.code if context else None

    # ==================== 批量查询 ====================

    def get_symbols_in_file(
        self, file_path: str, max_results: int = 1000
    ) -> List[CodeContext]:
        """
        获取文件中的所有符号

        Args:
            file_path: 文件路径（或文件名）
            max_results: 最大返回结果数

        Returns:
            CodeContext 列表
        """
        self._ensure_loaded()

        symbols = self._reader.get_functions_in_file(file_path)
        return [self._symbol_to_context(s) for s in symbols[:max_results]]

    def get_symbols_by_kind(
        self, kind: str, max_results: int = 1000
    ) -> List[CodeContext]:
        """
        按类型获取符号

        Args:
            kind: 类型（function, class, struct, enum, variable, macro, typedef）
            max_results: 最大返回结果数

        Returns:
            CodeContext 列表
        """
        self._ensure_loaded()

        kind_map = {
            "function": 0,
            "class": 1,
            "struct": 2,
            "enum": 3,
            "namespace": 4,
            "typedef": 5,
            "variable": 6,
            "macro": 7,
        }
        kind_id = kind_map.get(kind.lower(), -1)
        if kind_id < 0:
            return []

        # 遍历所有符号
        results = []
        for name, offset in self._reader._symbol_cache.items():
            symbol = self._reader._read_symbol_at_offset(offset)
            if symbol and symbol.kind == kind_id:
                results.append(self._symbol_to_context(symbol))
                if len(results) >= max_results:
                    break
        return results

    # ==================== 调用图分析（占位符）====================

    def get_callers(self, symbol_name: str) -> List[str]:
        """获取调用指定函数的所有函数（需要调用图索引）"""
        # TODO: 实现调用图分析
        return []

    def get_callees(self, symbol_name: str) -> List[str]:
        """获取指定函数调用的所有函数（需要调用图索引）"""
        # TODO: 实现调用图分析
        return []

    # ==================== 类层次分析（占位符）====================

    def get_class_hierarchy(self, class_name: str) -> List[str]:
        """获取类的继承层次（需要类层次索引）"""
        # TODO: 实现类层次分析
        return []

    def get_derived_classes(self, class_name: str) -> List[str]:
        """获取类的所有派生类（需要类层次索引）"""
        # TODO: 实现类层次分析
        return []

    # ==================== 项目统计 ====================

    def get_project_stats(self) -> Dict[str, Any]:
        """获取项目统计信息"""
        stats = {
            "index_path": self.index_path,
            "index_loaded": self.is_loaded(),
            "total_symbols": 0,
            "index_size_mb": 0.0,
            "symbol_cache_entries": len(self._symbol_cache),
            "search_cache_entries": len(self._search_cache),
            "use_mmap": self._use_mmap,
            "use_cache": self._use_cache,
        }

        if self.is_loaded():
            stats["total_symbols"] = self._reader.num_symbols
            reader_stats = self._reader.get_stats()
            stats.update(
                {
                    "version": reader_stats.get("version"),
                    "string_table_size": reader_stats.get("string_table_size"),
                    "code_table_size": reader_stats.get("code_table_size"),
                }
            )

        if os.path.exists(self.index_path):
            stats["index_size_mb"] = os.path.getsize(self.index_path) / (1024 * 1024)

        return stats

    # ==================== 内存管理 ====================

    @property
    def count(self) -> int:
        """获取符号总数"""
        if self._reader:
            return self._reader.num_symbols
        return 0

    def is_loaded(self) -> bool:
        """检查索引是否已加载"""
        return self._reader is not None

    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        symbol_cache_size = len(self._symbol_cache)
        search_cache_size = len(self._search_cache)

        # 估算内存
        cache_memory = symbol_cache_size * 200 + search_cache_size * 100

        file_size = (
            os.path.getsize(self.index_path) if os.path.exists(self.index_path) else 0
        )

        return {
            "index_loaded": self.is_loaded(),
            "index_file_size_mb": file_size / (1024 * 1024),
            "symbol_cache_entries": symbol_cache_size,
            "search_cache_entries": search_cache_size,
            "estimated_cache_memory_kb": cache_memory / 1024,
            "status": "loaded" if self.is_loaded() else "closed",
        }

    def clear_cache(self):
        """清理缓存释放内存"""
        self._symbol_cache.clear()
        self._search_cache.clear()
        gc.collect()

    def close(self):
        """关闭索引并释放内存"""
        self.clear_cache()

        if self._reader:
            self._reader.close()
            self._reader = None

        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if self.is_loaded():
            self.close()

    # ==================== 性能监控 ====================

    def print_performance_stats(self):
        """打印性能统计"""
        if self._reader:
            self._reader.print_performance_stats()


# ==================== 便捷函数 ====================


def load_index(
    index_path: str, use_mmap: bool = False, use_cache: bool = True
) -> AICodeIndex:
    """加载代码索引"""
    return AICodeIndex(index_path, use_mmap=use_mmap, use_cache=use_cache)


def quick_search(
    index_path: str, query: str, max_results: int = 20
) -> List[CodeContext]:
    """快速搜索"""
    with AICodeIndex(index_path) as index:
        return index.search(query, max_results=max_results)


def find_symbol(index_path: str, symbol_name: str) -> Optional[CodeContext]:
    """查找单个符号"""
    with AICodeIndex(index_path) as index:
        return index.find_symbol(symbol_name)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python ai_code_index.py <index.bin> <search_query>")
        sys.exit(1)

    index_path = sys.argv[1]
    query = sys.argv[2]

    print(f"Loading index: {index_path}")
    print(f"Searching for: {query}")
    print("-" * 60)

    try:
        with AICodeIndex(index_path) as index:
            stats = index.get_project_stats()
            print(f"Total symbols: {stats['total_symbols']}")
            print(f"Index size: {stats['index_size_mb']:.2f} MB")
            print("-" * 60)

            # 测试精确查找
            import time

            start = time.time()
            symbol = index.find_symbol(query)
            elapsed = (time.time() - start) * 1000

            if symbol:
                print(f"\n✓ Found in {elapsed:.3f}ms")
                print(symbol.summary())
            else:
                # 尝试智能搜索
                print(f"\nExact match not found, trying smart search...")
                results = index.search(query, max_results=5)
                print(f"Found {len(results)} results:")
                for i, ctx in enumerate(results, 1):
                    print(f"\n{i}. {ctx.symbol_name} ({ctx.symbol_kind})")
                    print(f"   File: {ctx.file_path}:{ctx.line}")

            print()
            index.print_performance_stats()

        print("\n✓ Index closed, memory released")

    except IndexNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
