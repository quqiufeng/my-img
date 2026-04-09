#!/usr/bin/env python3
"""
Code Index Builder - 代码索引构建工具

【使用方法】

1. 直接运行（使用默认路径）:
   python3 code_index.py

2. 修改脚本中的路径变量后运行:
   project_path = "/path/to/your/project"
   output_path = "/path/to/output.bin"

【功能说明】
- 扫描项目中的所有源代码文件（.c, .cpp, .h, .hpp 等）
- 提取函数、类、结构体等符号信息
- 自动过滤超过10MB的大文件（避免解析超大词汇表文件）
- 生成二进制索引文件(.bin)，供 code_search.py 查询使用

【输出文件】
- 生成 .bin 索引文件，包含所有符号的元数据和代码片段
- 索引格式：V3（支持文件上下文：includes, macros, typedefs）

【依赖】
- 需要 code_index 项目已安装: /home/dministrator/my-agi/code_index
- Python 3.8+

【示例】
$ cd /home/dministrator/my-img
$ python3 code_index.py
开始构建索引...
项目路径: /home/dministrator/stable-diffusion.cpp
输出路径: /home/dministrator/stable-diffusion-cpp.bin
...
Build complete in 114.2s
  Symbols: 88750
  Output: 109,857,234 bytes
"""

import os
import sys
import glob
import time

sys.path.insert(0, "/home/dministrator/my-agi/code_index/src/ctags")

from enhanced_index_builder import build_index

project_path = "/home/dministrator/stable-diffusion.cpp"
output_path = "/home/dministrator/stable-diffusion-cpp.bin"

print(f"开始构建索引...")
print(f"项目路径: {project_path}")
print(f"输出路径: {output_path}")
print()

# 先检查文件数量
cpp_files = glob.glob(os.path.join(project_path, "**", "*.cpp"), recursive=True)
c_files = glob.glob(os.path.join(project_path, "**", "*.c"), recursive=True)
h_files = glob.glob(os.path.join(project_path, "**", "*.h"), recursive=True)
hpp_files = glob.glob(os.path.join(project_path, "**", "*.hpp"), recursive=True)

print(f"找到 {len(cpp_files)} 个 .cpp 文件")
print(f"找到 {len(c_files)} 个 .c 文件")
print(f"找到 {len(h_files)} 个 .h 文件")
print(f"找到 {len(hpp_files)} 个 .hpp 文件")
print(f"总计: {len(cpp_files) + len(c_files) + len(h_files) + len(hpp_files)} 个文件")
print()

# 显示前10个文件
print("前10个文件:")
for f in (cpp_files + c_files + h_files + hpp_files)[:10]:
    size = os.path.getsize(f)
    print(f"  {os.path.basename(f)} ({size:,} bytes)")
print()


def progress_callback(current_file, current_num, total_num):
    pct = (current_num / total_num) * 100
    print(
        f"  [{current_num}/{total_num} {pct:.1f}%] {os.path.basename(current_file)}",
        flush=True,
    )


print("开始解析...")
start_time = time.time()

try:
    result = build_index(
        project_path=project_path,
        output_path=output_path,
        verbose=True,
        progress_callback=progress_callback,
    )

    print()
    print(f"构建完成!")
    print(f"  符号数量: {result['total_symbols']}")
    print(f"  耗时: {result['elapsed_time']:.1f}s")
    print(f"  输出大小: {result['output_size']:,} bytes")
except Exception as e:
    print(f"错误: {e}")
    import traceback

    traceback.print_exc()
