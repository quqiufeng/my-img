# Symbol Index V3 Adapter Layer

Clean C/C++/LuaJIT adapter layer for the Symbol Index V3 library, providing high-performance code symbol lookup with multiple language bindings.

## Overview

The adapter layer provides:
- **C API**: Core low-level interface
- **C++ Wrapper**: Modern RAII-based C++ interface with iterators and full search support
- **LuaJIT FFI**: High-performance Lua bindings

All adapters use the same compiled shared library (`libsymbol_index_v3.so`).

## Files

| File | Description |
|------|-------------|
| `symbol_index_v3.c` | Core C implementation with adapter functions |
| `symbol_index_v3.h` | C/C++ header file (clean interface) |
| `symbol_index_v3.hpp` | Modern C++ wrapper class with iterators |
| `symbol_index_v3_ffi.lua` | LuaJIT FFI bindings |
| `libsymbol_index_v3.so` | Compiled shared library |
| `test_symbol_index_cpp.cpp` | Comprehensive C++ test program |
| `test_symbol_index_lua.lua` | LuaJIT test program |

## Building

```bash
# Compile shared library
gcc -O2 -fPIC -shared -o libsymbol_index_v3.so symbol_index_v3.c

# Compile C++ test
g++ -std=c++17 -I. test_symbol_index_cpp.cpp -L. -lsymbol_index_v3 -o test_symbol_index_cpp

# Run C++ test
LD_LIBRARY_PATH=. ./test_symbol_index_cpp test_index.bin

# Run LuaJIT test (no compilation needed)
luajit test_symbol_index_lua.lua test_index.bin
```

## Usage Examples

### C API

```c
#include "symbol_index_v3.h"

SymbolIndex *idx = symbol_index_open("project.bin");
if (idx) {
    // Single lookup
    SymbolResult *res = symbol_index_find(idx, "my_function");
    if (res) {
        printf("Found: %s at %s:%d\n",
               symbol_result_get_name(res),
               symbol_result_get_file(res),
               symbol_result_get_line(res));
        symbol_result_free(res);
    }
    
    // Prefix search
    int count;
    SymbolResult *results = symbol_index_find_prefix(idx, "test_", &count);
    if (results) {
        for (int i = 0; i < count; i++) {
            printf("  - %s\n", symbol_result_get_name(&results[i]));
        }
        symbol_results_free(results, count);
    }
    
    // Glob, fuzzy, regex searches also available
    // symbol_index_glob(), symbol_index_fuzzy(), symbol_index_regex()
    
    symbol_index_close(idx);
}
```

### C++ API

```cpp
#include "symbol_index_v3.hpp"

// Open index
code_index::SymbolIndexV3 idx("project.bin");

// Single lookup
auto result = idx.find("my_function");
if (result) {
    std::cout << result->name << " at " 
              << result->file << ":" << result->line;
}

// Range-based iteration over all symbols
for (const auto& sym : idx) {
    std::cout << sym.name << std::endl;
}

// Prefix search
auto matches = idx.find_prefix("test_");
for (const auto& sym : matches) {
    std::cout << "- " << sym.name << std::endl;
}

// Glob pattern search
auto globs = idx.glob("foo*bar");

// Fuzzy search (edit distance)
auto fuzzy = idx.fuzzy("myfuncton", 2);  // max distance 2

// Regex search
auto regex = idx.regex("^test_.*");
```

### LuaJIT API

```lua
local idxlib = require("symbol_index_v3_ffi")
local idx = idxlib.open("project.bin")
local result = idx:find("my_function")
if result then
    print(result.name .. " at " .. result.file .. ":" .. result.line)
end
idx:close()
```

## API Reference

### Core Functions (C API)

| Function | Description |
|----------|-------------|
| `symbol_index_open(path)` | Open index file (memory-mapped) |
| `symbol_index_close(idx)` | Close index |
| `symbol_index_count(idx)` | Get symbol count |
| `symbol_index_find(idx, name)` | Find symbol by exact name |
| `symbol_index_find_prefix(idx, prefix, &count)` | Find by prefix |
| `symbol_index_glob(idx, pattern, &count)` | Glob pattern search |
| `symbol_index_fuzzy(idx, name, max_dist, &count)` | Fuzzy search |
| `symbol_index_regex(idx, pattern, &count)` | Regex search |
| `symbol_index_get_by_index(idx, index)` | Get symbol by index |

### C++ Wrapper

The `SymbolIndexV3` class provides:

- **RAII**: Automatic resource management
- **Move semantics**: Efficient transfer of ownership
- **Iterators**: Range-based for loop support (`for (auto& sym : idx)`)
- **Search methods**: find(), find_prefix(), glob(), fuzzy(), regex()
- **Properties**: count(), file_count(), valid()

### Search Types

All adapters support:
- **Exact match**: Find by exact symbol name
- **Prefix**: Find symbols starting with prefix
- **Glob**: Unix-style wildcards (`*`, `?`)
- **Fuzzy**: Edit distance search (Levenshtein)
- **Regex**: POSIX extended regular expressions

### Symbol Information

All adapters return:
- `name` - Symbol name
- `kind` - Symbol type (function, class, etc.)
- `file` - Source file path
- `line` - Line number
- `signature` - Function signature (if available)
- `code_snippet` - Source code snippet (if available)
- `context_json` - Additional context (if available)

## Symbol Kinds

| Value | Name | Description |
|-------|------|-------------|
| 0 | function | Function or method |
| 1 | class | Class definition |
| 2 | struct | Struct definition |
| 3 | enum | Enum definition |
| 4 | namespace | Namespace |
| 5 | typedef | Type definition |
| 6 | variable | Variable |
| 7 | macro | Preprocessor macro |

## Performance

- **Lookup speed**: O(n) linear search (safe for unsorted indices)
- **Memory**: Memory-mapped files, minimal RAM usage
- **Shared library**: ~22KB compiled size
- **C++ iterators**: Lazy evaluation, minimal overhead

## Testing

```bash
# Run C test
./symbol_index_v3_test test_index.bin test_func

# Run comprehensive C++ test
LD_LIBRARY_PATH=. ./test_symbol_index_cpp test_index.bin

# Run LuaJIT test
luajit test_symbol_index_lua.lua test_index.bin test_func
```

All tests include:
- Iterator/range-based for loops
- Single symbol lookup
- Prefix search
- Glob pattern matching
- Fuzzy search
- Regex search

## License

Same as the main project (MIT License)
