# Symbol Index V3 Adapter Layer

Clean C/C++/LuaJIT adapter layer for the Symbol Index V3 library, providing high-performance code symbol lookup with multiple language bindings.

## Overview

The adapter layer provides:
- **C API**: Core low-level interface
- **C++ Wrapper**: Modern RAII-based C++ interface
- **LuaJIT FFI**: High-performance Lua bindings

All adapters use the same compiled shared library (`libsymbol_index_v3.so`).

## Files

| File | Description |
|------|-------------|
| `symbol_index_v3.c` | Core C implementation with adapter functions |
| `symbol_index_v3.h` | C/C++ header file (clean interface) |
| `symbol_index_v3.hpp` | Modern C++ wrapper class |
| `symbol_index_v3_ffi.lua` | LuaJIT FFI bindings |
| `libsymbol_index_v3.so` | Compiled shared library |

## Building

```bash
# Compile shared library
gcc -O2 -fPIC -shared -o libsymbol_index_v3.so symbol_index_v3.c

# Compile C++ test
g++ -std=c++17 -I. test_symbol_index_cpp.cpp -L. -lsymbol_index_v3 -o test_symbol_index_cpp

# Run LuaJIT test (no compilation needed)
luajit test_symbol_index_lua.lua <index.bin> <symbol_name>
```

## Usage Examples

### C API

```c
#include "symbol_index_v3.h"

SymbolIndex *idx = symbol_index_open("project.bin");
if (idx) {
    SymbolResultPy *res = symbol_index_find_py(idx, "my_function");
    if (res) {
        printf("Found: %s at %s:%d\n",
               symbol_result_name(res),
               symbol_result_file(res),
               symbol_result_line(res));
        symbol_result_py_free(res);
    }
    symbol_index_close(idx);
}
```

### C++ API

```cpp
#include "symbol_index_v3.hpp"

code_index::SymbolIndexV3 idx("project.bin");
auto result = idx.find("my_function");
if (result) {
    std::cout << result->name << " at " 
              << result->file << ":" << result->line;
}
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

### Core Functions

- `symbol_index_open(path)` - Open index file
- `symbol_index_close(idx)` - Close index
- `symbol_index_count(idx)` - Get symbol count
- `symbol_index_find_py(idx, name)` - Find symbol by name

### Search Types

- Exact match: `symbol_index_find()` / `idx:find()`
- Prefix search: `symbol_index_find_prefix()`
- Glob patterns: `symbol_index_glob()`
- Fuzzy search: `symbol_index_fuzzy()`
- Regex: `symbol_index_regex()`

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

## Testing

```bash
# Run C test
./symbol_index_v3_test test_index.bin test_func

# Run C++ test
LD_LIBRARY_PATH=. ./test_symbol_index_cpp test_index.bin test_func

# Run LuaJIT test
luajit test_symbol_index_lua.lua test_index.bin test_func
```

## License

Same as the main project (MIT License)
