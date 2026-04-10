/* Symbol Index V3 - C/C++ Header
 * Clean interface for C and C++ integration
 * 
 * Usage:
 *   C:   #include "symbol_index_v3.h"
 *   C++: #include "symbol_index_v3.hpp"
 */

#ifndef SYMBOL_INDEX_V3_H
#define SYMBOL_INDEX_V3_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Symbol kinds */
#define KIND_FUNCTION   0
#define KIND_CLASS      1
#define KIND_STRUCT     2
#define KIND_ENUM       3
#define KIND_NAMESPACE  4
#define KIND_TYPEDEF    5
#define KIND_VARIABLE   6
#define KIND_MACRO      7
#define KIND_MAX        8

/* Opaque handle types */
typedef struct SymbolIndex SymbolIndex;
typedef struct SymbolResultPy SymbolResultPy;

/* Search result structure - exposed for C++ wrapper */
typedef struct {
    void *symbol_ptr;        /* Internal symbol pointer */
    char *name;              /* Symbol name */
    char *signature;         /* Function signature */
    char *file;              /* Source file path */
    uint32_t line;           /* Line number */
    uint32_t kind;           /* Symbol kind */
    char *code_snippet;      /* Source code snippet */
    char *context_json;      /* Additional context */
} SymbolResult;

/*===========================
 * Core API - Index Management
 *==========================*/

/* Open an index file (memory-mapped)
 * Returns: handle on success, NULL on failure
 */
SymbolIndex *symbol_index_open(const char *path);

/* Close index and free resources */
void symbol_index_close(SymbolIndex *idx);

/* Get number of symbols in index */
uint32_t symbol_index_count(SymbolIndex *idx);

/* Get number of unique files in index */
uint32_t symbol_index_file_count(SymbolIndex *idx);

/* Get human-readable kind name */
const char *symbol_kind_name(int kind);

/*===========================
 * Core API - Search Functions
 *==========================*/

/* Find symbol by exact name match
 * Returns: result on success, NULL if not found
 * Note: Caller must free result with symbol_result_free()
 */
SymbolResult *symbol_index_find(SymbolIndex *idx, const char *name);

/* Get symbol by index (0-based) */
SymbolResult *symbol_index_get_by_index(SymbolIndex *idx, uint32_t index);

/* Find symbols by prefix
 * Parameters:
 *   idx    - index handle
 *   prefix - prefix to search for
 *   count  - output: number of matches
 * Returns: array of results, NULL if none found
 * Note: Caller must free results with symbol_results_free()
 */
SymbolResult *symbol_index_find_prefix(SymbolIndex *idx, const char *prefix, int *count);

/* Find symbols by glob pattern (e.g., "foo*", "*bar", "f?o")
 * Parameters:
 *   idx     - index handle
 *   pattern - glob pattern
 *   count   - output: number of matches
 * Returns: array of results, NULL if none found
 */
SymbolResult *symbol_index_glob(SymbolIndex *idx, const char *pattern, int *count);

/* Fuzzy search by edit distance
 * Parameters:
 *   idx      - index handle
 *   name     - search term
 *   max_dist - maximum edit distance (typically 1-3)
 *   count    - output: number of matches
 * Returns: array of results sorted by distance, NULL if none found
 */
SymbolResult *symbol_index_fuzzy(SymbolIndex *idx, const char *name, int max_dist, int *count);

/* Regex search (POSIX extended regex)
 * Parameters:
 *   idx     - index handle
 *   pattern - regex pattern
 *   count   - output: number of matches
 * Returns: array of results, NULL if none found or invalid pattern
 */
SymbolResult *symbol_index_regex(SymbolIndex *idx, const char *pattern, int *count);

/*===========================
 * Core API - Result Management
 *==========================*/

/* Free a single search result */
void symbol_result_free(SymbolResult *res);

/* Free array of search results */
void symbol_results_free(SymbolResult *res, int count);

/* SymbolResult field accessors for C++ wrapper and other bindings */
const char *symbol_result_get_name(const SymbolResult *res);
const char *symbol_result_get_signature(const SymbolResult *res);
const char *symbol_result_get_file(const SymbolResult *res);
uint32_t symbol_result_get_line(const SymbolResult *res);
int symbol_result_get_kind(const SymbolResult *res);
const char *symbol_result_get_code_snippet(const SymbolResult *res);
const char *symbol_result_get_context_json(const SymbolResult *res);

/*===========================
 * Python-Compatible API
 * Use these for ctypes/luaJIT FFI bindings
 *==========================*/

/* Find symbol (simplified interface for bindings) */
SymbolResultPy *symbol_index_find_py(SymbolIndex *idx, const char *name);

/* Free Python-compatible result */
void symbol_result_py_free(SymbolResultPy *res);

/* Field accessors for Python/FFI
 * These return pointers that remain valid until result is freed
 */
char *symbol_result_name(void *res);
char *symbol_result_signature(void *res);
char *symbol_result_file(void *res);
int symbol_result_line(void *res);
int symbol_result_kind(void *res);
char *symbol_result_code_snippet(void *res);
char *symbol_result_context_json(void *res);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SYMBOL_INDEX_V3_H */
