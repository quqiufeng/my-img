/* Symbol Index V3 - Full Implementation
 * Based on V2 structure, modified for V3 format (80-byte symbols with context)
 * 
 * Build: gcc -O2 -fPIC -shared -o libsymbol_index_v3.so symbol_index_v3.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <ctype.h>
#include <regex.h>

#define SYMBOL_INDEX_MAGIC 0x53594458  /* "SYDX" */
#define SYMBOL_INDEX_VERSION 3

/* Symbol kinds */
typedef enum {
    KIND_FUNCTION = 0,
    KIND_CLASS = 1,
    KIND_STRUCT = 2,
    KIND_ENUM = 3,
    KIND_NAMESPACE = 4,
    KIND_TYPEDEF = 5,
    KIND_VARIABLE = 6,
    KIND_MACRO = 7,
    KIND_MAX = 8
} SymbolKind;

/* Symbol - V3 format (80 bytes) */
typedef struct {
    uint32_t name_offset;        /* 0: Symbol name */
    uint32_t reserved0;          /* 1 */
    uint32_t reserved1;          /* 2 */
    uint32_t signature_offset;   /* 3: Function signature */
    uint32_t reserved2;          /* 4 */
    uint32_t file_offset;        /* 5: File path */
    uint32_t reserved3[2];       /* 6-7 */
    uint32_t line;               /* 8: Line number */
    uint32_t reserved4[2];       /* 9-10 */
    uint32_t code_offset;        /* 11: Offset to code */
    uint32_t code_length;        /* 12: Length of code */
    uint32_t context_offset;     /* 13: Offset to context */
    uint32_t context_length;     /* 14: Length of context */
    uint32_t reserved5[4];       /* 15-18 */
    uint32_t kind;               /* 19: Symbol kind */
} SymbolV3;

/* File header - V3 format (128 bytes) */
typedef struct {
    uint32_t magic;              /* 0 */
    uint32_t version;            /* 1 */
    uint32_t num_symbols;        /* 2 */
    uint32_t num_files;          /* 3 */
    uint32_t reserved0;          /* 4 */
    uint32_t string_table_size;  /* 5 */
    uint32_t string_table_offset;/* 6 */
    uint32_t symbols_offset;     /* 7 */
    uint32_t reserved1;          /* 8 */
    uint32_t code_table_size;    /* 9 */
    uint32_t code_table_offset;  /* 10 */
    uint32_t context_table_size; /* 11 */
    uint32_t context_table_offset;/* 12 */
    uint32_t reserved2[19];      /* 13-31 (76 bytes) */
} IndexHeaderV3;

/* Memory-mapped index */
typedef struct {
    uint8_t *data;
    size_t size;
    int fd;
    IndexHeaderV3 *header;
    char *string_table;
    char *code_table;
    char *context_table;
    SymbolV3 *symbols;
} SymbolIndex;

/* Search result */
typedef struct {
    SymbolV3 *symbol;
    char *name;
    char *signature;
    char *file;
    uint32_t line;
    uint32_t kind;
    char *code_snippet;
    char *context_json;
} SymbolResult;

/* Python wrapper struct for ctypes compatibility */
typedef struct {
    void *symbol_ptr;
    char *name;
    char *signature;
    char *file;
    int line;
    int kind;
    char *code_snippet;
    char *context_json;
} SymbolResultPy;

/* Verify struct sizes */
_Static_assert(sizeof(SymbolV3) == 80, "SymbolV3 must be 80 bytes");
_Static_assert(sizeof(IndexHeaderV3) == 128, "IndexHeaderV3 must be 128 bytes");

/* Open index from file (mmap) */
SymbolIndex *symbol_index_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }
    if ((size_t)st.st_size < sizeof(IndexHeaderV3)) { close(fd); return NULL; }

    uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { close(fd); return NULL; }

    IndexHeaderV3 *h = (IndexHeaderV3 *)data;
    if (h->magic != SYMBOL_INDEX_MAGIC || h->version != SYMBOL_INDEX_VERSION) {
        munmap(data, st.st_size);
        close(fd);
        return NULL;
    }

    SymbolIndex *idx = calloc(1, sizeof(SymbolIndex));
    if (!idx) { munmap(data, st.st_size); close(fd); return NULL; }

    idx->data = data;
    idx->size = st.st_size;
    idx->fd = fd;
    idx->header = h;
    idx->string_table = (char *)(data + h->string_table_offset);
    idx->symbols = (SymbolV3 *)(data + h->symbols_offset);
    
    if (h->code_table_offset > 0 && h->code_table_size > 0) {
        idx->code_table = (char *)(data + h->code_table_offset);
    }
    if (h->context_table_offset > 0 && h->context_table_size > 0) {
        idx->context_table = (char *)(data + h->context_table_offset);
    }

    return idx;
}

/* Close index and release mmap */
void symbol_index_close(SymbolIndex *idx) {
    if (!idx) return;
    if (idx->data) munmap(idx->data, idx->size);
    if (idx->fd >= 0) close(idx->fd);
    free(idx);
}

/* Get string from string table */
static inline char *get_string(SymbolIndex *idx, uint32_t offset) {
    if (offset == 0 || offset >= idx->header->string_table_size) return NULL;
    return idx->string_table + offset;
}

/* Get code snippet (allocates memory - caller must free) */
static char *get_code_snippet(SymbolIndex *idx, SymbolV3 *sym) {
    if (!sym || sym->code_offset == 0 || sym->code_length == 0) return NULL;
    if (!idx->code_table) return NULL;
    
    char *code = malloc(sym->code_length + 1);
    if (!code) return NULL;
    
    memcpy(code, idx->code_table + sym->code_offset, sym->code_length);
    code[sym->code_length] = '\0';
    return code;
}

/* Get context JSON (allocates memory - caller must free) */
static char *get_context_json(SymbolIndex *idx, SymbolV3 *sym) {
    if (!sym || sym->context_offset == 0 || sym->context_length == 0) return NULL;
    if (!idx->context_table) return NULL;
    
    char *context = malloc(sym->context_length + 1);
    if (!context) return NULL;
    
    memcpy(context, idx->context_table + sym->context_offset, sym->context_length);
    context[sym->context_length] = '\0';
    return context;
}

/* Allocate and fill result */
static SymbolResult *alloc_result(void) {
    return calloc(1, sizeof(SymbolResult));
}

static void fill_result(SymbolIndex *idx, SymbolV3 *sym, SymbolResult *res) {
    if (!sym || !res) return;
    res->symbol = sym;
    res->name = get_string(idx, sym->name_offset);
    res->signature = get_string(idx, sym->signature_offset);
    res->file = get_string(idx, sym->file_offset);
    res->line = sym->line;
    res->kind = sym->kind;
    res->code_snippet = get_code_snippet(idx, sym);
    res->context_json = get_context_json(idx, sym);
}

/* Find symbol by name (linear search - safe but slower) */
SymbolResult *symbol_index_find(SymbolIndex *idx, const char *name) {
    if (!idx || !name) return NULL;
    uint32_t n = idx->header->num_symbols;
    if (n == 0) return NULL;

    for (uint32_t i = 0; i < n; i++) {
        char *sname = get_string(idx, idx->symbols[i].name_offset);
        if (sname && strcmp(sname, name) == 0) {
            SymbolResult *res = alloc_result();
            fill_result(idx, &idx->symbols[i], res);
            return res;
        }
    }
    return NULL;
}

/* Prefix search */
SymbolResult *symbol_index_find_prefix(SymbolIndex *idx, const char *prefix, int *count) {
    if (!idx || !prefix || !count) return NULL;
    *count = 0;
    size_t plen = strlen(prefix);
    uint32_t n = idx->header->num_symbols;
    if (n == 0) return NULL;

    int l = 0, r = (int)n - 1, first = -1;
    while (l <= r) {
        int m = l + (r - l) / 2;
        char *sname = get_string(idx, idx->symbols[m].name_offset);
        if (!sname) { r = m - 1; continue; }
        int cmp = strncmp(sname, prefix, plen);
        if (cmp == 0) { first = m; r = m - 1; }
        else if (cmp < 0) l = m + 1; else r = m - 1;
    }
    if (first == -1) return NULL;

    int k = 0;
    for (uint32_t i = first; i < n; i++) {
        char *sname = get_string(idx, idx->symbols[i].name_offset);
        if (!sname || strncmp(sname, prefix, plen) != 0) break;
        k++;
    }
    if (k == 0) return NULL;

    SymbolResult *res = calloc(k, sizeof(SymbolResult));
    for (int i = 0; i < k; i++)
        fill_result(idx, &idx->symbols[first + i], &res[i]);
    *count = k;
    return res;
}

/* Glob pattern matching */
static int glob_match(const char *pattern, const char *name) {
    const char *p = pattern, *n = name;
    while (*p && *n) {
        if (*p == '*') {
            if (*(p+1) == '\0') return 1;
            for (const char *q = n; *q; q++) {
                if (glob_match(p + 1, q)) return 1;
            }
            return 0;
        } else if (*p == '?' || *p == *n) {
            p++; n++;
        } else return 0;
    }
    if (*p == '*') return glob_match(p + 1, n);
    return (*p == '\0' && *n == '\0');
}

SymbolResult *symbol_index_glob(SymbolIndex *idx, const char *pattern, int *count) {
    if (!idx || !pattern || !count) return NULL;
    *count = 0;
    uint32_t n = idx->header->num_symbols;

    uint32_t *matches = calloc(n, sizeof(uint32_t));
    if (!matches) return NULL;

    for (uint32_t i = 0; i < n; i++) {
        char *name = get_string(idx, idx->symbols[i].name_offset);
        if (name && glob_match(pattern, name))
            matches[(*count)++] = i;
    }
    if (*count == 0) { free(matches); return NULL; }

    SymbolResult *res = calloc(*count, sizeof(SymbolResult));
    for (uint32_t i = 0; i < (uint32_t)*count; i++)
        fill_result(idx, &idx->symbols[matches[i]], &res[i]);
    free(matches);
    return res;
}

/* Levenshtein edit distance */
static int edit_distance(const char *a, const char *b) {
    size_t la = strlen(a), lb = strlen(b);
    if (la > 32) la = 32;
    if (lb > 32) lb = 32;
    int d[33][33];
    for (size_t i = 0; i <= la; i++) d[i][0] = i;
    for (size_t j = 0; j <= lb; j++) d[0][j] = j;
    for (size_t i = 1; i <= la; i++)
        for (size_t j = 1; j <= lb; j++) {
            int cost = (a[i-1] == b[j-1]) ? 0 : 1;
            int del = d[i-1][j] + 1;
            int ins = d[i][j-1] + 1;
            int sub = d[i-1][j-1] + cost;
            d[i][j] = (del < ins) ? ((del < sub) ? del : sub) : ((ins < sub) ? ins : sub);
        }
    return d[la][lb];
}

SymbolResult *symbol_index_fuzzy(SymbolIndex *idx, const char *name, int max_dist, int *count) {
    if (!idx || !name || !count) return NULL;
    *count = 0;
    uint32_t n = idx->header->num_symbols;

    uint32_t *matches = calloc(n, sizeof(uint32_t));
    int *dists = calloc(n, sizeof(int));
    if (!matches || !dists) { free(matches); free(dists); return NULL; }

    for (uint32_t i = 0; i < n; i++) {
        char *sname = get_string(idx, idx->symbols[i].name_offset);
        if (sname) {
            int d = edit_distance(name, sname);
            if (d <= max_dist) {
                dists[*count] = d;
                matches[(*count)++] = i;
            }
        }
    }

    if (*count == 0) { free(matches); free(dists); return NULL; }

    /* Sort by edit distance (insertion sort) */
    for (int i = 1; i < *count; i++) {
        int key_d = dists[i], key_m = matches[i];
        int j = i - 1;
        while (j >= 0 && dists[j] > key_d) {
            dists[j+1] = dists[j];
            matches[j+1] = matches[j];
            j--;
        }
        dists[j+1] = key_d;
        matches[j+1] = key_m;
    }

    SymbolResult *res = calloc(*count, sizeof(SymbolResult));
    for (int i = 0; i < *count; i++)
        fill_result(idx, &idx->symbols[matches[i]], &res[i]);

    free(matches);
    free(dists);
    return res;
}

/* Regex search */
SymbolResult *symbol_index_regex(SymbolIndex *idx, const char *pattern, int *count) {
    if (!idx || !pattern || !count) return NULL;
    *count = 0;

    regex_t regex;
    if (regcomp(&regex, pattern, REG_EXTENDED | REG_NOSUB | REG_ICASE) != 0)
        return NULL;

    uint32_t n = idx->header->num_symbols;
    uint32_t *matches = calloc(n, sizeof(uint32_t));
    if (!matches) { regfree(&regex); return NULL; }

    for (uint32_t i = 0; i < n; i++) {
        char *name = get_string(idx, idx->symbols[i].name_offset);
        if (name && regexec(&regex, name, 0, NULL, 0) == 0)
            matches[(*count)++] = i;
    }

    regfree(&regex);
    if (*count == 0) { free(matches); return NULL; }

    SymbolResult *res = calloc(*count, sizeof(SymbolResult));
    for (uint32_t i = 0; i < (uint32_t)*count; i++)
        fill_result(idx, &idx->symbols[matches[i]], &res[i]);
    free(matches);
    return res;
}

/* Free result */
void symbol_result_free(SymbolResult *res) {
    if (!res) return;
    if (res->code_snippet) free(res->code_snippet);
    if (res->context_json) free(res->context_json);
    free(res);
}

void symbol_results_free(SymbolResult *res, int count) {
    if (!res) return;
    for (int i = 0; i < count; i++) {
        if (res[i].code_snippet) free(res[i].code_snippet);
        if (res[i].context_json) free(res[i].context_json);
    }
    free(res);
}

/* Getters for Python bindings */
uint32_t symbol_index_count(SymbolIndex *idx) {
    return idx ? idx->header->num_symbols : 0;
}

uint32_t symbol_index_file_count(SymbolIndex *idx) {
    return idx ? idx->header->num_files : 0;
}

const char *symbol_kind_name(int kind) {
    static const char *names[] = {
        "function", "class", "struct", "enum", 
        "namespace", "typedef", "variable", "macro"
    };
    return (kind >= 0 && kind < KIND_MAX) ? names[kind] : "unknown";
}

/* Python-compatible accessor functions */
SymbolResultPy *symbol_index_find_py(SymbolIndex *idx, const char *name) {
    SymbolResult *res = symbol_index_find(idx, name);
    if (!res) return NULL;

    SymbolResultPy *pyres = calloc(1, sizeof(SymbolResultPy));
    if (!pyres) {
        symbol_result_free(res);
        return NULL;
    }

    pyres->symbol_ptr = res->symbol;
    pyres->name = res->name;
    pyres->signature = res->signature;
    pyres->file = res->file;
    pyres->line = res->line;
    pyres->kind = res->kind;
    pyres->code_snippet = res->code_snippet;
    pyres->context_json = res->context_json;

    /* Don't free res->code_snippet and res->context_json as they're now owned by pyres */
    free(res);
    return pyres;
}

void symbol_result_py_free(SymbolResultPy *res) {
    if (!res) return;
    if (res->code_snippet) free(res->code_snippet);
    if (res->context_json) free(res->context_json);
    free(res);
}

/* Individual field accessors for Python */
char *symbol_result_name(void *res) { return res ? ((SymbolResultPy *)res)->name : NULL; }
char *symbol_result_signature(void *res) { return res ? ((SymbolResultPy *)res)->signature : NULL; }
char *symbol_result_file(void *res) { return res ? ((SymbolResultPy *)res)->file : NULL; }
int symbol_result_line(void *res) { return res ? ((SymbolResultPy *)res)->line : 0; }
int symbol_result_kind(void *res) { return res ? ((SymbolResultPy *)res)->kind : -1; }
char *symbol_result_code_snippet(void *res) { return res ? ((SymbolResultPy *)res)->code_snippet : NULL; }
char *symbol_result_context_json(void *res) { return res ? ((SymbolResultPy *)res)->context_json : NULL; }

/* Test main */
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <index.bin> <symbol_name>\n", argv[0]);
        return 1;
    }

    SymbolIndex *idx = symbol_index_open(argv[1]);
    if (!idx) {
        fprintf(stderr, "Failed to open index: %s\n", argv[1]);
        return 1;
    }

    printf("\nSearching for: %s\n", argv[2]);
    printf("------------------------------------------------\n");

    SymbolResult *res = symbol_index_find(idx, argv[2]);
    if (res) {
        printf("Found: %s\n", res->name ? res->name : "(null)");
        printf("Kind: %s (%d)\n", symbol_kind_name(res->kind), res->kind);
        printf("File: %s:%u\n", res->file ? res->file : "unknown", res->line);
        if (res->signature) printf("Signature: %s\n", res->signature);
        if (res->code_snippet) {
            printf("\nCode (%zu bytes):\n", strlen(res->code_snippet));
            printf("%.500s%s\n", res->code_snippet, 
                   strlen(res->code_snippet) > 500 ? "..." : "");
        }
        if (res->context_json) {
            printf("\nContext (%zu bytes):\n", strlen(res->context_json));
            printf("%.500s%s\n", res->context_json,
                   strlen(res->context_json) > 500 ? "..." : "");
        }
        symbol_result_free(res);
    } else {
        printf("Symbol not found: %s\n", argv[2]);
    }

    symbol_index_close(idx);
    return 0;
}
