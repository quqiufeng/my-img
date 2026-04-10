/* Index Writer - Build symbol index from source code
 * Supports parsing C/C++ files and generating binary index files
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#define SYMBOL_INDEX_MAGIC 0x53594458  /* "SYDX" */
#define SYMBOL_INDEX_VERSION 3
#define MAX_SYMBOLS 100000
#define MAX_STRING_TABLE_SIZE (16 * 1024 * 1024)

/* Symbol kinds - must match symbol_index_v3.h */
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

/* Symbol structure for building */
typedef struct {
    char *name;
    char *signature;
    char *file;
    uint32_t line;
    uint32_t kind;
    char *code_snippet;
    char *context_json;
} BuildSymbol;

/* Index builder context */
typedef struct {
    BuildSymbol *symbols;
    int symbol_count;
    int symbol_capacity;
    char *string_table;
    size_t string_table_size;
    size_t string_table_capacity;
    char **files;
    int file_count;
    int file_capacity;
} IndexBuilder;

/* File header - V3 format (128 bytes) */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t num_symbols;
    uint32_t num_files;
    uint32_t reserved0;
    uint32_t string_table_size;
    uint32_t string_table_offset;
    uint32_t symbols_offset;
    uint32_t reserved1;
    uint32_t code_table_size;
    uint32_t code_table_offset;
    uint32_t context_table_size;
    uint32_t context_table_offset;
    uint32_t reserved2[19];
} IndexHeaderV3;

/* Symbol - V3 format (80 bytes) */
typedef struct {
    uint32_t name_offset;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t signature_offset;
    uint32_t reserved2;
    uint32_t file_offset;
    uint32_t reserved3[2];
    uint32_t line;
    uint32_t reserved4[2];
    uint32_t code_offset;
    uint32_t code_length;
    uint32_t context_offset;
    uint32_t context_length;
    uint32_t reserved5[4];
    uint32_t kind;
} SymbolV3;

/* Initialize index builder */
IndexBuilder* index_writer_init(void) {
    IndexBuilder *builder = calloc(1, sizeof(IndexBuilder));
    if (!builder) return NULL;
    
    builder->symbol_capacity = 1024;
    builder->symbols = calloc(builder->symbol_capacity, sizeof(BuildSymbol));
    if (!builder->symbols) {
        free(builder);
        return NULL;
    }
    
    builder->string_table_capacity = 65536;
    builder->string_table = calloc(1, builder->string_table_capacity);
    if (!builder->string_table) {
        free(builder->symbols);
        free(builder);
        return NULL;
    }
    builder->string_table[0] = '\0';
    builder->string_table_size = 1;
    
    builder->file_capacity = 256;
    builder->files = calloc(builder->file_capacity, sizeof(char*));
    if (!builder->files) {
        free(builder->string_table);
        free(builder->symbols);
        free(builder);
        return NULL;
    }
    
    return builder;
}

/* Free index builder */
void index_writer_free(IndexBuilder *builder) {
    if (!builder) return;
    
    for (int i = 0; i < builder->symbol_count; i++) {
        free(builder->symbols[i].name);
        free(builder->symbols[i].signature);
        free(builder->symbols[i].file);
        free(builder->symbols[i].code_snippet);
        free(builder->symbols[i].context_json);
    }
    free(builder->symbols);
    
    for (int i = 0; i < builder->file_count; i++) {
        free(builder->files[i]);
    }
    free(builder->files);
    
    free(builder->string_table);
    free(builder);
}

/* Add string to string table, return offset */
static uint32_t add_string(IndexBuilder *builder, const char *str) {
    if (!str || !*str) return 0;
    
    size_t len = strlen(str) + 1;
    
    /* Check for existing string (simple linear search) */
    for (size_t i = 1; i < builder->string_table_size; i++) {
        if (strcmp(builder->string_table + i, str) == 0) {
            return (uint32_t)i;
        }
        i += strlen(builder->string_table + i);
    }
    
    /* Ensure capacity */
    if (builder->string_table_size + len > builder->string_table_capacity) {
        size_t new_cap = builder->string_table_capacity * 2;
        char *new_table = realloc(builder->string_table, new_cap);
        if (!new_table) return 0;
        builder->string_table = new_table;
        builder->string_table_capacity = new_cap;
    }
    
    uint32_t offset = (uint32_t)builder->string_table_size;
    memcpy(builder->string_table + offset, str, len);
    builder->string_table_size += len;
    
    return offset;
}

/* Add symbol to builder */
int index_writer_add_symbol(IndexBuilder *builder, const char *name,
                            const char *signature, const char *file,
                            uint32_t line, uint32_t kind,
                            const char *code, const char *context) {
    if (!builder || !name) return -1;
    
    /* Expand capacity if needed */
    if (builder->symbol_count >= builder->symbol_capacity) {
        int new_cap = builder->symbol_capacity * 2;
        BuildSymbol *new_symbols = realloc(builder->symbols, 
                                           new_cap * sizeof(BuildSymbol));
        if (!new_symbols) return -1;
        builder->symbols = new_symbols;
        builder->symbol_capacity = new_cap;
    }
    
    BuildSymbol *sym = &builder->symbols[builder->symbol_count];
    sym->name = strdup(name);
    sym->signature = signature ? strdup(signature) : NULL;
    sym->file = file ? strdup(file) : NULL;
    sym->line = line;
    sym->kind = kind;
    sym->code_snippet = code ? strdup(code) : NULL;
    sym->context_json = context ? strdup(context) : NULL;
    
    builder->symbol_count++;
    return 0;
}

/* Add file to file list */
static int add_file(IndexBuilder *builder, const char *file) {
    if (!builder || !file) return -1;
    
    /* Check for duplicates */
    for (int i = 0; i < builder->file_count; i++) {
        if (strcmp(builder->files[i], file) == 0) {
            return i;
        }
    }
    
    if (builder->file_count >= builder->file_capacity) {
        int new_cap = builder->file_capacity * 2;
        char **new_files = realloc(builder->files, new_cap * sizeof(char*));
        if (!new_files) return -1;
        builder->files = new_files;
        builder->file_capacity = new_cap;
    }
    
    builder->files[builder->file_count] = strdup(file);
    return builder->file_count++;
}

/* Write index to file */
int index_writer_save(IndexBuilder *builder, const char *path) {
    if (!builder || !path) return -1;
    
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    
    /* Calculate offsets */
    size_t header_size = sizeof(IndexHeaderV3);
    size_t symbols_size = builder->symbol_count * sizeof(SymbolV3);
    size_t code_table_size = 0;
    size_t context_table_size = 0;
    
    /* Calculate code and context table sizes */
    for (int i = 0; i < builder->symbol_count; i++) {
        if (builder->symbols[i].code_snippet) {
            code_table_size += strlen(builder->symbols[i].code_snippet) + 1;
        }
        if (builder->symbols[i].context_json) {
            context_table_size += strlen(builder->symbols[i].context_json) + 1;
        }
    }
    
    /* Build file */
    size_t symbols_offset = header_size;
    size_t string_table_offset = symbols_offset + symbols_size;
    size_t code_table_offset = string_table_offset + builder->string_table_size;
    size_t context_table_offset = code_table_offset + code_table_size;
    
    /* Write header */
    IndexHeaderV3 header = {0};
    header.magic = SYMBOL_INDEX_MAGIC;
    header.version = SYMBOL_INDEX_VERSION;
    header.num_symbols = builder->symbol_count;
    header.num_files = builder->file_count;
    header.string_table_size = builder->string_table_size;
    header.string_table_offset = string_table_offset;
    header.symbols_offset = symbols_offset;
    header.code_table_size = code_table_size;
    header.code_table_offset = code_table_size > 0 ? code_table_offset : 0;
    header.context_table_size = context_table_size;
    header.context_table_offset = context_table_size > 0 ? context_table_offset : 0;
    
    fwrite(&header, sizeof(header), 1, fp);
    
    /* Write symbols */
    fseek(fp, symbols_offset, SEEK_SET);
    for (int i = 0; i < builder->symbol_count; i++) {
        SymbolV3 sym = {0};
        BuildSymbol *bs = &builder->symbols[i];
        
        sym.name_offset = add_string(builder, bs->name);
        sym.signature_offset = bs->signature ? add_string(builder, bs->signature) : 0;
        sym.file_offset = bs->file ? add_string(builder, bs->file) : 0;
        sym.line = bs->line;
        sym.kind = bs->kind;
        
        /* Code and context will be written separately */
        fwrite(&sym, sizeof(sym), 1, fp);
    }
    
    /* Write string table */
    fseek(fp, string_table_offset, SEEK_SET);
    fwrite(builder->string_table, 1, builder->string_table_size, fp);
    
    /* Write code table */
    if (code_table_size > 0) {
        fseek(fp, code_table_offset, SEEK_SET);
        for (int i = 0; i < builder->symbol_count; i++) {
            if (builder->symbols[i].code_snippet) {
                fwrite(builder->symbols[i].code_snippet, 1,
                       strlen(builder->symbols[i].code_snippet) + 1, fp);
            }
        }
    }
    
    /* Write context table */
    if (context_table_size > 0) {
        fseek(fp, context_table_offset, SEEK_SET);
        for (int i = 0; i < builder->symbol_count; i++) {
            if (builder->symbols[i].context_json) {
                fwrite(builder->symbols[i].context_json, 1,
                       strlen(builder->symbols[i].context_json) + 1, fp);
            }
        }
    }
    
    fclose(fp);
    return 0;
}

/* Check if file is a C/C++ source or header file */
static bool is_source_file(const char *path) {
    const char *ext = strrchr(path, '.');
    if (!ext) return false;
    
    return (strcmp(ext, ".c") == 0 ||
            strcmp(ext, ".h") == 0 ||
            strcmp(ext, ".cpp") == 0 ||
            strcmp(ext, ".hpp") == 0 ||
            strcmp(ext, ".cc") == 0 ||
            strcmp(ext, ".cxx") == 0);
}

/* Recursively scan directory for source files */
int index_writer_scan_directory(IndexBuilder *builder, const char *path) {
    if (!builder || !path) return -1;
    
    DIR *dir = opendir(path);
    if (!dir) return -1;
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        
        char full_path[4096];
        snprintf(full_path, sizeof(full_path), "%s/%s", path, entry->d_name);
        
        struct stat st;
        if (stat(full_path, &st) != 0) continue;
        
        if (S_ISDIR(st.st_mode)) {
            /* Recursively scan subdirectory */
            index_writer_scan_directory(builder, full_path);
        } else if (S_ISREG(st.st_mode) && is_source_file(full_path)) {
            /* Process source file */
            add_file(builder, full_path);
        }
    }
    
    closedir(dir);
    return 0;
}

/* Get number of files scanned */
int index_writer_get_file_count(const IndexBuilder *builder) {
    return builder ? builder->file_count : 0;
}

/* Get number of symbols */
int index_writer_get_symbol_count(const IndexBuilder *builder) {
    return builder ? builder->symbol_count : 0;
}
