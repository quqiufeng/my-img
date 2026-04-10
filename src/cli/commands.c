#include "cli.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Include the symbol index API */
#include "../../symbol_index_v3.h"

/* External JSON writer functions */
extern char* json_writer_init(void);
extern void json_writer_free(char *json);
extern char* json_writer_begin_object(void);
extern void json_writer_end_object(char *json);
extern char* json_writer_add_string(char *json, const char *key, const char *value);
extern char* json_writer_add_int(char *json, const char *key, int value);
extern char* json_writer_add_bool(char *json, const char *key, int value);
extern char* json_writer_begin_array(char *json, const char *key);
extern void json_writer_end_array(char *json);

/* Print symbol result in text format */
static void print_symbol_text(const SymbolResult *res, int idx, 
                               const cli_config_t *config) {
    (void)idx; /* Unused for now */
    
    printf("Symbol: %s\n", res->name ? res->name : "(null)");
    printf("  Kind: %s\n", symbol_kind_name(res->kind));
    printf("  File: %s:%u\n", 
           res->file ? res->file : "unknown", 
           res->line);
    
    if (res->signature) {
        printf("  Signature: %s\n", res->signature);
    }
    
    if (config->include_code && res->code_snippet) {
        printf("  Code:\n%s\n", res->code_snippet);
    }
    
    if (config->include_context && res->context_json) {
        printf("  Context: %s\n", res->context_json);
    }
    
    printf("\n");
}

/* Print symbol result in JSON format */
static void print_symbol_json(const SymbolResult *res, int idx, 
                               int is_last, char **json) {
    (void)idx; /* Unused for now */
    
    *json = json_writer_begin_object();
    *json = json_writer_add_string(*json, "name", res->name);
    *json = json_writer_add_string(*json, "kind", symbol_kind_name(res->kind));
    *json = json_writer_add_string(*json, "file", res->file);
    *json = json_writer_add_int(*json, "line", res->line);
    
    if (res->signature) {
        *json = json_writer_add_string(*json, "signature", res->signature);
    }
    
    if (res->code_snippet) {
        *json = json_writer_add_string(*json, "code", res->code_snippet);
    }
    
    if (res->context_json) {
        *json = json_writer_add_string(*json, "context", res->context_json);
    }
    
    json_writer_end_object(*json);
    
    if (!is_last) {
        printf(",\n");
    }
}

/* Build index from source directory */
int cmd_build(const cli_config_t *config) {
    printf("Building index...\n");
    printf("Source: %s\n", config->input_path);
    printf("Output: %s\n", config->output_path ? config->output_path : "index.bin");
    
    if (config->verbose) {
        printf("Scanning source files...\n");
    }
    
    /* 
     * Placeholder for actual index building logic.
     * In a real implementation, this would:
     * 1. Recursively scan the source directory
     * 2. Parse C/C++ files to extract symbols
     * 3. Build the symbol index structure
     * 4. Write to the output file
     */
    
    printf("Index build complete (placeholder implementation)\n");
    
    if (config->json_output) {
        printf("{\"status\": \"success\", \"output\": \"%s\"}\n",
               config->output_path ? config->output_path : "index.bin");
    }
    
    return 0;
}

/* Query symbols by exact name */
int cmd_query(const cli_config_t *config) {
    SymbolIndex *idx = symbol_index_open(config->index_file);
    if (!idx) {
        fprintf(stderr, "Error: Failed to open index '%s'\n", config->index_file);
        return 1;
    }
    
    if (config->verbose) {
        printf("Querying symbol: %s\n", config->symbol_name);
    }
    
    SymbolResult *res = symbol_index_find(idx, config->symbol_name);
    
    if (config->json_output) {
        printf("{\n");
        printf("  \"query\": \"%s\",\n", config->symbol_name);
        printf("  \"results\": ");
        
        if (res) {
            printf("[\n");
            char *json = NULL;
            print_symbol_json(res, 0, 1, &json);
            printf("%s", json ? json : "");
            json_writer_free(json);
            printf("\n  ]");
        } else {
            printf("[]");
        }
        
        printf("\n}\n");
    } else {
        if (res) {
            print_symbol_text(res, 0, config);
        } else {
            printf("Symbol '%s' not found\n", config->symbol_name);
        }
    }
    
    if (res) {
        symbol_result_free(res);
    }
    
    symbol_index_close(idx);
    return 0;
}

/* Search symbols with various options */
int cmd_search(const cli_config_t *config) {
    SymbolIndex *idx = symbol_index_open(config->index_file);
    if (!idx) {
        fprintf(stderr, "Error: Failed to open index '%s'\n", config->index_file);
        return 1;
    }
    
    if (config->verbose) {
        printf("Searching with pattern: %s (type: %d)\n", 
               config->pattern, config->search_type);
    }
    
    SymbolResult *results = NULL;
    int count = 0;
    
    /* Perform search based on type */
    switch (config->search_type) {
        case SEARCH_PREFIX:
            results = symbol_index_find_prefix(idx, config->pattern, &count);
            break;
        case SEARCH_GLOB:
            results = symbol_index_glob(idx, config->pattern, &count);
            break;
        case SEARCH_FUZZY:
            results = symbol_index_fuzzy(idx, config->pattern, 
                                         config->max_distance, &count);
            break;
        case SEARCH_REGEX:
            results = symbol_index_regex(idx, config->pattern, &count);
            break;
        case SEARCH_EXACT:
        default:
            results = symbol_index_find(idx, config->pattern);
            if (results) count = 1;
            break;
    }
    
    /* Apply limit if specified */
    if (config->limit > 0 && count > config->limit) {
        count = config->limit;
    }
    
    if (config->json_output) {
        printf("{\n");
        printf("  \"pattern\": \"%s\",\n", config->pattern);
        printf("  \"search_type\": \"%s\",\n",
               config->search_type == SEARCH_PREFIX ? "prefix" :
               config->search_type == SEARCH_GLOB ? "glob" :
               config->search_type == SEARCH_FUZZY ? "fuzzy" :
               config->search_type == SEARCH_REGEX ? "regex" : "exact");
        printf("  \"count\": %d,\n", count);
        printf("  \"results\": ");
        
        if (results && count > 0) {
            printf("[\n");
            for (int i = 0; i < count; i++) {
                char *json = NULL;
                print_symbol_json(&results[i], i, i == count - 1, &json);
                printf("%s", json ? json : "");
                json_writer_free(json);
                if (i < count - 1) {
                    printf(",\n");
                }
            }
            printf("\n  ]");
        } else {
            printf("[]");
        }
        
        printf("\n}\n");
    } else {
        if (results && count > 0) {
            printf("Found %d symbol(s) matching '%s':\n\n", count, config->pattern);
            for (int i = 0; i < count; i++) {
                print_symbol_text(&results[i], i, config);
            }
        } else {
            printf("No symbols found matching '%s'\n", config->pattern);
        }
    }
    
    if (results) {
        if (count == 1 && config->search_type == SEARCH_EXACT) {
            symbol_result_free(results);
        } else {
            symbol_results_free(results, count);
        }
    }
    
    symbol_index_close(idx);
    return 0;
}

/* Show index statistics */
int cmd_stats(const cli_config_t *config) {
    SymbolIndex *idx = symbol_index_open(config->index_file);
    if (!idx) {
        fprintf(stderr, "Error: Failed to open index '%s'\n", config->index_file);
        return 1;
    }
    
    uint32_t symbol_count = symbol_index_count(idx);
    uint32_t file_count = symbol_index_file_count(idx);
    
    /* Count symbols by kind */
    int kind_counts[KIND_MAX] = {0};
    for (uint32_t i = 0; i < symbol_count && i < 10000; i++) {
        SymbolResult *res = symbol_index_get_by_index(idx, i);
        if (res) {
            if (res->kind < KIND_MAX) {
                kind_counts[res->kind]++;
            }
            symbol_result_free(res);
        }
    }
    
    if (config->json_output) {
        printf("{\n");
        printf("  \"index_file\": \"%s\",\n", config->index_file);
        printf("  \"total_symbols\": %u,\n", symbol_count);
        printf("  \"total_files\": %u,\n", file_count);
        printf("  \"symbols_by_kind\": {\n");
        
        for (int i = 0; i < KIND_MAX; i++) {
            printf("    \"%s\": %d%s\n",
                   symbol_kind_name(i),
                   kind_counts[i],
                   i < KIND_MAX - 1 ? "," : "");
        }
        
        printf("  }\n");
        printf("}\n");
    } else {
        printf("Index Statistics\n");
        printf("================\n");
        printf("Index file:   %s\n", config->index_file);
        printf("Total symbols: %u\n", symbol_count);
        printf("Total files:   %u\n", file_count);
        printf("\nSymbols by kind:\n");
        
        for (int i = 0; i < KIND_MAX; i++) {
            if (kind_counts[i] > 0) {
                printf("  %s: %d\n", symbol_kind_name(i), kind_counts[i]);
            }
        }
    }
    
    symbol_index_close(idx);
    return 0;
}
