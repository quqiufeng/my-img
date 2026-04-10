#include "cli.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>

/* External command implementations */
extern int cmd_build(const cli_config_t *config);
extern int cmd_query(const cli_config_t *config);
extern int cmd_search(const cli_config_t *config);
extern int cmd_stats(const cli_config_t *config);

/* Static configuration instance */
static cli_config_t g_config = {
    .command = CMD_NONE,
    .search_type = SEARCH_EXACT,
    .input_path = NULL,
    .output_path = NULL,
    .symbol_name = NULL,
    .pattern = NULL,
    .index_file = NULL,
    .max_distance = 2,
    .verbose = false,
    .json_output = false,
    .include_code = false,
    .include_context = false,
    .limit = 0
};

void cli_print_version(void) {
    printf("Symbol Index CLI v3.0\n");
}

void cli_print_help(const char *prog_name) {
    printf("Usage: %s <command> [options]\n\n", prog_name ? prog_name : "symbol-index");
    printf("Commands:\n");
    printf("  build   - Build index from source directory\n");
    printf("  query   - Query symbols by name\n");
    printf("  search  - Search symbols with patterns\n");
    printf("  stats   - Show index statistics\n");
    printf("  help    - Show this help message\n");
    printf("\nGlobal Options:\n");
    printf("  -i, --input <path>      Input path (source dir or index file)\n");
    printf("  -o, --output <path>     Output path (for build command)\n");
    printf("  -v, --verbose           Enable verbose output\n");
    printf("  -j, --json              Output in JSON format\n");
    printf("  -h, --help              Show help\n");
    printf("\nQuery Options:\n");
    printf("  -n, --name <symbol>     Symbol name to query\n");
    printf("  -c, --code              Include code snippets\n");
    printf("  -x, --context           Include context JSON\n");
    printf("\nSearch Options:\n");
    printf("  -p, --pattern <pat>     Search pattern\n");
    printf("  -t, --type <type>       Search type: exact, prefix, glob, fuzzy, regex\n");
    printf("  -d, --distance <n>      Max edit distance for fuzzy search (default: 2)\n");
    printf("  -l, --limit <n>         Limit number of results\n");
    printf("\nExamples:\n");
    printf("  %s build -i ./src -o ./index.bin\n", prog_name ? prog_name : "symbol-index");
    printf("  %s query -i ./index.bin -n main\n", prog_name ? prog_name : "symbol-index");
    printf("  %s search -i ./index.bin -p \"foo*\" -t glob\n", prog_name ? prog_name : "symbol-index");
    printf("  %s stats -i ./index.bin\n", prog_name ? prog_name : "symbol-index");
}

static cli_command_t parse_command(const char *cmd) {
    if (!cmd) return CMD_NONE;
    if (strcmp(cmd, "build") == 0) return CMD_BUILD;
    if (strcmp(cmd, "query") == 0) return CMD_QUERY;
    if (strcmp(cmd, "search") == 0) return CMD_SEARCH;
    if (strcmp(cmd, "stats") == 0) return CMD_STATS;
    if (strcmp(cmd, "help") == 0 || strcmp(cmd, "--help") == 0) return CMD_HELP;
    return CMD_NONE;
}

static search_type_t parse_search_type(const char *type) {
    if (!type) return SEARCH_EXACT;
    if (strcmp(type, "exact") == 0) return SEARCH_EXACT;
    if (strcmp(type, "prefix") == 0) return SEARCH_PREFIX;
    if (strcmp(type, "glob") == 0) return SEARCH_GLOB;
    if (strcmp(type, "fuzzy") == 0) return SEARCH_FUZZY;
    if (strcmp(type, "regex") == 0) return SEARCH_REGEX;
    return SEARCH_EXACT;
}

int cli_init(int argc, char **argv) {
    if (argc < 2) {
        cli_print_help(argv[0]);
        return 1;
    }
    
    /* First argument is command */
    g_config.command = parse_command(argv[1]);
    
    if (g_config.command == CMD_NONE) {
        fprintf(stderr, "Error: Unknown command '%s'\n", argv[1]);
        cli_print_help(argv[0]);
        return 1;
    }
    
    if (g_config.command == CMD_HELP) {
        cli_print_help(argv[0]);
        return 0;
    }
    
    /* Parse options */
    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"name", required_argument, 0, 'n'},
        {"pattern", required_argument, 0, 'p'},
        {"type", required_argument, 0, 't'},
        {"distance", required_argument, 0, 'd'},
        {"limit", required_argument, 0, 'l'},
        {"verbose", no_argument, 0, 'v'},
        {"json", no_argument, 0, 'j'},
        {"code", no_argument, 0, 'c'},
        {"context", no_argument, 0, 'x'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    /* Adjust argc/argv to skip command name */
    int cmd_argc = argc - 1;
    char **cmd_argv = argv + 1;
    
    int opt;
    int option_index = 0;
    
    while ((opt = getopt_long(cmd_argc, cmd_argv, "i:o:n:p:t:d:l:vjcxh", 
                               long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                g_config.input_path = strdup(optarg);
                break;
            case 'o':
                g_config.output_path = strdup(optarg);
                break;
            case 'n':
                g_config.symbol_name = strdup(optarg);
                break;
            case 'p':
                g_config.pattern = strdup(optarg);
                break;
            case 't':
                g_config.search_type = parse_search_type(optarg);
                break;
            case 'd':
                g_config.max_distance = atoi(optarg);
                break;
            case 'l':
                g_config.limit = atoi(optarg);
                break;
            case 'v':
                g_config.verbose = true;
                break;
            case 'j':
                g_config.json_output = true;
                break;
            case 'c':
                g_config.include_code = true;
                break;
            case 'x':
                g_config.include_context = true;
                break;
            case 'h':
                cli_print_help(argv[0]);
                return 0;
            case '?':
                return 1;
            default:
                fprintf(stderr, "Error: Unknown option\n");
                return 1;
        }
    }
    
    /* Set index file (defaults to input path) */
    if (g_config.input_path) {
        g_config.index_file = strdup(g_config.input_path);
    }
    
    return 0;
}

const cli_config_t* cli_get_config(void) {
    return &g_config;
}

int cli_run(void) {
    const cli_config_t *config = cli_get_config();
    
    if (config->command == CMD_NONE || config->command == CMD_HELP) {
        return 0;
    }
    
    /* Validate required arguments */
    if (config->command != CMD_BUILD && !config->index_file) {
        fprintf(stderr, "Error: -i <index_file> is required for this command\n");
        return 1;
    }
    
    if (config->command == CMD_BUILD && !config->input_path) {
        fprintf(stderr, "Error: -i <source_dir> is required for build command\n");
        return 1;
    }
    
    if (config->command == CMD_QUERY && !config->symbol_name) {
        fprintf(stderr, "Error: -n <symbol_name> is required for query command\n");
        return 1;
    }
    
    if (config->command == CMD_SEARCH && !config->pattern) {
        fprintf(stderr, "Error: -p <pattern> is required for search command\n");
        return 1;
    }
    
    /* Dispatch to command handler */
    switch (config->command) {
        case CMD_BUILD:
            return cmd_build(config);
        case CMD_QUERY:
            return cmd_query(config);
        case CMD_SEARCH:
            return cmd_search(config);
        case CMD_STATS:
            return cmd_stats(config);
        default:
            fprintf(stderr, "Error: Unknown command\n");
            return 1;
    }
}

void cli_shutdown(void) {
    /* Free allocated strings */
    if (g_config.input_path) {
        free(g_config.input_path);
        g_config.input_path = NULL;
    }
    if (g_config.output_path) {
        free(g_config.output_path);
        g_config.output_path = NULL;
    }
    if (g_config.symbol_name) {
        free(g_config.symbol_name);
        g_config.symbol_name = NULL;
    }
    if (g_config.pattern) {
        free(g_config.pattern);
        g_config.pattern = NULL;
    }
    if (g_config.index_file) {
        free(g_config.index_file);
        g_config.index_file = NULL;
    }
}
