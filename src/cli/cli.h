#ifndef CLI_H
#define CLI_H

#include <stddef.h>
#include <stdbool.h>

/* Command type enumeration */
typedef enum {
    CMD_NONE = 0,
    CMD_BUILD,
    CMD_QUERY,
    CMD_SEARCH,
    CMD_STATS,
    CMD_HELP
} cli_command_t;

/* Search type for query command */
typedef enum {
    SEARCH_EXACT = 0,
    SEARCH_PREFIX,
    SEARCH_GLOB,
    SEARCH_FUZZY,
    SEARCH_REGEX
} search_type_t;

/* CLI configuration structure */
typedef struct {
    cli_command_t command;
    search_type_t search_type;
    char *input_path;
    char *output_path;
    char *symbol_name;
    char *pattern;
    char *index_file;
    int max_distance;
    bool verbose;
    bool json_output;
    bool include_code;
    bool include_context;
    int limit;
} cli_config_t;

/* Command handler function type */
typedef int (*command_handler_t)(const cli_config_t *config);

/*===========================
 * CLI Lifecycle Functions
 *==========================*/

/* Initialize CLI with arguments
 * Returns: 0 on success, non-zero on error
 */
int cli_init(int argc, char **argv);

/* Run the CLI - parse arguments and execute command
 * Returns: exit code (0 for success)
 */
int cli_run(void);

/* Shutdown CLI and free resources */
void cli_shutdown(void);

/*===========================
 * Configuration Access
 *==========================*/

/* Get current CLI configuration */
const cli_config_t* cli_get_config(void);

/* Print help message */
void cli_print_help(const char *prog_name);

/* Print version information */
void cli_print_version(void);

#endif /* CLI_H */
