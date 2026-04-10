/* Cache Layer - LRU Cache with Hash Table Index
 * Provides O(1) lookup with memory limit enforcement
 */

#ifndef CACHE_H
#define CACHE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque cache handle */
typedef struct Cache Cache;

/* Cache entry structure */
typedef struct {
    const char *key;           /* Key (owned by cache) */
    void *value;               /* Value pointer (owned by caller) */
    size_t value_size;         /* Size of value for memory tracking */
    uint64_t access_count;     /* For LRU tracking */
} CacheEntry;

/* Callback for value cleanup when evicted */
typedef void (*CacheFreeFunc)(void *value);

/*===========================
 * Cache Management
 *==========================*/

/* Create a new cache with memory limit
 * Parameters:
 *   max_memory_bytes - Maximum memory limit (0 = unlimited)
 *   free_func        - Callback to free values when evicted (can be NULL)
 * Returns: cache handle on success, NULL on failure
 */
Cache *cache_create(size_t max_memory_bytes, CacheFreeFunc free_func);

/* Destroy cache and free all resources
 * Note: This will call free_func for all remaining entries
 */
void cache_destroy(Cache *cache);

/* Get current memory usage */
size_t cache_memory_used(const Cache *cache);

/* Get number of entries */
size_t cache_entry_count(const Cache *cache);

/*===========================
 * Cache Operations
 *==========================*/

/* Insert or update a cache entry
 * Parameters:
 *   cache       - cache handle
 *   key         - key string (will be copied)
 *   value       - value pointer
 *   value_size  - size for memory tracking
 * Returns: true on success, false on failure
 * Note: If key exists, old value is freed via free_func
 */
bool cache_put(Cache *cache, const char *key, void *value, size_t value_size);

/* Lookup a cache entry (updates LRU order)
 * Parameters:
 *   cache - cache handle
 *   key   - key to lookup
 * Returns: value pointer on success, NULL if not found
 * Note: Returned pointer remains valid until entry is evicted
 */
void *cache_get(Cache *cache, const char *key);

/* Check if key exists without updating LRU */
bool cache_contains(const Cache *cache, const char *key);

/* Remove a cache entry
 * Returns: true if found and removed, false otherwise
 * Note: Value is freed via free_func
 */
bool cache_remove(Cache *cache, const char *key);

/* Clear all entries */
void cache_clear(Cache *cache);

/*===========================
 * Cache Statistics
 *==========================*/

typedef struct {
    size_t hits;               /* Number of cache hits */
    size_t misses;             /* Number of cache misses */
    size_t evictions;          /* Number of evictions */
    size_t memory_used;        /* Current memory usage */
    size_t max_memory;         /* Memory limit */
    size_t entry_count;        /* Number of entries */
} CacheStats;

/* Get cache statistics */
void cache_get_stats(const Cache *cache, CacheStats *stats);

/* Reset statistics */
void cache_reset_stats(Cache *cache);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CACHE_H */
