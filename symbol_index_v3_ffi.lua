-- Symbol Index V3 - LuaJIT FFI Bindings
-- High-performance Lua interface to symbol index library
--
-- Usage:
--   local idx = require("symbol_index_v3_ffi").open("/path/to/index.bin")
--   local result = idx:find("my_function")
--   if result then
--       print(result.name .. " at " .. result.file .. ":" .. result.line)
--   end
--   idx:close()

local ffi = require("ffi")
local C = ffi.C

-- C declarations
ffi.cdef[[
/* Opaque handle types */
typedef struct SymbolIndex SymbolIndex;
typedef struct SymbolResultPy SymbolResultPy;

/* Core functions */
SymbolIndex *symbol_index_open(const char *path);
void symbol_index_close(SymbolIndex *idx);
uint32_t symbol_index_count(SymbolIndex *idx);
uint32_t symbol_index_file_count(SymbolIndex *idx);
const char *symbol_kind_name(int kind);

/* Python-compatible search (simpler interface) */
SymbolResultPy *symbol_index_find_py(SymbolIndex *idx, const char *name);
void symbol_result_py_free(SymbolResultPy *res);

/* Field accessors */
char *symbol_result_name(void *res);
char *symbol_result_signature(void *res);
char *symbol_result_file(void *res);
int symbol_result_line(void *res);
int symbol_result_kind(void *res);
char *symbol_result_code_snippet(void *res);
char *symbol_result_context_json(void *res);
]]

-- Load library
local lib = ffi.load("./libsymbol_index_v3.so")

-- Symbol kinds table
local KIND_NAMES = {
    [0] = "function",
    [1] = "class",
    [2] = "struct",
    [3] = "enum",
    [4] = "namespace",
    [5] = "typedef",
    [6] = "variable",
    [7] = "macro"
}

-- Result metatable
local ResultMT = {
    __index = {
        -- Check if result has code snippet
        has_code = function(self)
            return self.code_snippet and #self.code_snippet > 0
        end,
        
        -- Check if result has context
        has_context = function(self)
            return self.context_json and #self.context_json > 0
        end,
        
        -- Pretty print
        __tostring = function(self)
            local s = string.format("%s (%s) at %s:%d", 
                self.name, 
                self.kind_name or KIND_NAMES[self.kind] or "unknown",
                self.file or "unknown",
                self.line)
            if self.signature and #self.signature > 0 then
                s = s .. "\n  Signature: " .. self.signature
            end
            return s
        end
    }
}

-- Index metatable
local IndexMT = {
    __index = {
        -- Find symbol by name
        find = function(self, name)
            local res = lib.symbol_index_find_py(self._handle, name)
            if res == nil then return nil end
            
            -- Extract fields
            local result = {
                name = ffi.string(lib.symbol_result_name(res)),
                signature = lib.symbol_result_signature(res),
                file = lib.symbol_result_file(res),
                line = lib.symbol_result_line(res),
                kind = lib.symbol_result_kind(res),
                code_snippet = lib.symbol_result_code_snippet(res),
                context_json = lib.symbol_result_context_json(res),
                _cres = res  -- Keep reference for cleanup
            }
            
            -- Convert nullable strings
            if result.signature ~= nil then
                result.signature = ffi.string(result.signature)
            else
                result.signature = ""
            end
            
            if result.file ~= nil then
                result.file = ffi.string(result.file)
            else
                result.file = ""
            end
            
            if result.code_snippet ~= nil then
                result.code_snippet = ffi.string(result.code_snippet)
            else
                result.code_snippet = ""
            end
            
            if result.context_json ~= nil then
                result.context_json = ffi.string(result.context_json)
            else
                result.context_json = ""
            end
            
            result.kind_name = KIND_NAMES[result.kind] or "unknown"
            
            setmetatable(result, ResultMT)
            return result
        end,
        
        -- Get symbol count
        count = function(self)
            return tonumber(lib.symbol_index_count(self._handle))
        end,
        
        -- Get file count
        file_count = function(self)
            return tonumber(lib.symbol_index_file_count(self._handle))
        end,
        
        -- Close index
        close = function(self)
            if self._handle then
                lib.symbol_index_close(self._handle)
                self._handle = nil
            end
        end,
        
        -- Check if valid
        valid = function(self)
            return self._handle ~= nil
        end,
        
        -- GC finalizer
        __gc = function(self)
            self:close()
        end
    }
}

-- Module table
local M = {}

-- Open index
function M.open(path)
    local handle = lib.symbol_index_open(path)
    if handle == nil then
        return nil, "Failed to open index: " .. tostring(path)
    end
    
    local idx = {_handle = handle}
    setmetatable(idx, IndexMT)
    return idx
end

-- Get kind name
function M.kind_name(kind)
    local name = lib.symbol_kind_name(kind)
    if name ~= nil then
        return ffi.string(name)
    end
    return "unknown"
end

-- Kind constants
M.KIND = {
    FUNCTION = 0,
    CLASS = 1,
    STRUCT = 2,
    ENUM = 3,
    NAMESPACE = 4,
    TYPEDEF = 5,
    VARIABLE = 6,
    MACRO = 7
}

-- Utility: load all symbols (for small indices)
function M.load_all(path)
    local idx, err = M.open(path)
    if not idx then return nil, err end
    
    -- Note: This would require iterating through all symbols
    -- Currently the C API doesn't expose iteration, so we'd need
    -- to add that functionality or use a different approach
    
    idx:close()
    return nil, "Iteration not yet implemented in C API"
end

-- Utility: search with multiple patterns
function M.search(path, patterns)
    local idx, err = M.open(path)
    if not idx then return nil, err end
    
    local results = {}
    
    for _, pattern in ipairs(patterns) do
        local result = idx:find(pattern)
        if result then
            table.insert(results, result)
        end
    end
    
    idx:close()
    return results
end

return M
