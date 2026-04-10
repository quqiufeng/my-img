#!/usr/bin/env luajit
-- Test program for Symbol Index V3 LuaJIT FFI bindings

local idxlib = require("symbol_index_v3_ffi")

if #arg < 2 then
    print("Usage: " .. arg[0] .. " <index.bin> <symbol_name>")
    os.exit(1)
end

-- Open index
local idx, err = idxlib.open(arg[1])
if not idx then
    print("Error: " .. tostring(err))
    os.exit(1)
end

print("Index loaded successfully!")
print("Symbols: " .. idx:count())
print("Files: " .. idx:file_count())
print("")

-- Find symbol
local result = idx:find(arg[2])

if result then
    print("Found: " .. result.name)
    print("Kind: " .. result.kind_name .. " (" .. result.kind .. ")")
    print("File: " .. result.file .. ":" .. result.line)
    
    if result.signature and #result.signature > 0 then
        print("Signature: " .. result.signature)
    end
    
    if result:has_code() then
        print("\nCode snippet (" .. #result.code_snippet .. " bytes):")
        local snippet = result.code_snippet
        if #snippet > 500 then
            snippet = snippet:sub(1, 500) .. "..."
        end
        print(snippet)
    end
    
    if result:has_context() then
        print("\nContext JSON (" .. #result.context_json .. " bytes):")
        local ctx = result.context_json
        if #ctx > 500 then
            ctx = ctx:sub(1, 500) .. "..."
        end
        print(ctx)
    end
else
    print("Symbol not found: " .. arg[2])
end

-- Close index
idx:close()
print("\nTest completed successfully!")
