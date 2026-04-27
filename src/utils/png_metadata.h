#pragma once

#include <string>
#include <map>
#include <string>
#include <vector>

namespace myimg {

// Read PNG text chunks (tEXt, iTXt, zTXt)
// Returns map of key->value pairs
std::map<std::string, std::string> read_png_metadata(const std::string& path);

// Check if file is a PNG
bool is_png_file(const std::string& path);

} // namespace myimg
