#include "utils/png_metadata.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <map>
#include <cstdint>
#include <zlib.h>

namespace myimg {

// PNG signature
static const uint8_t PNG_SIGNATURE[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};

// Read 4 bytes as big-endian uint32
static uint32_t read_u32_be(const uint8_t* data) {
    return (static_cast<uint32_t>(data[0]) << 24) |
           (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) |
           static_cast<uint32_t>(data[3]);
}

// Write uint32 as big-endian
static void write_u32_be(uint8_t* data, uint32_t value) {
    data[0] = static_cast<uint8_t>((value >> 24) & 0xFF);
    data[1] = static_cast<uint8_t>((value >> 16) & 0xFF);
    data[2] = static_cast<uint8_t>((value >> 8) & 0xFF);
    data[3] = static_cast<uint8_t>(value & 0xFF);
}

// PNG CRC32 table
static uint32_t crc32_table[256];
static bool crc32_table_initialized = false;

static void init_crc32_table() {
    if (crc32_table_initialized) return;
    for (int i = 0; i < 256; ++i) {
        uint32_t c = static_cast<uint32_t>(i);
        for (int j = 0; j < 8; ++j) {
            c = (c & 1) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1);
        }
        crc32_table[i] = c;
    }
    crc32_table_initialized = true;
}

static uint32_t png_crc32(const uint8_t* data, size_t len) {
    init_crc32_table();
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

// Build a tEXt chunk: key\0value
static std::vector<uint8_t> build_text_chunk(const std::string& key, const std::string& value) {
    std::vector<uint8_t> data;
    data.insert(data.end(), key.begin(), key.end());
    data.push_back(0); // null separator
    data.insert(data.end(), value.begin(), value.end());
    
    std::vector<uint8_t> chunk;
    chunk.resize(4 + 4 + data.size() + 4);
    
    write_u32_be(chunk.data(), static_cast<uint32_t>(data.size()));
    std::memcpy(chunk.data() + 4, "tEXt", 4);
    std::memcpy(chunk.data() + 8, data.data(), data.size());
    
    uint32_t crc = png_crc32(chunk.data() + 4, data.size() + 4);
    write_u32_be(chunk.data() + 8 + data.size(), crc);
    
    return chunk;
}

bool is_png_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    uint8_t sig[8];
    file.read(reinterpret_cast<char*>(sig), 8);
    return std::memcmp(sig, PNG_SIGNATURE, 8) == 0;
}

std::map<std::string, std::string> read_png_metadata(const std::string& path) {
    std::map<std::string, std::string> metadata;
    
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "[PNG Metadata] Failed to open file: " << path << std::endl;
        return metadata;
    }
    
    // Check PNG signature
    uint8_t sig[8];
    file.read(reinterpret_cast<char*>(sig), 8);
    if (std::memcmp(sig, PNG_SIGNATURE, 8) != 0) {
        std::cerr << "[PNG Metadata] Not a PNG file: " << path << std::endl;
        return metadata;
    }
    
    // Read chunks
    while (file) {
        uint8_t len_bytes[4];
        uint8_t type_bytes[4];
        
        file.read(reinterpret_cast<char*>(len_bytes), 4);
        file.read(reinterpret_cast<char*>(type_bytes), 4);
        
        if (!file) break;
        
        uint32_t length = read_u32_be(len_bytes);
        std::string type(reinterpret_cast<char*>(type_bytes), 4);
        
        // Read chunk data
        std::vector<uint8_t> chunk_data(length);
        if (length > 0) {
            file.read(reinterpret_cast<char*>(chunk_data.data()), length);
        }
        
        // Skip CRC
        uint8_t crc_bytes[4];
        file.read(reinterpret_cast<char*>(crc_bytes), 4);
        
        if (type == "tEXt") {
            // tEXt: key\0value
            size_t sep = 0;
            for (size_t i = 0; i < chunk_data.size(); ++i) {
                if (chunk_data[i] == 0) {
                    sep = i;
                    break;
                }
            }
            if (sep > 0 && sep < chunk_data.size() - 1) {
                std::string key(reinterpret_cast<char*>(chunk_data.data()), sep);
                std::string value(reinterpret_cast<char*>(chunk_data.data() + sep + 1), chunk_data.size() - sep - 1);
                metadata[key] = value;
            }
        }
        else if (type == "iTXt") {
            // iTXt: key\0compression\0lang\0transkey\0value
            size_t sep = 0;
            for (size_t i = 0; i < chunk_data.size(); ++i) {
                if (chunk_data[i] == 0) {
                    sep = i;
                    break;
                }
            }
            if (sep > 0 && sep < chunk_data.size() - 1) {
                std::string key(reinterpret_cast<char*>(chunk_data.data()), sep);
                // Skip compression flag, language, translated key
                size_t pos = sep + 1;
                if (pos < chunk_data.size()) pos++; // compression flag
                while (pos < chunk_data.size() && chunk_data[pos] != 0) pos++; // language
                pos++;
                while (pos < chunk_data.size() && chunk_data[pos] != 0) pos++; // translated key
                pos++;
                if (pos < chunk_data.size()) {
                    std::string value(reinterpret_cast<char*>(chunk_data.data() + pos), chunk_data.size() - pos);
                    metadata[key] = value;
                }
            }
        }
        else if (type == "zTXt") {
            // zTXt: key\0compression_method\0compressed_value
            size_t sep = 0;
            for (size_t i = 0; i < chunk_data.size(); ++i) {
                if (chunk_data[i] == 0) {
                    sep = i;
                    break;
                }
            }
            if (sep > 0 && sep < chunk_data.size() - 2) {
                std::string key(reinterpret_cast<char*>(chunk_data.data()), sep);
                // Skip compression method
                size_t pos = sep + 2;
                if (pos < chunk_data.size()) {
                    // Decompress
                    uLongf dest_len = chunk_data.size() * 10; // Estimate
                    std::vector<Bytef> dest(dest_len);
                    int ret = uncompress(dest.data(), &dest_len,
                                        chunk_data.data() + pos, chunk_data.size() - pos);
                    if (ret == Z_OK) {
                        std::string value(reinterpret_cast<char*>(dest.data()), dest_len);
                        metadata[key] = value;
                    }
                }
            }
        }
        else if (type == "IEND") {
            break;
        }
    }
    
    return metadata;
}

bool write_png_metadata(const std::string& path, const std::map<std::string, std::string>& metadata) {
    // Read entire file
    std::ifstream infile(path, std::ios::binary | std::ios::ate);
    if (!infile) {
        std::cerr << "[PNG Metadata] Failed to open file for writing: " << path << std::endl;
        return false;
    }
    
    std::streamsize size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::vector<uint8_t> file_data(size);
    if (!infile.read(reinterpret_cast<char*>(file_data.data()), size)) {
        std::cerr << "[PNG Metadata] Failed to read file: " << path << std::endl;
        return false;
    }
    infile.close();
    
    // Check PNG signature
    if (size < 8 || std::memcmp(file_data.data(), PNG_SIGNATURE, 8) != 0) {
        std::cerr << "[PNG Metadata] Not a PNG file: " << path << std::endl;
        return false;
    }
    
    // Find IEND chunk position
    size_t pos = 8; // Skip signature
    size_t iend_pos = 0;
    while (pos + 8 <= file_data.size()) {
        uint32_t length = read_u32_be(file_data.data() + pos);
        std::string type(reinterpret_cast<char*>(file_data.data() + pos + 4), 4);
        if (type == "IEND") {
            iend_pos = pos;
            break;
        }
        pos += 4 + 4 + length + 4; // length + type + data + crc
    }
    
    if (iend_pos == 0) {
        std::cerr << "[PNG Metadata] IEND chunk not found: " << path << std::endl;
        return false;
    }
    
    // Build tEXt chunks
    std::vector<uint8_t> text_chunks;
    for (const auto& [key, value] : metadata) {
        if (key.empty() || key.size() > 79) continue; // PNG tEXt key limit
        auto chunk = build_text_chunk(key, value);
        text_chunks.insert(text_chunks.end(), chunk.begin(), chunk.end());
    }
    
    if (text_chunks.empty()) {
        return true; // Nothing to embed
    }
    
    // Insert text chunks before IEND
    std::vector<uint8_t> new_file;
    new_file.reserve(file_data.size() + text_chunks.size());
    new_file.insert(new_file.end(), file_data.begin(), file_data.begin() + iend_pos);
    new_file.insert(new_file.end(), text_chunks.begin(), text_chunks.end());
    new_file.insert(new_file.end(), file_data.begin() + iend_pos, file_data.end());
    
    // Write back
    std::ofstream outfile(path, std::ios::binary | std::ios::trunc);
    if (!outfile) {
        std::cerr << "[PNG Metadata] Failed to open file for writing: " << path << std::endl;
        return false;
    }
    outfile.write(reinterpret_cast<const char*>(new_file.data()), new_file.size());
    return outfile.good();
}

} // namespace myimg
