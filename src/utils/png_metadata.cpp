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

} // namespace myimg
