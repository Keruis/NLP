#ifndef READIMAGE_HPP
#define READIMAGE_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <zlib.h>
#include <vector>
#include <fstream>
#include <stdexcept>

namespace ReadPng {
    using PixelMatrix = std::vector<std::vector<std::vector<unsigned char>>>; // 对于 RGB 图像

    struct PNGHeader {
        uint32_t width;
        uint32_t height;
        uint8_t bit_depth;
        uint8_t color_type;
        uint8_t compression_method;
        uint8_t filter_method;
        uint8_t interlace_method;
    };

    struct PNGChunk {
        uint32_t length;
        std::string type;
        std::vector<uint8_t> data;
        uint32_t crc;
    };

    struct PNGImage {
        PNGHeader header;
        PixelMatrix pixels;
    };

    bool VerifyPNGSignature(const std::vector<uint8_t>& signature) {
        return signature[0] == 0x89 && signature[1] == 0x50 && signature[2] == 0x4E && signature[3] == 0x47 &&
               signature[4] == 0x0D && signature[5] == 0x0A && signature[6] == 0x1A && signature[7] == 0x0A;
    }

    PNGHeader ParseIHDR(const PNGChunk& ihdr_chunk) {
        PNGHeader header;
        const uint8_t* data = ihdr_chunk.data.data();

        header.width = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
        header.height = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7];
        header.bit_depth = data[8];
        header.color_type = data[9];
        header.compression_method = data[10];
        header.filter_method = data[11];
        header.interlace_method = data[12];

        return header;
    }

    PNGChunk ReadChunk(std::ifstream& file) {
        PNGChunk chunk;

        uint8_t length_bytes[4];
        file.read(reinterpret_cast<char*>(length_bytes), 4);
        chunk.length = (length_bytes[0] << 24) | (length_bytes[1] << 16) | (length_bytes[2] << 8) | length_bytes[3];

        char type_bytes[5] = {0};
        file.read(type_bytes, 4);
        chunk.type = std::string(type_bytes);

        chunk.data.resize(chunk.length);
        file.read(reinterpret_cast<char*>(chunk.data.data()), chunk.length);

        uint8_t crc_bytes[4];
        file.read(reinterpret_cast<char*>(crc_bytes), 4);
        chunk.crc = (crc_bytes[0] << 24) | (crc_bytes[1] << 16) | (crc_bytes[2] << 8) | crc_bytes[3];

        return chunk;
    }

    std::vector<uint8_t> DecompressIDAT(const std::vector<uint8_t>& compressed_data) {
        z_stream strm = {};
        inflateInit(&strm);

        std::vector<uint8_t> decompressed_data;
        decompressed_data.resize(1024 * 1024); 

        strm.avail_in = compressed_data.size();
        strm.next_in = const_cast<uint8_t*>(compressed_data.data());
        strm.avail_out = decompressed_data.size();
        strm.next_out = decompressed_data.data();

        while (true) {
            int ret = inflate(&strm, Z_NO_FLUSH);

            if (ret == Z_STREAM_END) break;
            if (ret != Z_OK) {
                std::cerr << "Zlib 解压失败！" << std::endl;
                break;
            }

            // 如果空间不足，扩展缓冲区
            if (strm.avail_out == 0) {
                size_t old_size = decompressed_data.size();
                decompressed_data.resize(old_size * 2);
                strm.avail_out = old_size;
                strm.next_out = decompressed_data.data() + old_size;
            }
        }

        decompressed_data.resize(decompressed_data.size() - strm.avail_out);

        inflateEnd(&strm);

        return decompressed_data;
    }

    void RemoveFilter(const PNGHeader& header, const std::vector<uint8_t>& raw_data, PixelMatrix& output) {
        size_t bytes_per_pixel = (header.bit_depth / 8) * 3; 
        size_t stride = header.width * bytes_per_pixel;
        size_t offset = 0;

        for (uint32_t y = 0; y < header.height; ++y) {
            uint8_t filter_type = raw_data[offset];
            ++offset;

            for (uint32_t x = 0; x < header.width; ++x) {
                for (uint32_t c = 0; c < 3; ++c) {
                    uint8_t value = raw_data[offset];
                    output[y][x][c] = value; 
                    ++offset;
                }
            }
        }
    }

    PNGImage ParsePNG(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开文件！");
        }

        uint8_t signature[8];
        file.read(reinterpret_cast<char*>(signature), 8);

        if (!VerifyPNGSignature(std::vector<uint8_t>(signature, signature + 8))) {
            throw std::runtime_error("文件不是合法的 PNG！");
        }

        PNGHeader header;
        std::vector<uint8_t> compressed_data;
        while (!file.eof()) {
            PNGChunk chunk = ReadChunk(file);
            if (chunk.type == "IHDR") {
                header = ParseIHDR(chunk);
            } else if (chunk.type == "IDAT") {
                compressed_data.insert(compressed_data.end(), chunk.data.begin(), chunk.data.end());
            } else if (chunk.type == "IEND") {
                break;
            }
        }

        std::vector<uint8_t> decompressed_data = DecompressIDAT(compressed_data);

        PixelMatrix output(header.height, std::vector<std::vector<uint8_t>>(header.width, std::vector<uint8_t>(3)));

        RemoveFilter(header, decompressed_data, output);

        PNGImage image = { header, output };
        return image;
    }

        void WriteChunk(std::ofstream& file, const std::string& type, const std::vector<uint8_t>& data) {
        uint32_t length = static_cast<uint32_t>(data.size());
        uint32_t crc;

        uint8_t length_bytes[4] = { 
            static_cast<uint8_t>((length >> 24) & 0xFF),
            static_cast<uint8_t>((length >> 16) & 0xFF),
            static_cast<uint8_t>((length >> 8) & 0xFF),
            static_cast<uint8_t>(length & 0xFF)
        };
        file.write(reinterpret_cast<const char*>(length_bytes), 4);

        file.write(type.c_str(), 4);

        file.write(reinterpret_cast<const char*>(data.data()), data.size());

        crc = crc32(0, Z_NULL, 0); 
        crc = crc32(crc, reinterpret_cast<const unsigned char*>(type.c_str()), 4);
        crc = crc32(crc, data.data(), data.size());

        uint8_t crc_bytes[4] = { 
            static_cast<uint8_t>((crc >> 24) & 0xFF),
            static_cast<uint8_t>((crc >> 16) & 0xFF),
            static_cast<uint8_t>((crc >> 8) & 0xFF),
            static_cast<uint8_t>(crc & 0xFF)
        };
        file.write(reinterpret_cast<const char*>(crc_bytes), 4);
    }

    std::vector<uint8_t> CompressIDAT(const std::vector<uint8_t>& raw_data) {
        z_stream strm = {};
        deflateInit(&strm, Z_BEST_COMPRESSION);

        std::vector<uint8_t> compressed_data;
        compressed_data.resize(raw_data.size());

        strm.avail_in = raw_data.size();
        strm.next_in = const_cast<uint8_t*>(raw_data.data());
        strm.avail_out = compressed_data.size();
        strm.next_out = compressed_data.data();

        while (true) {
            int ret = deflate(&strm, Z_FINISH);

            if (ret == Z_STREAM_END) break;
            if (ret != Z_OK) {
                std::cerr << "Zlib 压缩失败！" << std::endl;
                break;
            }

            if (strm.avail_out == 0) {
                size_t old_size = compressed_data.size();
                compressed_data.resize(old_size * 2);
                strm.avail_out = compressed_data.size() - old_size;
                strm.next_out = compressed_data.data() + old_size;
            }
        }

        compressed_data.resize(compressed_data.size() - strm.avail_out);

        deflateEnd(&strm);

        return compressed_data;
    }

    void WritePNG(const std::string& filename, const PNGHeader& header, const std::vector<std::vector<float>>& pixels) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("无法创建文件！");
        }

        const uint8_t png_signature[8] = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
        file.write(reinterpret_cast<const char*>(png_signature), 8);

        std::vector<uint8_t> ihdr_data(13);
        ihdr_data[0] = (header.width >> 24) & 0xFF;
        ihdr_data[1] = (header.width >> 16) & 0xFF;
        ihdr_data[2] = (header.width >> 8) & 0xFF;
        ihdr_data[3] = header.width & 0xFF;

        ihdr_data[4] = (header.height >> 24) & 0xFF;
        ihdr_data[5] = (header.height >> 16) & 0xFF;
        ihdr_data[6] = (header.height >> 8) & 0xFF;
        ihdr_data[7] = header.height & 0xFF;

        ihdr_data[8] = header.bit_depth;
        ihdr_data[9] = header.color_type;
        ihdr_data[10] = header.compression_method;
        ihdr_data[11] = header.filter_method;
        ihdr_data[12] = header.interlace_method;

        WriteChunk(file, "IHDR", ihdr_data);

        std::vector<uint8_t> raw_data;
        size_t bytes_per_pixel = 1;
        for (size_t y = 0; y < header.height; ++y) {
            raw_data.push_back(0); 
            for (size_t x = 0; x < header.width; ++x) {
                for (size_t c = 0; c < bytes_per_pixel; ++c) {
                    raw_data.push_back(static_cast<uint8_t>(pixels[y][x]));
                }
            }
        }

        std::vector<uint8_t> compressed_data = CompressIDAT(raw_data);
        WriteChunk(file, "IDAT", compressed_data);

        WriteChunk(file, "IEND", {});

        file.close();
    }

    template<typename T>
    void WritePNG(const std::string& filename, const PNGHeader& header, const std::vector<std::vector<std::vector<T>>>& pixels) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("无法创建文件！");
        }

        const uint8_t png_signature[8] = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
        file.write(reinterpret_cast<const char*>(png_signature), 8);

        std::vector<uint8_t> ihdr_data(13);
        ihdr_data[0] = (header.width >> 24) & 0xFF;
        ihdr_data[1] = (header.width >> 16) & 0xFF;
        ihdr_data[2] = (header.width >> 8) & 0xFF;
        ihdr_data[3] = header.width & 0xFF;

        ihdr_data[4] = (header.height >> 24) & 0xFF;
        ihdr_data[5] = (header.height >> 16) & 0xFF;
        ihdr_data[6] = (header.height >> 8) & 0xFF;
        ihdr_data[7] = header.height & 0xFF;

        ihdr_data[8] = header.bit_depth;
        ihdr_data[9] = header.color_type;
        ihdr_data[10] = header.compression_method;
        ihdr_data[11] = header.filter_method;
        ihdr_data[12] = header.interlace_method;

        WriteChunk(file, "IHDR", ihdr_data);

        std::vector<uint8_t> raw_data;
        size_t bytes_per_pixel = 3; 
        for (size_t y = 0; y < header.height; ++y) {
            raw_data.push_back(0);
            for (size_t x = 0; x < header.width; ++x) {
                for (size_t c = 0; c < bytes_per_pixel; ++c) {
                    raw_data.push_back(pixels[y][x][c]);
                }
            }
        }

        std::vector<uint8_t> compressed_data = CompressIDAT(raw_data);
        WriteChunk(file, "IDAT", compressed_data);

        WriteChunk(file, "IEND", {});

        file.close();
    }
}

#endif // READIMAGE_HPP