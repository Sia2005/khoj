#include "vecs_io.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>

namespace khoj::bench {
namespace {

template <typename T>
void read_records(const std::string& path,
                  std::size_t max_count,
                  std::size_t& count,
                  std::size_t& dimension,
                  std::vector<T>& data) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path);
    }

    stream.seekg(0, std::ios::end);
    const std::streamoff file_size = stream.tellg();
    stream.seekg(0, std::ios::beg);

    if (file_size < static_cast<std::streamoff>(sizeof(std::int32_t))) {
        throw std::runtime_error(path + " is too small to hold a record header");
    }

    std::int32_t header = 0;
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!stream || header <= 0) {
        throw std::runtime_error(path + " declares a non-positive dimension");
    }

    dimension = static_cast<std::size_t>(header);
    const std::size_t record_bytes = sizeof(std::int32_t) + dimension * sizeof(T);
    if (static_cast<std::size_t>(file_size) % record_bytes != 0) {
        throw std::runtime_error(path + " size is not a multiple of its record length");
    }

    const std::size_t available = static_cast<std::size_t>(file_size) / record_bytes;
    count = max_count == 0 ? available : std::min(available, max_count);
    data.resize(count * dimension);

    stream.seekg(0, std::ios::beg);
    for (std::size_t index = 0; index < count; ++index) {
        std::int32_t record_dimension = 0;
        stream.read(reinterpret_cast<char*>(&record_dimension), sizeof(record_dimension));
        if (!stream || static_cast<std::size_t>(record_dimension) != dimension) {
            throw std::runtime_error(path + " contains records of differing dimension");
        }
        stream.read(reinterpret_cast<char*>(data.data() + index * dimension),
                    static_cast<std::streamsize>(dimension * sizeof(T)));
        if (!stream) {
            throw std::runtime_error(path + " ended before record " + std::to_string(index));
        }
    }
}

}

FloatVectors read_fvecs(const std::string& path, std::size_t max_count) {
    FloatVectors vectors;
    read_records(path, max_count, vectors.count, vectors.dimension, vectors.data);
    return vectors;
}

IntVectors read_ivecs(const std::string& path, std::size_t max_count) {
    IntVectors vectors;
    read_records(path, max_count, vectors.count, vectors.dimension, vectors.data);
    return vectors;
}

}
