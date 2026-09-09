#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "khoj/flat_index.hpp"
#include "khoj/hnsw_index.hpp"

#include "vecs_io.hpp"

namespace {

using khoj::bench::FloatVectors;
using khoj::bench::IntVectors;

struct Options {
    std::string base_path;
    std::string query_path;
    std::string groundtruth_path;
    std::string index_kind = "hnsw";
    std::string dataset = "sift1m";
    std::string metric = "l2";
    std::string csv_path;
    std::size_t k = 10;
    std::size_t threads = 1;
    std::size_t runs = 5;
    std::size_t base_limit = 0;
    std::size_t query_limit = 0;
    std::size_t max_neighbors = 16;
    std::size_t max_neighbors_layer0 = 32;
    std::size_t ef_construction = 200;
    std::vector<std::size_t> ef_search{10, 20, 40, 80, 160, 320};
};

struct Row {
    std::string engine = "khoj";
    std::string index_kind;
    std::string dataset;
    std::string metric;
    std::string groundtruth_source;
    std::size_t num_base = 0;
    std::size_t num_queries = 0;
    std::size_t dimension = 0;
    std::size_t k = 0;
    std::string max_neighbors;
    std::string max_neighbors_layer0;
    std::string ef_construction;
    std::string ef_search;
    std::size_t threads = 0;
    std::size_t runs = 0;
    double recall = 0.0;
    double qps_median = 0.0;
    double qps_min = 0.0;
    double qps_max = 0.0;
    double build_seconds = 0.0;
    std::size_t index_rss_bytes = 0;
    std::size_t peak_rss_bytes = 0;
};

const char* csv_header() {
    return "engine,index,dataset,metric,groundtruth_source,num_base,num_queries,dimension,k,"
           "max_neighbors,max_neighbors_layer0,ef_construction,ef_search,threads,runs,"
           "recall_at_k,qps_median,qps_min,qps_max,build_seconds,index_rss_bytes,peak_rss_bytes";
}

void write_row(std::ostream& out, const Row& row) {
    std::ostringstream line;
    line << row.engine << ',' << row.index_kind << ',' << row.dataset << ',' << row.metric << ','
         << row.groundtruth_source << ',' << row.num_base << ',' << row.num_queries << ','
         << row.dimension << ',' << row.k << ',' << row.max_neighbors << ','
         << row.max_neighbors_layer0 << ',' << row.ef_construction << ',' << row.ef_search << ','
         << row.threads << ',' << row.runs << ',';
    line << std::fixed << std::setprecision(6) << row.recall << ',';
    line << std::setprecision(3) << row.qps_median << ',' << row.qps_min << ',' << row.qps_max
         << ',';
    line << std::setprecision(6) << row.build_seconds << ',';
    line << row.index_rss_bytes << ',' << row.peak_rss_bytes;
    out << line.str() << '\n';
}

std::size_t read_status_bytes(const char* key) {
    std::ifstream status("/proc/self/status");
    if (!status) {
        return 0;
    }
    std::string line;
    const std::size_t key_length = std::strlen(key);
    while (std::getline(status, line)) {
        if (line.size() >= key_length && line.compare(0, key_length, key) == 0) {
            std::istringstream parts(line.substr(key_length));
            std::size_t kilobytes = 0;
            parts >> kilobytes;
            return kilobytes * 1024;
        }
    }
    return 0;
}

std::size_t current_rss() {
    return read_status_bytes("VmRSS:");
}

std::size_t peak_rss() {
    return read_status_bytes("VmHWM:");
}

std::vector<std::size_t> parse_size_list(const std::string& text) {
    std::vector<std::size_t> values;
    std::istringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            values.push_back(static_cast<std::size_t>(std::stoull(item)));
        }
    }
    return values;
}

double median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[middle];
    }
    return 0.5 * (values[middle - 1] + values[middle]);
}

std::vector<std::vector<float>> split_queries(const FloatVectors& queries) {
    std::vector<std::vector<float>> split;
    split.reserve(queries.count);
    for (std::size_t index = 0; index < queries.count; ++index) {
        const float* start = queries.at(index);
        split.emplace_back(start, start + queries.dimension);
    }
    return split;
}

double recall_against(const std::vector<std::vector<std::uint64_t>>& retrieved,
                      const IntVectors& truth,
                      std::size_t k) {
    if (truth.dimension < k) {
        throw std::runtime_error("groundtruth has fewer neighbours per query than k");
    }
    if (truth.count < retrieved.size()) {
        throw std::runtime_error("groundtruth has fewer rows than the query set");
    }

    double total = 0.0;
    std::unordered_set<std::uint64_t> expected;
    expected.reserve(k * 2);

    for (std::size_t query = 0; query < retrieved.size(); ++query) {
        expected.clear();
        const std::int32_t* row = truth.at(query);
        for (std::size_t rank = 0; rank < k; ++rank) {
            expected.insert(static_cast<std::uint64_t>(row[rank]));
        }

        std::size_t hits = 0;
        for (const std::uint64_t label : retrieved[query]) {
            if (expected.count(label) != 0) {
                ++hits;
            }
        }
        total += static_cast<double>(hits) / static_cast<double>(k);
    }

    return retrieved.empty() ? 0.0 : total / static_cast<double>(retrieved.size());
}

template <typename SearchFn>
double timed_pass(const SearchFn& search,
                  std::size_t num_queries,
                  std::size_t threads,
                  std::vector<std::vector<std::uint64_t>>& retrieved) {
    const auto start = std::chrono::steady_clock::now();

    if (threads <= 1) {
        for (std::size_t query = 0; query < num_queries; ++query) {
            search(query, retrieved[query]);
        }
    } else {
        std::vector<std::thread> workers;
        workers.reserve(threads);
        for (std::size_t worker = 0; worker < threads; ++worker) {
            workers.emplace_back([&, worker]() {
                for (std::size_t query = worker; query < num_queries; query += threads) {
                    search(query, retrieved[query]);
                }
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

    const auto finish = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(finish - start).count();
}

template <typename SearchFn>
std::vector<std::vector<std::uint64_t>> measure(const SearchFn& search,
                                                std::size_t num_queries,
                                                std::size_t threads,
                                                std::size_t runs,
                                                Row& row) {
    std::vector<std::vector<std::uint64_t>> retrieved(num_queries);

    std::vector<double> rates;
    rates.reserve(runs);
    for (std::size_t run = 0; run < runs; ++run) {
        const double elapsed = timed_pass(search, num_queries, threads, retrieved);
        rates.push_back(elapsed > 0.0 ? static_cast<double>(num_queries) / elapsed : 0.0);
    }

    row.qps_median = median_of(rates);
    row.qps_min = *std::min_element(rates.begin(), rates.end());
    row.qps_max = *std::max_element(rates.begin(), rates.end());
    return retrieved;
}

IntVectors groundtruth_from_flat(const FloatVectors& base,
                                 const std::vector<std::vector<float>>& queries,
                                 khoj::Metric metric,
                                 std::size_t k,
                                 std::size_t threads) {
    khoj::FlatIndex index(base.dimension, metric);
    index.reserve(base.count);
    index.add_batch(base.data);

    IntVectors truth;
    truth.count = queries.size();
    truth.dimension = k;
    truth.data.assign(queries.size() * k, 0);

    const auto fill = [&](std::size_t query) {
        const std::vector<khoj::SearchResult> found = index.search(queries[query], k);
        for (std::size_t rank = 0; rank < k; ++rank) {
            truth.data[query * k + rank] =
                rank < found.size() ? static_cast<std::int32_t>(found[rank].label) : -1;
        }
    };

    if (threads <= 1) {
        for (std::size_t query = 0; query < queries.size(); ++query) {
            fill(query);
        }
    } else {
        std::vector<std::thread> workers;
        workers.reserve(threads);
        for (std::size_t worker = 0; worker < threads; ++worker) {
            workers.emplace_back([&, worker]() {
                for (std::size_t query = worker; query < queries.size(); query += threads) {
                    fill(query);
                }
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

    return truth;
}

khoj::Metric parse_metric(const std::string& text) {
    if (text == "l2") {
        return khoj::Metric::L2;
    }
    if (text == "ip") {
        return khoj::Metric::InnerProduct;
    }
    throw std::runtime_error("unknown metric " + text);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        const auto next = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error("missing value for " + flag);
            }
            return argv[++index];
        };

        if (flag == "--base") {
            options.base_path = next();
        } else if (flag == "--query") {
            options.query_path = next();
        } else if (flag == "--groundtruth") {
            options.groundtruth_path = next();
        } else if (flag == "--index") {
            options.index_kind = next();
        } else if (flag == "--dataset") {
            options.dataset = next();
        } else if (flag == "--metric") {
            options.metric = next();
        } else if (flag == "--csv") {
            options.csv_path = next();
        } else if (flag == "--k") {
            options.k = static_cast<std::size_t>(std::stoull(next()));
        } else if (flag == "--threads") {
            options.threads = static_cast<std::size_t>(std::stoull(next()));
        } else if (flag == "--runs") {
            options.runs = static_cast<std::size_t>(std::stoull(next()));
        } else if (flag == "--base-limit") {
            options.base_limit = static_cast<std::size_t>(std::stoull(next()));
        } else if (flag == "--query-limit") {
            options.query_limit = static_cast<std::size_t>(std::stoull(next()));
        } else if (flag == "--max-neighbors") {
            options.max_neighbors = static_cast<std::size_t>(std::stoull(next()));
        } else if (flag == "--max-neighbors-layer0") {
            options.max_neighbors_layer0 = static_cast<std::size_t>(std::stoull(next()));
        } else if (flag == "--ef-construction") {
            options.ef_construction = static_cast<std::size_t>(std::stoull(next()));
        } else if (flag == "--ef-search") {
            options.ef_search = parse_size_list(next());
        } else {
            throw std::runtime_error("unknown flag " + flag);
        }
    }

    if (options.base_path.empty() || options.query_path.empty()) {
        throw std::runtime_error("--base and --query are required");
    }
    if (options.runs < 5) {
        throw std::runtime_error("benchmark discipline requires at least 5 runs per configuration");
    }
    if (options.threads == 0) {
        throw std::runtime_error("--threads must be positive");
    }
    if (options.k == 0) {
        throw std::runtime_error("--k must be positive");
    }
    if (options.index_kind != "flat" && options.index_kind != "hnsw") {
        throw std::runtime_error("--index must be flat or hnsw");
    }
    if (options.index_kind == "hnsw" && options.ef_search.empty()) {
        throw std::runtime_error("--ef-search must list at least one value");
    }
    return options;
}

Row make_row(const Options& options, const FloatVectors& base, const FloatVectors& queries) {
    Row row;
    row.index_kind = options.index_kind;
    row.dataset = options.dataset;
    row.metric = options.metric;
    row.num_base = base.count;
    row.num_queries = queries.count;
    row.dimension = base.dimension;
    row.k = options.k;
    row.threads = options.threads;
    row.runs = options.runs;
    return row;
}

void report(std::ostream& out, const Row& row) {
    out << "  " << row.index_kind
        << (row.ef_search.empty() ? std::string() : " ef_search=" + row.ef_search)
        << " threads=" << row.threads << " runs=" << row.runs << std::fixed
        << std::setprecision(4) << " recall@" << row.k << "=" << row.recall
        << std::setprecision(1) << " qps_median=" << row.qps_median << " peak_rss_mib="
        << static_cast<double>(row.peak_rss_bytes) / (1024.0 * 1024.0) << '\n';
}

}

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const khoj::Metric metric = parse_metric(options.metric);

        const FloatVectors base = khoj::bench::read_fvecs(options.base_path, options.base_limit);
        const FloatVectors queries =
            khoj::bench::read_fvecs(options.query_path, options.query_limit);

        if (base.dimension != queries.dimension) {
            throw std::runtime_error("base and query vectors have different dimensions");
        }
        if (base.count == 0 || queries.count == 0) {
            throw std::runtime_error("base and query sets must be non-empty");
        }

        std::cerr << "loaded " << base.count << " base and " << queries.count
                  << " query vectors of dimension " << base.dimension << '\n';

        const std::vector<std::vector<float>> query_vectors = split_queries(queries);

        IntVectors truth;
        std::string truth_source;
        if (!options.groundtruth_path.empty()) {
            truth = khoj::bench::read_ivecs(options.groundtruth_path, options.query_limit);
            truth_source = "file";
        } else if (options.index_kind == "hnsw") {
            std::cerr << "computing groundtruth with the flat index\n";
            truth = groundtruth_from_flat(base, query_vectors, metric, options.k, options.threads);
            truth_source = "flat";
        } else {
            truth_source = "self";
        }

        std::vector<Row> rows;

        const std::size_t rss_before = current_rss();
        const auto build_start = std::chrono::steady_clock::now();

        if (options.index_kind == "flat") {
            khoj::FlatIndex index(base.dimension, metric);
            index.reserve(base.count);
            index.add_batch(base.data);
            const double build_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start)
                    .count();
            const std::size_t index_rss = current_rss();

            Row row = make_row(options, base, queries);
            row.groundtruth_source = truth_source;
            row.build_seconds = build_seconds;
            row.index_rss_bytes = index_rss > rss_before ? index_rss - rss_before : 0;

            const auto search = [&](std::size_t query, std::vector<std::uint64_t>& out) {
                const std::vector<khoj::SearchResult> found =
                    index.search(query_vectors[query], options.k);
                out.resize(found.size());
                for (std::size_t rank = 0; rank < found.size(); ++rank) {
                    out[rank] = found[rank].label;
                }
            };

            const std::vector<std::vector<std::uint64_t>> retrieved =
                measure(search, queries.count, options.threads, options.runs, row);

            row.recall = truth_source == "self" ? 1.0 : recall_against(retrieved, truth, options.k);
            row.peak_rss_bytes = peak_rss();
            rows.push_back(row);
        } else {
            khoj::HnswParams params;
            params.dimension = base.dimension;
            params.max_neighbors = options.max_neighbors;
            params.max_neighbors_layer0 = options.max_neighbors_layer0;
            params.ef_construction = options.ef_construction;
            params.metric = metric;

            khoj::HnswIndex index(params);
            index.reserve(base.count);
            for (std::size_t position = 0; position < base.count; ++position) {
                index.add(static_cast<std::uint64_t>(position), base.at(position));
            }
            const double build_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start)
                    .count();
            const std::size_t index_rss = current_rss();

            std::cerr << "built hnsw over " << index.size() << " vectors in " << std::fixed
                      << std::setprecision(2) << build_seconds << "s\n";

            for (const std::size_t ef : options.ef_search) {
                if (ef < options.k) {
                    throw std::runtime_error("ef_search must be at least k");
                }

                Row row = make_row(options, base, queries);
                row.groundtruth_source = truth_source;
                row.build_seconds = build_seconds;
                row.index_rss_bytes = index_rss > rss_before ? index_rss - rss_before : 0;
                row.max_neighbors = std::to_string(options.max_neighbors);
                row.max_neighbors_layer0 = std::to_string(options.max_neighbors_layer0);
                row.ef_construction = std::to_string(options.ef_construction);
                row.ef_search = std::to_string(ef);

                const auto search = [&](std::size_t query, std::vector<std::uint64_t>& out) {
                    const std::vector<khoj::SearchResult> found =
                        index.search(queries.at(query), options.k, ef);
                    out.resize(found.size());
                    for (std::size_t rank = 0; rank < found.size(); ++rank) {
                        out[rank] = found[rank].label;
                    }
                };

                const std::vector<std::vector<std::uint64_t>> retrieved =
                    measure(search, queries.count, options.threads, options.runs, row);

                row.recall = recall_against(retrieved, truth, options.k);
                row.peak_rss_bytes = peak_rss();
                rows.push_back(row);
            }
        }

        std::ofstream file;
        if (!options.csv_path.empty()) {
            file.open(options.csv_path);
            if (!file) {
                throw std::runtime_error("cannot write " + options.csv_path);
            }
        }
        std::ostream& out = options.csv_path.empty() ? std::cout : file;

        out << csv_header() << '\n';
        for (const Row& row : rows) {
            write_row(out, row);
        }

        std::cerr << "results:\n";
        for (const Row& row : rows) {
            report(std::cerr, row);
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bench_khoj: " << error.what() << '\n';
        return 1;
    }
}
