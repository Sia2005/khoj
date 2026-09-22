#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "khoj/flat_index.hpp"
#include "khoj/hnsw_index.hpp"
#include "khoj/types.hpp"

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using LabelArray = py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>;

constexpr std::uint64_t kMissingLabel = std::numeric_limits<std::uint64_t>::max();

float missing_distance(khoj::Metric metric) {
    return metric == khoj::Metric::L2 ? std::numeric_limits<float>::infinity()
                                      : -std::numeric_limits<float>::infinity();
}

std::size_t row_count(const FloatArray& vectors, std::size_t dimension, const std::string& context) {
    if (vectors.ndim() != 2) {
        throw std::invalid_argument(context + " expects a two-dimensional array, got " +
                                    std::to_string(vectors.ndim()) + " dimensions");
    }
    if (static_cast<std::size_t>(vectors.shape(1)) != dimension) {
        throw std::invalid_argument(context + " expects rows of width " + std::to_string(dimension) +
                                    ", got " + std::to_string(vectors.shape(1)));
    }
    return static_cast<std::size_t>(vectors.shape(0));
}

const float* single_vector(const FloatArray& vector, std::size_t dimension, const std::string& context) {
    if (vector.ndim() != 1) {
        throw std::invalid_argument(context + " expects a one-dimensional array, got " +
                                    std::to_string(vector.ndim()) + " dimensions");
    }
    if (static_cast<std::size_t>(vector.shape(0)) != dimension) {
        throw std::invalid_argument(context + " expects a vector of length " +
                                    std::to_string(dimension) + ", got " +
                                    std::to_string(vector.shape(0)));
    }
    return vector.data();
}

template <typename T>
py::array_t<T> make_matrix(std::size_t rows, std::size_t columns) {
    return py::array_t<T>({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(columns)});
}

void write_row(const std::vector<khoj::SearchResult>& results,
               std::size_t k,
               float missing,
               std::uint64_t* labels,
               float* distances) {
    const std::size_t filled = std::min(k, results.size());

    for (std::size_t rank = 0; rank < filled; ++rank) {
        labels[rank] = results[rank].label;
        distances[rank] = results[rank].distance;
    }
    for (std::size_t rank = filled; rank < k; ++rank) {
        labels[rank] = kMissingLabel;
        distances[rank] = missing;
    }
}

std::uint64_t flat_add(khoj::FlatIndex& index, const FloatArray& vector) {
    const std::size_t dimension = index.dimension();
    const float* const values = single_vector(vector, dimension, "add");
    return index.add(std::vector<float>(values, values + dimension));
}

void flat_add_batch(khoj::FlatIndex& index, const FloatArray& vectors) {
    const std::size_t dimension = index.dimension();
    const std::size_t rows = row_count(vectors, dimension, "add_batch");
    if (rows == 0) {
        return;
    }

    const float* const values = vectors.data();
    index.reserve(index.size() + rows);
    index.add_batch(std::vector<float>(values, values + rows * dimension));
}

std::vector<khoj::SearchResult> flat_search(const khoj::FlatIndex& index,
                                            const FloatArray& query,
                                            std::size_t k) {
    const std::size_t dimension = index.dimension();
    const float* const values = single_vector(query, dimension, "search");
    return index.search(std::vector<float>(values, values + dimension), k);
}

py::tuple flat_search_batch(const khoj::FlatIndex& index, const FloatArray& queries, std::size_t k) {
    const std::size_t dimension = index.dimension();
    const std::size_t rows = row_count(queries, dimension, "search_batch");

    py::array_t<std::uint64_t> labels = make_matrix<std::uint64_t>(rows, k);
    py::array_t<float> distances = make_matrix<float>(rows, k);
    if (rows == 0 || k == 0) {
        return py::make_tuple(labels, distances);
    }

    const float* const query_values = queries.data();
    const std::vector<float> query_block(query_values, query_values + rows * dimension);
    std::uint64_t* const label_data = labels.mutable_data();
    float* const distance_data = distances.mutable_data();
    const float missing = missing_distance(index.metric());

    {
        const py::gil_scoped_release release;
        const std::vector<std::vector<khoj::SearchResult>> batch = index.search_batch(query_block, k);
        for (std::size_t row = 0; row < rows; ++row) {
            write_row(batch[row], k, missing, label_data + row * k, distance_data + row * k);
        }
    }

    return py::make_tuple(labels, distances);
}

py::array_t<float> flat_to_numpy(const khoj::FlatIndex& index) {
    py::array_t<float> values = make_matrix<float>(index.size(), index.dimension());
    const std::vector<float>& stored = index.data();
    std::copy(stored.begin(), stored.end(), values.mutable_data());
    return values;
}

khoj::HnswIndex make_hnsw_index(std::size_t dimension,
                                std::size_t max_neighbors,
                                std::size_t max_neighbors_layer0,
                                std::size_t ef_construction,
                                double level_multiplier,
                                std::uint64_t seed,
                                khoj::Metric metric) {
    khoj::HnswParams params;
    params.dimension = dimension;
    params.max_neighbors = max_neighbors;
    params.max_neighbors_layer0 = max_neighbors_layer0;
    params.ef_construction = ef_construction;
    params.level_multiplier = level_multiplier;
    params.seed = seed;
    params.metric = metric;

    return khoj::HnswIndex(params);
}

void hnsw_add(khoj::HnswIndex& index, std::uint64_t label, const FloatArray& vector) {
    index.add(label, single_vector(vector, index.dimension(), "add"));
}

void hnsw_add_batch(khoj::HnswIndex& index,
                    const FloatArray& vectors,
                    const std::optional<LabelArray>& labels) {
    const std::size_t dimension = index.dimension();
    const std::size_t rows = row_count(vectors, dimension, "add_batch");

    if (labels.has_value()) {
        if (labels->ndim() != 1) {
            throw std::invalid_argument("add_batch expects a one-dimensional array of labels, got " +
                                        std::to_string(labels->ndim()) + " dimensions");
        }
        if (static_cast<std::size_t>(labels->shape(0)) != rows) {
            throw std::invalid_argument("add_batch expects one label per vector, got " +
                                        std::to_string(labels->shape(0)) + " labels for " +
                                        std::to_string(rows) + " vectors");
        }
    }
    if (rows == 0) {
        return;
    }

    const float* const values = vectors.data();
    const std::uint64_t* const label_values = labels.has_value() ? labels->data() : nullptr;
    const std::uint64_t first_label = static_cast<std::uint64_t>(index.size());

    index.reserve(index.size() + rows);
    for (std::size_t row = 0; row < rows; ++row) {
        const std::uint64_t label =
            label_values != nullptr ? label_values[row] : first_label + static_cast<std::uint64_t>(row);
        index.add(label, values + row * dimension);
    }
}

std::vector<khoj::SearchResult> hnsw_search(const khoj::HnswIndex& index,
                                            const FloatArray& query,
                                            std::size_t k,
                                            std::size_t ef_search) {
    const float* const values = single_vector(query, index.dimension(), "search");

    const py::gil_scoped_release release;
    return index.search(values, k, ef_search);
}

py::tuple hnsw_search_batch(const khoj::HnswIndex& index,
                            const FloatArray& queries,
                            std::size_t k,
                            std::size_t ef_search) {
    const std::size_t dimension = index.dimension();
    const std::size_t rows = row_count(queries, dimension, "search_batch");

    py::array_t<std::uint64_t> labels = make_matrix<std::uint64_t>(rows, k);
    py::array_t<float> distances = make_matrix<float>(rows, k);
    if (rows == 0 || k == 0) {
        return py::make_tuple(labels, distances);
    }

    const float* const query_values = queries.data();
    std::uint64_t* const label_data = labels.mutable_data();
    float* const distance_data = distances.mutable_data();
    const float missing = missing_distance(index.params().metric);

    {
        const py::gil_scoped_release release;
        for (std::size_t row = 0; row < rows; ++row) {
            write_row(index.search(query_values + row * dimension, k, ef_search), k, missing,
                      label_data + row * k, distance_data + row * k);
        }
    }

    return py::make_tuple(labels, distances);
}

}

PYBIND11_MODULE(khoj, m) {
    const khoj::HnswParams defaults;

    py::enum_<khoj::Metric>(m, "Metric")
        .value("L2", khoj::Metric::L2)
        .value("InnerProduct", khoj::Metric::InnerProduct);

    py::class_<khoj::HnswParams>(m, "HnswParams")
        .def(py::init<>())
        .def_readwrite("dimension", &khoj::HnswParams::dimension)
        .def_readwrite("max_neighbors", &khoj::HnswParams::max_neighbors)
        .def_readwrite("max_neighbors_layer0", &khoj::HnswParams::max_neighbors_layer0)
        .def_readwrite("ef_construction", &khoj::HnswParams::ef_construction)
        .def_readwrite("level_multiplier", &khoj::HnswParams::level_multiplier)
        .def_readwrite("seed", &khoj::HnswParams::seed)
        .def_readwrite("metric", &khoj::HnswParams::metric);

    py::class_<khoj::SearchResult>(m, "SearchResult")
        .def(py::init<>())
        .def_readwrite("label", &khoj::SearchResult::label)
        .def_readwrite("distance", &khoj::SearchResult::distance);

    py::class_<khoj::FlatIndex>(m, "FlatIndex")
        .def(py::init<std::size_t, khoj::Metric>(), py::arg("dimension"),
             py::arg("metric") = khoj::Metric::L2)
        .def("reserve", &khoj::FlatIndex::reserve, py::arg("count"))
        .def("add", &flat_add, py::arg("vector"))
        .def("add_batch", &flat_add_batch, py::arg("vectors"))
        .def("search", &flat_search, py::arg("query"), py::arg("k"))
        .def("search_batch", &flat_search_batch, py::arg("queries"), py::arg("k"))
        .def("to_numpy", &flat_to_numpy)
        .def("__len__", &khoj::FlatIndex::size)
        .def_property_readonly("size", &khoj::FlatIndex::size)
        .def_property_readonly("dimension", &khoj::FlatIndex::dimension)
        .def_property_readonly("metric", &khoj::FlatIndex::metric);

    py::class_<khoj::HnswIndex>(m, "HnswIndex")
        .def(py::init<const khoj::HnswParams&>(), py::arg("params"))
        .def(py::init(&make_hnsw_index),
             py::arg("dimension"),
             py::arg("max_neighbors") = defaults.max_neighbors,
             py::arg("max_neighbors_layer0") = defaults.max_neighbors_layer0,
             py::arg("ef_construction") = defaults.ef_construction,
             py::arg("level_multiplier") = defaults.level_multiplier,
             py::arg("seed") = defaults.seed,
             py::arg("metric") = defaults.metric)
        .def("reserve", &khoj::HnswIndex::reserve, py::arg("expected_elements"))
        .def("add", &hnsw_add, py::arg("label"), py::arg("vector"))
        .def("add_batch", &hnsw_add_batch, py::arg("vectors"), py::arg("labels") = py::none())
        .def("search", &hnsw_search, py::arg("query"), py::arg("k"), py::arg("ef_search"))
        .def("search_batch", &hnsw_search_batch, py::arg("queries"), py::arg("k"), py::arg("ef_search"))
        .def("save", &khoj::HnswIndex::save, py::arg("path"), py::call_guard<py::gil_scoped_release>())
        .def_static("load", &khoj::HnswIndex::load, py::arg("path"),
                    py::call_guard<py::gil_scoped_release>())
        .def("__len__", &khoj::HnswIndex::size)
        .def_property_readonly("size", &khoj::HnswIndex::size)
        .def_property_readonly("dimension", &khoj::HnswIndex::dimension)
        .def_property_readonly("max_level", &khoj::HnswIndex::max_level)
        .def_property_readonly("params", &khoj::HnswIndex::params, py::return_value_policy::copy);
}
