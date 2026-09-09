#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "khoj/hnsw_index.hpp"

namespace py = pybind11;

PYBIND11_MODULE(khoj, m) {
    py::class_<khoj::HnswParams>(m, "HnswParams")
        .def(py::init<>())
        .def_readwrite("dimension", &khoj::HnswParams::dimension)
        .def_readwrite("max_neighbors", &khoj::HnswParams::max_neighbors)
        .def_readwrite("ef_construction", &khoj::HnswParams::ef_construction)
        .def_readwrite("seed", &khoj::HnswParams::seed);

    py::class_<khoj::SearchResult>(m, "SearchResult")
        .def(py::init<>())
        .def_readwrite("id", &khoj::SearchResult::id)
        .def_readwrite("distance", &khoj::SearchResult::distance);
}
