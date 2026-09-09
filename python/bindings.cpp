#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "khoj/hnsw_index.hpp"
#include "khoj/types.hpp"

namespace py = pybind11;

PYBIND11_MODULE(khoj, m) {
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
}
