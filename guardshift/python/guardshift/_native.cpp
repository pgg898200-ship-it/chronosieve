#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "guardshift.hpp"
#include <memory>

namespace {
using namespace guardshift;
struct PythonError {};
struct DecRef { void operator()(PyObject* obj) const { Py_XDECREF(obj); } };
using Owned = std::unique_ptr<PyObject, DecRef>;

Owned as_tuple(PyObject* object, Py_ssize_t size = -1) {
    Owned seq(PySequence_Tuple(object));
    if (!seq) throw PythonError{};
    if (size >= 0 && PyTuple_GET_SIZE(seq.get()) != size) {
        PyErr_SetString(PyExc_ValueError, "wrong number of fields in record");
        throw PythonError{};
    }
    return seq;
}

Value value(PyObject* object) {
    auto n = PyLong_AsLongLong(object);
    if (n == -1 && PyErr_Occurred()) throw PythonError{};
    return n;
}

Id id(PyObject* object) {
    auto n = value(object);
    if (n < 0 || static_cast<unsigned long long>(n) > std::numeric_limits<Id>::max())
        throw std::out_of_range("ID must fit uint32");
    return static_cast<Id>(n);
}

template<class Work>
PyObject* translate(Work work) {
    try { return work(); }
    catch (const PythonError&) {}
    catch (const std::bad_alloc&) { PyErr_NoMemory(); }
    catch (const std::out_of_range& e) { PyErr_SetString(PyExc_IndexError, e.what()); }
    catch (const std::exception& e) { PyErr_SetString(PyExc_ValueError, e.what()); }
    return nullptr;
}

constexpr auto capsule_name = "guardshift.Index";
Index* unpack(PyObject* capsule) {
    auto* idx = static_cast<Index*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (!idx) throw PythonError{};
    return idx;
}

void destroy(PyObject* capsule) {
    delete static_cast<Index*>(PyCapsule_GetPointer(capsule, capsule_name));
}

PyObject* build(PyObject*, PyObject* args) {
    PyObject *initial, *guards;
    if (!PyArg_ParseTuple(args, "OO", &initial, &guards)) return nullptr;
    return translate([&]() -> PyObject* {
        auto inputs = as_tuple(initial);
        std::vector<Value> values;
        for (Py_ssize_t i=0; i<PyTuple_GET_SIZE(inputs.get()); ++i)
            values.push_back(value(PyTuple_GET_ITEM(inputs.get(), i)));
        auto rows = as_tuple(guards);
        std::vector<std::vector<Term>> definitions(PyTuple_GET_SIZE(rows.get()));
        for (Py_ssize_t i=0; i<PyTuple_GET_SIZE(rows.get()); ++i) {
            auto row = as_tuple(PyTuple_GET_ITEM(rows.get(), i));
            for (Py_ssize_t j=0; j<PyTuple_GET_SIZE(row.get()); ++j) {
                auto term = as_tuple(PyTuple_GET_ITEM(row.get(), j), 3);
                definitions[i].push_back({id(PyTuple_GET_ITEM(term.get(), 0)),
                    value(PyTuple_GET_ITEM(term.get(), 1)), value(PyTuple_GET_ITEM(term.get(), 2))});
            }
        }
        auto index = std::make_unique<Index>(values, definitions);
        PyObject* capsule = PyCapsule_New(index.get(), capsule_name, destroy);
        if (capsule) index.release();
        return capsule;
    });
}

PyObject* apply(PyObject*, PyObject* args) {
    PyObject *capsule, *updates;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &updates)) return nullptr;
    return translate([&]() -> PyObject* {
        auto* index = unpack(capsule);
        auto rows = as_tuple(updates);
        std::vector<Update> patch;
        patch.reserve(PyTuple_GET_SIZE(rows.get()));
        for (Py_ssize_t i=0; i<PyTuple_GET_SIZE(rows.get()); ++i) {
            auto row = as_tuple(PyTuple_GET_ITEM(rows.get(), i), 2);
            patch.push_back({id(PyTuple_GET_ITEM(row.get(), 0)), value(PyTuple_GET_ITEM(row.get(), 1))});
        }
        auto changes = index->apply(patch);
        Owned result(PyList_New(static_cast<Py_ssize_t>(changes.size())));
        if (!result) return nullptr;
        for (size_t i=0; i<changes.size(); ++i) {
            auto& change = changes[i];
            auto* item = Py_BuildValue("(IO)", static_cast<unsigned>(change.guard), change.active ? Py_True : Py_False);
            if (!item) return nullptr;
            PyList_SET_ITEM(result.get(), static_cast<Py_ssize_t>(i), item);
        }
        return result.release();
    });
}

PyObject* active(PyObject*, PyObject* args) {
    PyObject *capsule, *number;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &number)) return nullptr;
    return translate([&]() { return PyBool_FromLong(unpack(capsule)->active(id(number))); });
}

PyMethodDef methods[] = {
    {"build", build, METH_VARARGS, "Compile a guard index."},
    {"apply", apply, METH_VARARGS, "Apply one atomic patch."},
    {"active", active, METH_VARARGS, "Read committed guard state."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", nullptr, -1, methods, nullptr, nullptr, nullptr, nullptr};
}  // namespace

PyMODINIT_FUNC PyInit__native() { return PyModule_Create(&module); }
