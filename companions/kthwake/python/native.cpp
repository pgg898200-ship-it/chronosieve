#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "kthwake.hpp"

namespace {
constexpr auto capsule_name = "kthwake.Index";

void release(PyObject* capsule) {
    delete static_cast<kthwake::Index*>(PyCapsule_GetPointer(capsule, capsule_name));
}

PyObject* translate_exception() {
    try { throw; }
    catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    catch (const std::invalid_argument& error) { PyErr_SetString(PyExc_ValueError, error.what()); }
    catch (const std::exception& error) { PyErr_SetString(PyExc_RuntimeError, error.what()); }
    return nullptr;
}

PyObject* create(PyObject*, PyObject* input) {
    PyObject* rules = PySequence_Tuple(input);
    if (!rules) return nullptr;
    std::vector<kthwake::Guard> guards;
    try {
        const auto n = PyTuple_GET_SIZE(rules);
        guards.reserve(static_cast<std::size_t>(n));
        for (Py_ssize_t i=0; i<n; ++i) {
            PyObject* row = PySequence_Tuple(PyTuple_GET_ITEM(rules, i));
            if (!row) break;
            if (PyTuple_GET_SIZE(row) != 2) {
                PyErr_SetString(PyExc_ValueError, "expected exactly two elements per guard");
                Py_DECREF(row);
                break;
            }
            auto k = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(row, 0));
            auto w = PyErr_Occurred() ? 0 : PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(row, 1));
            Py_DECREF(row);
            if (PyErr_Occurred()) break;
            if (k > UINT32_MAX) {
                PyErr_SetString(PyExc_OverflowError, "k exceeds UINT32_MAX");
                break;
            }
            guards.push_back({static_cast<std::uint32_t>(k), w});
        }
    } catch (...) {
        Py_DECREF(rules);
        return translate_exception();
    }
    Py_DECREF(rules);
    if (PyErr_Occurred()) return nullptr;
    try {
        auto* index = new kthwake::Index(guards);
        PyObject* capsule = PyCapsule_New(index, capsule_name, release);
        if (!capsule) delete index;
        return capsule;
    } catch (...) { return translate_exception(); }
}

PyObject* step(PyObject*, PyObject* args) {
    PyObject *handle, *time, *count;
    if (!PyArg_UnpackTuple(args, "step", 3, 3, &handle, &time, &count)) return nullptr;
    auto* index = static_cast<kthwake::Index*>(PyCapsule_GetPointer(handle, capsule_name));
    if (!index) return nullptr;
    auto now = PyLong_AsUnsignedLongLong(time);
    if (PyErr_Occurred()) return nullptr;
    auto arrivals = PyLong_AsUnsignedLongLong(count);
    if (PyErr_Occurred()) return nullptr;
    PyObject* result = PyList_New(0);
    if (!result) return nullptr;
    try {
        index->step(now, arrivals, [&](std::uint32_t id, bool state) {
            if (PyErr_Occurred()) return;
            PyObject* edge = Py_BuildValue("(IO)", static_cast<unsigned>(id), state ? Py_True : Py_False);
            if (edge) {
                PyList_Append(result, edge);
                Py_DECREF(edge);
            }
        });
    } catch (...) {
        Py_DECREF(result);
        return translate_exception();
    }
    if (PyErr_Occurred()) { Py_DECREF(result); return nullptr; }
    return result;
}

PyObject* deadline(PyObject*, PyObject* handle) {
    auto* index = static_cast<kthwake::Index*>(PyCapsule_GetPointer(handle, capsule_name));
    if (!index) return nullptr;
    const auto next = index->next_expiry();
    if (next) return PyLong_FromUnsignedLongLong(*next);
    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"create", create, METH_O, nullptr},
    {"step", step, METH_VARARGS, nullptr},
    {"next_expiry", deadline, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", nullptr, -1, methods,
                      nullptr, nullptr, nullptr, nullptr};
}
PyMODINIT_FUNC PyInit__native() { return PyModule_Create(&module); }
