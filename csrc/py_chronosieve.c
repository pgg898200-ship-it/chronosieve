#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "chronosieve.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Zero-Copy Py_buffer Helpers (works with array.array, memoryview, NumPy)
 * ========================================================================= */

typedef struct {
    Py_buffer view;
    double *data;
    size_t len;
    bool has_view;
    bool owns_malloc;
} cs_py_f64_buf_t;

typedef struct {
    Py_buffer view;
    uint8_t *data;
    size_t len;
    bool has_view;
    bool owns_malloc;
} cs_py_u8_buf_t;

typedef struct {
    Py_buffer view;
    uint32_t *data;
    size_t len;
    bool has_view;
    bool owns_malloc;
} cs_py_u32_buf_t;

static void cs_py_f64_release(cs_py_f64_buf_t *b) {
    if (b->has_view) {
        PyBuffer_Release(&b->view);
        b->has_view = false;
    }
    if (b->owns_malloc) {
        free(b->data);
        b->owns_malloc = false;
    }
    b->data = NULL;
    b->len = 0U;
}

static void cs_py_u8_release(cs_py_u8_buf_t *b) {
    if (b->has_view) {
        PyBuffer_Release(&b->view);
        b->has_view = false;
    }
    if (b->owns_malloc) {
        free(b->data);
        b->owns_malloc = false;
    }
    b->data = NULL;
    b->len = 0U;
}

static void cs_py_u32_release(cs_py_u32_buf_t *b) {
    if (b->has_view) {
        PyBuffer_Release(&b->view);
        b->has_view = false;
    }
    if (b->owns_malloc) {
        free(b->data);
        b->owns_malloc = false;
    }
    b->data = NULL;
    b->len = 0U;
}

#if defined(_WIN32) || (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__))
#define CS_IS_LITTLE_ENDIAN 1
#else
#define CS_IS_LITTLE_ENDIAN 0
#endif

static bool cs_fmt_is_f64(const char *fmt, Py_ssize_t itemsize) {
    if (itemsize != (Py_ssize_t)sizeof(double)) {
        return false;
    }
    if (!fmt) {
        return true;
    }
    if (strcmp(fmt, "d") == 0 || strcmp(fmt, "=d") == 0 || strcmp(fmt, "@d") == 0) {
        return true;
    }
#if CS_IS_LITTLE_ENDIAN
    if (strcmp(fmt, "<d") == 0) {
        return true;
    }
#endif
    return false;
}

static bool cs_fmt_is_u8(const char *fmt, Py_ssize_t itemsize) {
    if (itemsize != 1) {
        return false;
    }
    if (!fmt) {
        return true;
    }
    return (strcmp(fmt, "B") == 0 || strcmp(fmt, "?") == 0 || strcmp(fmt, "b") == 0 ||
            strcmp(fmt, "=B") == 0 || strcmp(fmt, "<B") == 0 || strcmp(fmt, ">B") == 0);
}

static bool cs_fmt_is_u32(const char *fmt, Py_ssize_t itemsize) {
    if (itemsize != 4) {
        return false;
    }
    if (!fmt) {
        return true;
    }
    if (strcmp(fmt, "I") == 0 || strcmp(fmt, "i") == 0 ||
        strcmp(fmt, "=I") == 0 || strcmp(fmt, "=i") == 0 ||
        strcmp(fmt, "@I") == 0 || strcmp(fmt, "@i") == 0 ||
        (sizeof(unsigned long) == 4 && strcmp(fmt, "L") == 0)) {
        return true;
    }
#if CS_IS_LITTLE_ENDIAN
    if (strcmp(fmt, "<I") == 0 || strcmp(fmt, "<i") == 0) {
        return true;
    }
#endif
    return false;
}

static int cs_check_u32(long long val, const char *name, uint32_t *out) {
    if (val < 0 || val > (long long)UINT32_MAX) {
        PyErr_Format(PyExc_ValueError, "%s must be between 0 and %u", name, (unsigned)UINT32_MAX);
        return -1;
    }
    *out = (uint32_t)val;
    return 0;
}

/* Acquire read-only float64 buffer (zero-copy for C-contiguous buffer; strided copy; sequence fallback). */
static int cs_py_acquire_f64_ro(PyObject *obj, cs_py_f64_buf_t *out) {
    memset(out, 0, sizeof(*out));
    if (PyObject_CheckBuffer(obj)) {
        if (PyObject_GetBuffer(obj, &out->view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) == 0) {
            if (cs_fmt_is_f64(out->view.format, out->view.itemsize)) {
                out->has_view = true;
                out->data = (double *)out->view.buf;
                out->len = (size_t)(out->view.len / (Py_ssize_t)sizeof(double));
                return 0;
            }
            PyBuffer_Release(&out->view);
        } else {
            PyErr_Clear();
            Py_buffer sview;
            if (PyObject_GetBuffer(obj, &sview, PyBUF_FORMAT | PyBUF_STRIDES) == 0) {
                if (cs_fmt_is_f64(sview.format, sview.itemsize)) {
                    size_t n = (size_t)(sview.len / (Py_ssize_t)sizeof(double));
                    if (n == 0U) {
                        PyBuffer_Release(&sview);
                        return 0;
                    }
                    double *buf = (double *)malloc((size_t)sview.len);
                    if (!buf) {
                        PyBuffer_Release(&sview);
                        PyErr_NoMemory();
                        return -1;
                    }
                    if (PyBuffer_ToContiguous(buf, &sview, sview.len, 'C') != 0) {
                        free(buf);
                        PyBuffer_Release(&sview);
                        return -1;
                    }
                    PyBuffer_Release(&sview);
                    out->data = buf;
                    out->len = n;
                    out->owns_malloc = true;
                    return 0;
                }
                PyBuffer_Release(&sview);
            } else {
                PyErr_Clear();
            }
        }
    }

    PyObject *seq = PySequence_Fast(obj, "expected float64 buffer or numeric sequence");
    if (!seq) {
        return -1;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n == 0) {
        Py_DECREF(seq);
        return 0;
    }
    double *buf = (double *)malloc((size_t)n * sizeof(double));
    if (!buf) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return -1;
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        double v = PyFloat_AsDouble(items[i]);
        if (v == -1.0 && PyErr_Occurred()) {
            free(buf);
            Py_DECREF(seq);
            return -1;
        }
        buf[i] = v;
    }
    Py_DECREF(seq);
    out->data = buf;
    out->len = (size_t)n;
    out->owns_malloc = true;
    return 0;
}

/* Acquire writable zero-copy float64 buffer (required for out_* parameters). */
static int cs_py_acquire_f64_rw(PyObject *obj, size_t expected_len, cs_py_f64_buf_t *out) {
    memset(out, 0, sizeof(*out));
    if (PyObject_GetBuffer(obj, &out->view, PyBUF_WRITABLE | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) != 0) {
        return -1;
    }
    if (!cs_fmt_is_f64(out->view.format, out->view.itemsize)) {
        PyBuffer_Release(&out->view);
        PyErr_SetString(PyExc_TypeError, "output buffer must be C-contiguous float64 ('d')");
        return -1;
    }
    size_t len = (size_t)(out->view.len / (Py_ssize_t)sizeof(double));
    if (len < expected_len) {
        PyBuffer_Release(&out->view);
        PyErr_Format(PyExc_ValueError, "output float64 buffer too small (%zu < %zu)", len, expected_len);
        return -1;
    }
    out->has_view = true;
    out->data = (double *)out->view.buf;
    out->len = len;
    return 0;
}

static int cs_py_acquire_u8_ro(PyObject *obj, cs_py_u8_buf_t *out) {
    memset(out, 0, sizeof(*out));
    if (PyObject_CheckBuffer(obj)) {
        if (PyObject_GetBuffer(obj, &out->view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) == 0) {
            if (cs_fmt_is_u8(out->view.format, out->view.itemsize)) {
                out->has_view = true;
                out->data = (uint8_t *)out->view.buf;
                out->len = (size_t)out->view.len;
                return 0;
            }
            PyBuffer_Release(&out->view);
        } else {
            PyErr_Clear();
            Py_buffer sview;
            if (PyObject_GetBuffer(obj, &sview, PyBUF_FORMAT | PyBUF_STRIDES) == 0) {
                if (cs_fmt_is_u8(sview.format, sview.itemsize)) {
                    size_t n = (size_t)sview.len;
                    if (n == 0U) {
                        PyBuffer_Release(&sview);
                        return 0;
                    }
                    uint8_t *buf = (uint8_t *)malloc(n);
                    if (!buf) {
                        PyBuffer_Release(&sview);
                        PyErr_NoMemory();
                        return -1;
                    }
                    if (PyBuffer_ToContiguous(buf, &sview, sview.len, 'C') != 0) {
                        free(buf);
                        PyBuffer_Release(&sview);
                        return -1;
                    }
                    PyBuffer_Release(&sview);
                    out->data = buf;
                    out->len = n;
                    out->owns_malloc = true;
                    return 0;
                }
                PyBuffer_Release(&sview);
            } else {
                PyErr_Clear();
            }
        }
    }

    PyObject *seq = PySequence_Fast(obj, "expected uint8/bool buffer or sequence");
    if (!seq) {
        return -1;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n == 0) {
        Py_DECREF(seq);
        return 0;
    }
    uint8_t *buf = (uint8_t *)malloc((size_t)n);
    if (!buf) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return -1;
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        int truth = PyObject_IsTrue(items[i]);
        if (truth < 0) {
            free(buf);
            Py_DECREF(seq);
            return -1;
        }
        buf[i] = truth ? 1U : 0U;
    }
    Py_DECREF(seq);
    out->data = buf;
    out->len = (size_t)n;
    out->owns_malloc = true;
    return 0;
}

static int cs_py_acquire_u8_rw(PyObject *obj, size_t expected_len, cs_py_u8_buf_t *out) {
    memset(out, 0, sizeof(*out));
    if (PyObject_GetBuffer(obj, &out->view, PyBUF_WRITABLE | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) != 0) {
        return -1;
    }
    if (!cs_fmt_is_u8(out->view.format, out->view.itemsize)) {
        PyBuffer_Release(&out->view);
        PyErr_SetString(PyExc_TypeError, "output buffer must be C-contiguous uint8/bool ('B')");
        return -1;
    }
    size_t len = (size_t)out->view.len;
    if (len < expected_len) {
        PyBuffer_Release(&out->view);
        PyErr_Format(PyExc_ValueError, "output uint8 buffer too small (%zu < %zu)", len, expected_len);
        return -1;
    }
    out->has_view = true;
    out->data = (uint8_t *)out->view.buf;
    out->len = len;
    return 0;
}

static int cs_py_acquire_u32_ro(PyObject *obj, cs_py_u32_buf_t *out) {
    memset(out, 0, sizeof(*out));
    if (PyObject_CheckBuffer(obj)) {
        if (PyObject_GetBuffer(obj, &out->view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) == 0) {
            if (cs_fmt_is_u32(out->view.format, out->view.itemsize)) {
                out->has_view = true;
                out->data = (uint32_t *)out->view.buf;
                out->len = (size_t)(out->view.len / 4);
                return 0;
            }
            PyBuffer_Release(&out->view);
        } else {
            PyErr_Clear();
            Py_buffer sview;
            if (PyObject_GetBuffer(obj, &sview, PyBUF_FORMAT | PyBUF_STRIDES) == 0) {
                if (cs_fmt_is_u32(sview.format, sview.itemsize)) {
                    size_t n = (size_t)(sview.len / 4);
                    if (n == 0U) {
                        PyBuffer_Release(&sview);
                        return 0;
                    }
                    uint32_t *buf = (uint32_t *)malloc((size_t)sview.len);
                    if (!buf) {
                        PyBuffer_Release(&sview);
                        PyErr_NoMemory();
                        return -1;
                    }
                    if (PyBuffer_ToContiguous(buf, &sview, sview.len, 'C') != 0) {
                        free(buf);
                        PyBuffer_Release(&sview);
                        return -1;
                    }
                    PyBuffer_Release(&sview);
                    out->data = buf;
                    out->len = n;
                    out->owns_malloc = true;
                    return 0;
                }
                PyBuffer_Release(&sview);
            } else {
                PyErr_Clear();
            }
        }
    }

    PyObject *seq = PySequence_Fast(obj, "expected uint32 buffer or integer sequence");
    if (!seq) {
        return -1;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n == 0) {
        Py_DECREF(seq);
        return 0;
    }
    uint32_t *buf = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    if (!buf) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return -1;
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        unsigned long v = PyLong_AsUnsignedLong(items[i]);
        if (v == (unsigned long)-1 && PyErr_Occurred()) {
            free(buf);
            Py_DECREF(seq);
            return -1;
        }
        buf[i] = (uint32_t)v;
    }
    Py_DECREF(seq);
    out->data = buf;
    out->len = (size_t)n;
    out->owns_malloc = true;
    return 0;
}

static void cs_set_py_error(cs_status_t st, const char *ctx) {
    switch (st) {
        case CS_ERR_NONFINITE:
            PyErr_Format(PyExc_ValueError, "%s: non-finite value (NaN/Inf)", ctx);
            break;
        case CS_ERR_NOMEM:
            PyErr_NoMemory();
            break;
        case CS_ERR_EMPTY:
            PyErr_Format(PyExc_ValueError, "%s: empty state", ctx);
            break;
        case CS_ERR_BUFFER:
            PyErr_Format(PyExc_ValueError, "%s: output buffer too small", ctx);
            break;
        default:
            PyErr_Format(PyExc_ValueError, "%s: invalid argument", ctx);
            break;
    }
}

static int parse_family_str(const char *s, cs_edet_family_t *out) {
    if (!s || strcmp(s, "auto") == 0) { *out = CS_FAMILY_AUTO; return 0; }
    if (strcmp(s, "bounded") == 0) { *out = CS_FAMILY_BOUNDED; return 0; }
    if (strcmp(s, "catoni") == 0) { *out = CS_FAMILY_CATONI; return 0; }
    if (strcmp(s, "symmetry") == 0) { *out = CS_FAMILY_SYMMETRY; return 0; }
    if (strcmp(s, "vovk_rank") == 0 || strcmp(s, "rank") == 0 || strcmp(s, "vovk") == 0) {
        *out = CS_FAMILY_VOVK_RANK;
        return 0;
    }
    PyErr_Format(PyExc_ValueError, "unknown family '%s'", s);
    return -1;
}

static const char *family_to_str(cs_edet_family_t f) {
    switch (f) {
        case CS_FAMILY_BOUNDED: return "bounded";
        case CS_FAMILY_CATONI: return "catoni";
        case CS_FAMILY_SYMMETRY: return "symmetry";
        case CS_FAMILY_VOVK_RANK: return "vovk_rank";
        default: return "auto";
    }
}

static int parse_accum_str(const char *s, cs_edet_accum_t *out) {
    if (!s || strcmp(s, "esr") == 0 || strcmp(s, "sr") == 0) { *out = CS_ACCUM_ESR; return 0; }
    if (strcmp(s, "ecusum") == 0 || strcmp(s, "cusum") == 0) { *out = CS_ACCUM_ECUSUM; return 0; }
    PyErr_Format(PyExc_ValueError, "unknown accumulator '%s'", s);
    return -1;
}

static int parse_side_str(const char *s, cs_edet_side_t *out) {
    if (!s || strcmp(s, "two_sided") == 0 || strcmp(s, "both") == 0) {
        *out = CS_SIDE_TWO_SIDED;
        return 0;
    }
    if (strcmp(s, "upper") == 0) { *out = CS_SIDE_UPPER; return 0; }
    if (strcmp(s, "lower") == 0) { *out = CS_SIDE_LOWER; return 0; }
    PyErr_Format(PyExc_ValueError, "unknown side '%s'", s);
    return -1;
}

static PyObject *edet_result_to_dict(const cs_edet_result_t *r) {
    return Py_BuildValue(
        "{s:d,s:d,s:d,s:d,s:K,s:K,s:k,s:O}",
        "stat", r->stat,
        "log_stat", r->log_stat,
        "e_mean", r->last_e_mean,
        "p_value", r->last_p_value,
        "step_count", (unsigned long long)r->step_count,
        "alarm_count", (unsigned long long)r->alarm_count,
        "epoch_count", (unsigned long)r->epoch_count,
        "alarmed", r->alarmed ? Py_True : Py_False);
}

#define CS_CHECK_INIT(ptr, type_name)                                          \
    do {                                                                       \
        if (!(ptr)) {                                                          \
            PyErr_SetString(PyExc_RuntimeError, type_name " is not initialized"); \
            return NULL;                                                       \
        }                                                                      \
    } while (0)

/* =========================================================================
 * 1. EDetector Python Type
 * ========================================================================= */

typedef struct {
    PyObject_HEAD
    cs_edetector_t *det;
    double threshold;
} PyEDetectorObject;

static void PyEDetector_dealloc(PyEDetectorObject *self) {
    cs_edet_destroy(self->det);
    self->det = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int PyEDetector_init(PyEDetectorObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {
        "family", "accum", "side", "threshold", "mu0", "sigma",
        "max_prefix", "seed", "auto_reset", NULL
    };
    const char *family_s = "auto";
    const char *accum_s = "esr";
    const char *side_s = "two_sided";
    double threshold = 1000.0;
    double mu0 = NAN;
    double sigma = NAN;
    long long max_prefix = 65536LL;
    unsigned long long seed = 0x517cc1b727220a95ULL;
    int auto_reset = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "|sssdddLKp", kwlist,
            &family_s, &accum_s, &side_s, &threshold, &mu0, &sigma,
            &max_prefix, &seed, &auto_reset)) {
        return -1;
    }

    cs_edet_config_t cfg;
    cs_edet_config_default(&cfg);
    if (parse_family_str(family_s, &cfg.family) != 0 ||
        parse_accum_str(accum_s, &cfg.accum) != 0 ||
        parse_side_str(side_s, &cfg.side) != 0 ||
        cs_check_u32(max_prefix, "max_prefix", &cfg.max_prefix) != 0) {
        return -1;
    }
    cfg.threshold = threshold;
    cfg.mu0 = mu0;
    cfg.sigma = sigma;
    cfg.seed = (uint64_t)seed;
    cfg.auto_reset = (auto_reset != 0);

    cs_edetector_t *new_det = cs_edet_create(&cfg);
    if (!new_det) {
        PyErr_SetString(PyExc_ValueError, "invalid EDetector configuration");
        return -1;
    }
    cs_edet_destroy(self->det);
    self->det = new_det;
    self->threshold = threshold;
    return 0;
}

static PyObject *PyEDetector_calibrate_catoni(PyObject *cls, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"warmup", "threshold", "side", "inflation_z", "auto_reset", NULL};
    PyObject *warmup_obj = NULL;
    double threshold = 1000.0;
    const char *side_s = "two_sided";
    double inflation_z = 2.5;
    int auto_reset = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O|dsdp", kwlist,
            &warmup_obj, &threshold, &side_s, &inflation_z, &auto_reset)) {
        return NULL;
    }
    cs_edet_side_t side;
    if (parse_side_str(side_s, &side) != 0) {
        return NULL;
    }
    cs_py_f64_buf_t wbuf;
    if (cs_py_acquire_f64_ro(warmup_obj, &wbuf) != 0) {
        return NULL;
    }
    cs_edet_config_t cfg;
    cs_status_t st = cs_edet_calibrate_catoni(wbuf.data, wbuf.len, threshold, side, inflation_z, &cfg);
    cs_py_f64_release(&wbuf);
    if (st != CS_OK) {
        cs_set_py_error(st, "EDetector.calibrate_catoni");
        return NULL;
    }
    cfg.auto_reset = (auto_reset != 0);

    PyEDetectorObject *obj = (PyEDetectorObject *)((PyTypeObject *)cls)->tp_alloc((PyTypeObject *)cls, 0);
    if (!obj) {
        return NULL;
    }
    obj->det = cs_edet_create(&cfg);
    obj->threshold = threshold;
    if (!obj->det) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_RuntimeError, "failed to create calibrated Catoni EDetector");
        return NULL;
    }
    return (PyObject *)obj;
}

static PyObject *PyEDetector_step(PyEDetectorObject *self, PyObject *args) {
    CS_CHECK_INIT(self->det, "EDetector");
    double x = 0.0;
    if (!PyArg_ParseTuple(args, "d", &x)) {
        return NULL;
    }
    cs_edet_result_t res;
    cs_status_t st = cs_edet_step(self->det, x, &res);
    if (st != CS_OK) {
        cs_set_py_error(st, "EDetector.step");
        return NULL;
    }
    return edet_result_to_dict(&res);
}

static PyObject *PyEDetector_ingest_batch(PyEDetectorObject *self, PyObject *args, PyObject *kwds) {
    CS_CHECK_INIT(self->det, "EDetector");
    static char *kwlist[] = {"xs", "out_stats", "out_alarms", NULL};
    PyObject *xs_obj = NULL;
    PyObject *stats_obj = Py_None;
    PyObject *alarms_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", kwlist, &xs_obj, &stats_obj, &alarms_obj)) {
        return NULL;
    }

    cs_py_f64_buf_t xs_buf;
    cs_py_f64_buf_t st_buf;
    cs_py_u8_buf_t al_buf;
    memset(&st_buf, 0, sizeof(st_buf));
    memset(&al_buf, 0, sizeof(al_buf));

    if (cs_py_acquire_f64_ro(xs_obj, &xs_buf) != 0) {
        return NULL;
    }
    if (stats_obj != Py_None && cs_py_acquire_f64_rw(stats_obj, xs_buf.len, &st_buf) != 0) {
        cs_py_f64_release(&xs_buf);
        return NULL;
    }
    if (alarms_obj != Py_None && cs_py_acquire_u8_rw(alarms_obj, xs_buf.len, &al_buf) != 0) {
        cs_py_f64_release(&st_buf);
        cs_py_f64_release(&xs_buf);
        return NULL;
    }

    cs_edet_result_t last;
    memset(&last, 0, sizeof(last));
    cs_status_t st;
    Py_BEGIN_ALLOW_THREADS
    st = cs_edet_step_batch(self->det, xs_buf.data, xs_buf.len, st_buf.data, al_buf.data, &last);
    Py_END_ALLOW_THREADS

    cs_py_u8_release(&al_buf);
    cs_py_f64_release(&st_buf);
    cs_py_f64_release(&xs_buf);

    if (st != CS_OK) {
        cs_set_py_error(st, "EDetector.ingest_batch");
        return NULL;
    }
    return edet_result_to_dict(&last);
}

static PyObject *PyEDetector_reset(PyEDetectorObject *self, PyObject *Py_UNUSED(ignored)) {
    CS_CHECK_INIT(self->det, "EDetector");
    cs_edet_reset(self->det);
    Py_RETURN_NONE;
}

static PyObject *PyEDetector_reset_accum(PyEDetectorObject *self, PyObject *Py_UNUSED(ignored)) {
    CS_CHECK_INIT(self->det, "EDetector");
    cs_edet_reset_accum_only(self->det);
    Py_RETURN_NONE;
}

static PyObject *PyEDetector_get_stat(PyEDetectorObject *self, void *Py_UNUSED(c)) {
    CS_CHECK_INIT(self->det, "EDetector");
    return PyFloat_FromDouble(cs_edet_current_stat(self->det));
}

static PyObject *PyEDetector_get_step_count(PyEDetectorObject *self, void *Py_UNUSED(c)) {
    CS_CHECK_INIT(self->det, "EDetector");
    return PyLong_FromUnsignedLongLong(cs_edet_step_count(self->det));
}

static PyObject *PyEDetector_get_family(PyEDetectorObject *self, void *Py_UNUSED(c)) {
    CS_CHECK_INIT(self->det, "EDetector");
    return PyUnicode_FromString(family_to_str(cs_edet_effective_family(self->det)));
}

static PyObject *PyEDetector_get_threshold(PyEDetectorObject *self, void *Py_UNUSED(c)) {
    CS_CHECK_INIT(self->det, "EDetector");
    return PyFloat_FromDouble(self->threshold);
}

static PyMethodDef PyEDetector_methods[] = {
    {"step", (PyCFunction)PyEDetector_step, METH_VARARGS, "Process a single scalar observation."},
    {"ingest_batch", (PyCFunction)(void (*)(void))PyEDetector_ingest_batch,
     METH_VARARGS | METH_KEYWORDS, "Zero-copy batch ingestion over a float64 buffer."},
    {"calibrate_catoni", (PyCFunction)(void (*)(void))PyEDetector_calibrate_catoni,
     METH_CLASS | METH_VARARGS | METH_KEYWORDS,
     "Calibrate an Algebraic Catoni EDetector from a warmup buffer with variance inflation."},
    {"reset", (PyCFunction)PyEDetector_reset, METH_NOARGS, "Reset detector accumulators and prefix tree."},
    {"reset_accum", (PyCFunction)PyEDetector_reset_accum, METH_NOARGS,
     "Reset only accumulator lanes while preserving the Vovk Rank B-tree baseline."},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef PyEDetector_getset[] = {
    {"stat", (getter)PyEDetector_get_stat, NULL, "Current E-detector mixture statistic.", NULL},
    {"step_count", (getter)PyEDetector_get_step_count, NULL, "Total steps processed.", NULL},
    {"effective_family", (getter)PyEDetector_get_family, NULL, "Resolved e-value family.", NULL},
    {"threshold", (getter)PyEDetector_get_threshold, NULL, "ARL lower-bound threshold.", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject PyEDetectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "chronosieve._core.EDetector",
    .tp_basicsize = sizeof(PyEDetectorObject),
    .tp_dealloc = (destructor)PyEDetector_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Zero-exp SIMD E-Detector (Bounded, Catoni, Symmetry, Vovk Rank B-tree).",
    .tp_methods = PyEDetector_methods,
    .tp_getset = PyEDetector_getset,
    .tp_init = (initproc)PyEDetector_init,
    .tp_new = PyType_GenericNew,
};

/* =========================================================================
 * 2. EFusion Python Type
 * ========================================================================= */

typedef struct {
    PyObject_HEAD
    cs_efusion_t *fus;
    size_t n_agents;
    cs_efusion_mode_t mode;
} PyEFusionObject;

static void PyEFusion_dealloc(PyEFusionObject *self) {
    cs_efusion_destroy(self->fus);
    self->fus = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int PyEFusion_init(PyEFusionObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {
        "n_agents", "mode", "accum", "family", "side", "threshold",
        "mu0", "sigma", "weights", "max_prefix", "seed", "auto_reset", NULL
    };
    Py_ssize_t n_agents = 0;
    const char *mode_s = "step";
    const char *accum_s = "esr";
    const char *family_s = "catoni";
    const char *side_s = "two_sided";
    double threshold = 1000.0;
    double mu0 = 0.0;
    double sigma = 1.0;
    PyObject *weights_obj = Py_None;
    long long max_prefix_ll = 65536LL;
    unsigned long long seed = 0x517cc1b727220a95ULL;
    int auto_reset = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "n|ssssdddOLKp", kwlist,
            &n_agents, &mode_s, &accum_s, &family_s, &side_s, &threshold,
            &mu0, &sigma, &weights_obj, &max_prefix_ll, &seed, &auto_reset)) {
        return -1;
    }
    if (n_agents <= 0) {
        PyErr_SetString(PyExc_ValueError, "n_agents must be > 0");
        return -1;
    }

    cs_efusion_mode_t mode;
    if (strcmp(mode_s, "step") == 0) {
        mode = CS_FUSION_STEP;
    } else if (strcmp(mode_s, "process") == 0) {
        mode = CS_FUSION_PROCESS;
    } else {
        PyErr_Format(PyExc_ValueError, "unknown fusion mode '%s'", mode_s);
        return -1;
    }

    cs_edet_accum_t accum;
    cs_edet_family_t family;
    cs_edet_side_t side;
    uint32_t max_prefix = 0U;
    if (parse_accum_str(accum_s, &accum) != 0 ||
        parse_family_str(family_s, &family) != 0 ||
        parse_side_str(side_s, &side) != 0 ||
        cs_check_u32(max_prefix_ll, "max_prefix", &max_prefix) != 0) {
        return -1;
    }

    cs_py_f64_buf_t wbuf;
    memset(&wbuf, 0, sizeof(wbuf));
    if (weights_obj != Py_None) {
        if (cs_py_acquire_f64_ro(weights_obj, &wbuf) != 0) {
            return -1;
        }
        if (wbuf.len != (size_t)n_agents) {
            cs_py_f64_release(&wbuf);
            PyErr_SetString(PyExc_ValueError, "weights length must equal n_agents");
            return -1;
        }
    }

    cs_edet_config_t *acfg = (cs_edet_config_t *)malloc((size_t)n_agents * sizeof(*acfg));
    if (!acfg) {
        cs_py_f64_release(&wbuf);
        PyErr_NoMemory();
        return -1;
    }
    for (size_t k = 0; k < (size_t)n_agents; ++k) {
        cs_edet_config_default(&acfg[k]);
        acfg[k].family = family;
        acfg[k].accum = accum;
        acfg[k].side = side;
        acfg[k].threshold = threshold;
        acfg[k].mu0 = mu0;
        acfg[k].sigma = sigma;
        acfg[k].max_prefix = max_prefix;
        acfg[k].seed = (uint64_t)seed ^ ((uint64_t)(k + 1U) * 0x9e3779b97f4a7c15ULL);
    }

    cs_efusion_config_t fcfg;
    fcfg.n_agents = (size_t)n_agents;
    fcfg.mode = mode;
    fcfg.accum = accum;
    fcfg.threshold = threshold;
    fcfg.auto_reset = (auto_reset != 0);
    fcfg.agent_configs = acfg;
    fcfg.weights = (weights_obj != Py_None) ? wbuf.data : NULL;

    cs_efusion_t *new_fus = cs_efusion_create(&fcfg);
    free(acfg);
    cs_py_f64_release(&wbuf);

    if (!new_fus) {
        PyErr_SetString(PyExc_ValueError, "invalid EFusion configuration");
        return -1;
    }
    cs_efusion_destroy(self->fus);
    self->fus = new_fus;
    self->n_agents = (size_t)n_agents;
    self->mode = mode;
    return 0;
}

static PyObject *PyEFusion_step(PyEFusionObject *self, PyObject *args, PyObject *kwds) {
    CS_CHECK_INIT(self->fus, "EFusion");
    static char *kwlist[] = {"x_agents", "weights", NULL};
    PyObject *x_obj = NULL;
    PyObject *w_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &x_obj, &w_obj)) {
        return NULL;
    }
    if (self->mode == CS_FUSION_PROCESS && w_obj != Py_None) {
        PyErr_SetString(
            PyExc_ValueError,
            "per-step weights override is only valid in EFusion(mode='step'); "
            "mode='process' requires fixed prior weights to guarantee ARL >= gamma");
        return NULL;
    }

    cs_py_f64_buf_t xbuf, wbuf;
    memset(&wbuf, 0, sizeof(wbuf));
    if (cs_py_acquire_f64_ro(x_obj, &xbuf) != 0) {
        return NULL;
    }
    if (xbuf.len != self->n_agents) {
        cs_py_f64_release(&xbuf);
        PyErr_SetString(PyExc_ValueError, "x_agents length must equal n_agents");
        return NULL;
    }
    if (w_obj != Py_None) {
        if (cs_py_acquire_f64_ro(w_obj, &wbuf) != 0) {
            cs_py_f64_release(&xbuf);
            return NULL;
        }
        if (wbuf.len != self->n_agents) {
            cs_py_f64_release(&wbuf);
            cs_py_f64_release(&xbuf);
            PyErr_SetString(PyExc_ValueError, "weights length must equal n_agents");
            return NULL;
        }
    }

    cs_edet_result_t res;
    cs_status_t st = cs_efusion_step(self->fus, xbuf.data, w_obj != Py_None ? wbuf.data : NULL, &res);
    cs_py_f64_release(&wbuf);
    cs_py_f64_release(&xbuf);
    if (st != CS_OK) {
        cs_set_py_error(st, "EFusion.step");
        return NULL;
    }
    return edet_result_to_dict(&res);
}

static PyObject *PyEFusion_ingest_batch(PyEFusionObject *self, PyObject *args, PyObject *kwds) {
    CS_CHECK_INIT(self->fus, "EFusion");
    static char *kwlist[] = {"x_matrix", "out_stats", "out_alarms", NULL};
    PyObject *x_obj = NULL;
    PyObject *st_obj = Py_None;
    PyObject *al_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", kwlist, &x_obj, &st_obj, &al_obj)) {
        return NULL;
    }

    cs_py_f64_buf_t xbuf, sbuf;
    cs_py_u8_buf_t abuf;
    memset(&sbuf, 0, sizeof(sbuf));
    memset(&abuf, 0, sizeof(abuf));

    if (cs_py_acquire_f64_ro(x_obj, &xbuf) != 0) {
        return NULL;
    }
    if (xbuf.len % self->n_agents != 0U) {
        cs_py_f64_release(&xbuf);
        PyErr_SetString(PyExc_ValueError, "x_matrix size must be a multiple of n_agents");
        return NULL;
    }
    size_t n_steps = xbuf.len / self->n_agents;

    if (st_obj != Py_None && cs_py_acquire_f64_rw(st_obj, n_steps, &sbuf) != 0) {
        cs_py_f64_release(&xbuf);
        return NULL;
    }
    if (al_obj != Py_None && cs_py_acquire_u8_rw(al_obj, n_steps, &abuf) != 0) {
        cs_py_f64_release(&sbuf);
        cs_py_f64_release(&xbuf);
        return NULL;
    }

    cs_edet_result_t last;
    memset(&last, 0, sizeof(last));
    cs_status_t st;
    Py_BEGIN_ALLOW_THREADS
    st = cs_efusion_step_batch(self->fus, xbuf.data, n_steps, sbuf.data, abuf.data, &last);
    Py_END_ALLOW_THREADS

    cs_py_u8_release(&abuf);
    cs_py_f64_release(&sbuf);
    cs_py_f64_release(&xbuf);

    if (st != CS_OK) {
        cs_set_py_error(st, "EFusion.ingest_batch");
        return NULL;
    }
    return edet_result_to_dict(&last);
}

static PyObject *PyEFusion_reset(PyEFusionObject *self, PyObject *Py_UNUSED(ignored)) {
    CS_CHECK_INIT(self->fus, "EFusion");
    cs_efusion_reset(self->fus);
    Py_RETURN_NONE;
}

static PyMethodDef PyEFusion_methods[] = {
    {"step", (PyCFunction)(void (*)(void))PyEFusion_step, METH_VARARGS | METH_KEYWORDS,
     "Process one multi-agent observation vector."},
    {"ingest_batch", (PyCFunction)(void (*)(void))PyEFusion_ingest_batch,
     METH_VARARGS | METH_KEYWORDS, "Zero-copy batch multi-agent E-fusion."},
    {"reset", (PyCFunction)PyEFusion_reset, METH_NOARGS, "Reset all agent detectors and fusion state."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject PyEFusionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "chronosieve._core.EFusion",
    .tp_basicsize = sizeof(PyEFusionObject),
    .tp_dealloc = (destructor)PyEFusion_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Cross-agent E-Fusion valid under arbitrary unknown cross-agent correlation.",
    .tp_methods = PyEFusion_methods,
    .tp_init = (initproc)PyEFusion_init,
    .tp_new = PyType_GenericNew,
};

/* =========================================================================
 * 3. ConformalPID & ConformalBatchPID Python Types
 * ========================================================================= */

typedef struct {
    PyObject_HEAD
    cs_conformal_pid_t *cp;
} PyConformalPIDObject;

static void PyConformalPID_dealloc(PyConformalPIDObject *self) {
    cs_conformal_destroy(self->cp);
    self->cp = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int PyConformalPID_init(PyConformalPIDObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {
        "alpha", "eta", "ki", "c_sat", "score_bound", "init_q",
        "edet_threshold", "warmup_cal", "score_window", "edet_max_prefix",
        "seed", "enable_ereset", NULL
    };
    cs_conformal_config_t cfg;
    cs_conformal_config_default(&cfg);

    long long warmup_cal = (long long)cfg.warmup_cal;
    long long score_window = (long long)cfg.score_window;
    long long edet_max_prefix = (long long)cfg.edet_max_prefix;
    unsigned long long seed = cfg.seed;
    int enable_ereset = cfg.enable_ereset ? 1 : 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "|dddddddLLLKp", kwlist,
            &cfg.alpha, &cfg.eta, &cfg.ki, &cfg.c_sat, &cfg.score_bound, &cfg.init_q,
            &cfg.edet_threshold, &warmup_cal, &score_window, &edet_max_prefix,
            &seed, &enable_ereset)) {
        return -1;
    }
    if (cs_check_u32(warmup_cal, "warmup_cal", &cfg.warmup_cal) != 0 ||
        cs_check_u32(score_window, "score_window", &cfg.score_window) != 0 ||
        cs_check_u32(edet_max_prefix, "edet_max_prefix", &cfg.edet_max_prefix) != 0) {
        return -1;
    }
    cfg.seed = (uint64_t)seed;
    cfg.enable_ereset = (enable_ereset != 0);

    cs_conformal_pid_t *new_cp = cs_conformal_create(&cfg);
    if (!new_cp) {
        PyErr_SetString(PyExc_ValueError, "invalid ConformalPID configuration");
        return -1;
    }
    cs_conformal_destroy(self->cp);
    self->cp = new_cp;
    return 0;
}

static PyObject *conformal_result_to_dict(const cs_conformal_result_t *r) {
    return Py_BuildValue(
        "{s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:K,s:K,s:O,s:O,s:O}",
        "q_used", r->q_used,
        "q_next", r->q_next,
        "lower", r->lower,
        "upper", r->upper,
        "score", r->score,
        "q_p", r->q_p,
        "q_i", r->q_i,
        "q_d", r->q_d,
        "emp_coverage", r->emp_coverage,
        "edet_stat", r->edet_stat,
        "step_count", (unsigned long long)r->step_count,
        "reset_count", (unsigned long long)r->reset_count,
        "miscovered", r->miscovered ? Py_True : Py_False,
        "shift_reset", r->shift_reset ? Py_True : Py_False,
        "calibrating", r->calibrating ? Py_True : Py_False);
}

static PyObject *PyConformalPID_step(PyConformalPIDObject *self, PyObject *args) {
    CS_CHECK_INIT(self->cp, "ConformalPID");
    double y_true = 0.0, y_pred = 0.0;
    if (!PyArg_ParseTuple(args, "dd", &y_true, &y_pred)) {
        return NULL;
    }
    cs_conformal_result_t res;
    cs_status_t st = cs_conformal_step(self->cp, y_true, y_pred, &res);
    if (st != CS_OK) {
        cs_set_py_error(st, "ConformalPID.step");
        return NULL;
    }
    return conformal_result_to_dict(&res);
}

static PyObject *PyConformalPID_ingest_batch(PyConformalPIDObject *self, PyObject *args, PyObject *kwds) {
    CS_CHECK_INIT(self->cp, "ConformalPID");
    static char *kwlist[] = {
        "y_true", "y_pred", "out_q_used", "out_q_next", "out_miscov", "out_resets", NULL
    };
    PyObject *yt_obj = NULL, *yp_obj = NULL;
    PyObject *qu_obj = Py_None, *qn_obj = Py_None, *mc_obj = Py_None, *rs_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OO|OOOO", kwlist,
            &yt_obj, &yp_obj, &qu_obj, &qn_obj, &mc_obj, &rs_obj)) {
        return NULL;
    }

    cs_py_f64_buf_t yt, yp, qu, qn;
    cs_py_u8_buf_t mc, rs;
    memset(&yp, 0, sizeof(yp));
    memset(&qu, 0, sizeof(qu));
    memset(&qn, 0, sizeof(qn));
    memset(&mc, 0, sizeof(mc));
    memset(&rs, 0, sizeof(rs));

    if (cs_py_acquire_f64_ro(yt_obj, &yt) != 0) return NULL;
    if (cs_py_acquire_f64_ro(yp_obj, &yp) != 0) {
        cs_py_f64_release(&yt);
        return NULL;
    }
    if (yt.len != yp.len) {
        cs_py_f64_release(&yp);
        cs_py_f64_release(&yt);
        PyErr_SetString(PyExc_ValueError, "y_true and y_pred must have equal length");
        return NULL;
    }
    if ((qu_obj != Py_None && cs_py_acquire_f64_rw(qu_obj, yt.len, &qu) != 0) ||
        (qn_obj != Py_None && cs_py_acquire_f64_rw(qn_obj, yt.len, &qn) != 0) ||
        (mc_obj != Py_None && cs_py_acquire_u8_rw(mc_obj, yt.len, &mc) != 0) ||
        (rs_obj != Py_None && cs_py_acquire_u8_rw(rs_obj, yt.len, &rs) != 0)) {
        cs_py_u8_release(&rs);
        cs_py_u8_release(&mc);
        cs_py_f64_release(&qn);
        cs_py_f64_release(&qu);
        cs_py_f64_release(&yp);
        cs_py_f64_release(&yt);
        return NULL;
    }

    cs_conformal_result_t last;
    memset(&last, 0, sizeof(last));
    cs_status_t st;
    Py_BEGIN_ALLOW_THREADS
    st = cs_conformal_step_batch(self->cp, yt.data, yp.data, yt.len, qu.data, qn.data, mc.data, rs.data, &last);
    Py_END_ALLOW_THREADS

    cs_py_u8_release(&rs);
    cs_py_u8_release(&mc);
    cs_py_f64_release(&qn);
    cs_py_f64_release(&qu);
    cs_py_f64_release(&yp);
    cs_py_f64_release(&yt);

    if (st != CS_OK) {
        cs_set_py_error(st, "ConformalPID.ingest_batch");
        return NULL;
    }
    return conformal_result_to_dict(&last);
}

static PyObject *PyConformalPID_reset(PyConformalPIDObject *self, PyObject *Py_UNUSED(ignored)) {
    CS_CHECK_INIT(self->cp, "ConformalPID");
    cs_conformal_reset(self->cp);
    Py_RETURN_NONE;
}

static PyObject *PyConformalPID_get_q(PyConformalPIDObject *self, void *Py_UNUSED(c)) {
    CS_CHECK_INIT(self->cp, "ConformalPID");
    return PyFloat_FromDouble(cs_conformal_current_q(self->cp));
}

static PyMethodDef PyConformalPID_methods[] = {
    {"step", (PyCFunction)PyConformalPID_step, METH_VARARGS, "Process one (y_true, y_pred) pair."},
    {"ingest_batch", (PyCFunction)(void (*)(void))PyConformalPID_ingest_batch,
     METH_VARARGS | METH_KEYWORDS, "Zero-copy batch Conformal PID update."},
    {"reset", (PyCFunction)PyConformalPID_reset, METH_NOARGS, "Reset Conformal PID state."},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef PyConformalPID_getset[] = {
    {"q", (getter)PyConformalPID_get_q, NULL, "Current prediction interval radius q_t.", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject PyConformalPIDType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "chronosieve._core.ConformalPID",
    .tp_basicsize = sizeof(PyConformalPIDObject),
    .tp_dealloc = (destructor)PyConformalPID_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Pre-Control E-Reset Conformal PID (Angelopoulos et al. NeurIPS 2023 + bounded D-term reset).",
    .tp_methods = PyConformalPID_methods,
    .tp_getset = PyConformalPID_getset,
    .tp_init = (initproc)PyConformalPID_init,
    .tp_new = PyType_GenericNew,
};

typedef struct {
    PyObject_HEAD
    cs_conformal_soa_t *soa;
    size_t n_channels;
} PyConformalSoAObject;

static void PyConformalSoA_dealloc(PyConformalSoAObject *self) {
    cs_conformal_soa_destroy(self->soa);
    self->soa = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int PyConformalSoA_init(PyConformalSoAObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {
        "n_channels", "alpha", "eta", "ki", "c_sat", "score_bound", "init_q", NULL
    };
    Py_ssize_t n_channels = 0;
    cs_conformal_config_t cfg;
    cs_conformal_config_default(&cfg);
    cfg.enable_ereset = false;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "n|dddddd", kwlist,
            &n_channels, &cfg.alpha, &cfg.eta, &cfg.ki, &cfg.c_sat, &cfg.score_bound, &cfg.init_q)) {
        return -1;
    }
    if (n_channels <= 0) {
        PyErr_SetString(PyExc_ValueError, "n_channels must be > 0");
        return -1;
    }
    cs_conformal_soa_t *new_soa = cs_conformal_soa_create((size_t)n_channels, &cfg);
    if (!new_soa) {
        PyErr_SetString(PyExc_ValueError, "invalid ConformalBatchPID configuration");
        return -1;
    }
    cs_conformal_soa_destroy(self->soa);
    self->soa = new_soa;
    self->n_channels = (size_t)n_channels;
    return 0;
}

static PyObject *PyConformalSoA_ingest_batch(PyConformalSoAObject *self, PyObject *args, PyObject *kwds) {
    CS_CHECK_INIT(self->soa, "ConformalBatchPID");
    static char *kwlist[] = {"y_true", "y_pred", "out_q_next", "out_miscov", NULL};
    PyObject *yt_obj = NULL, *yp_obj = NULL, *qn_obj = Py_None, *mc_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OO", kwlist, &yt_obj, &yp_obj, &qn_obj, &mc_obj)) {
        return NULL;
    }

    cs_py_f64_buf_t yt, yp, qn;
    cs_py_u8_buf_t mc;
    memset(&yp, 0, sizeof(yp));
    memset(&qn, 0, sizeof(qn));
    memset(&mc, 0, sizeof(mc));

    if (cs_py_acquire_f64_ro(yt_obj, &yt) != 0) return NULL;
    if (cs_py_acquire_f64_ro(yp_obj, &yp) != 0) {
        cs_py_f64_release(&yt);
        return NULL;
    }
    if (yt.len != yp.len || yt.len % self->n_channels != 0U) {
        cs_py_f64_release(&yp);
        cs_py_f64_release(&yt);
        PyErr_SetString(PyExc_ValueError, "y_true and y_pred must have equal size divisible by n_channels");
        return NULL;
    }
    if ((qn_obj != Py_None && cs_py_acquire_f64_rw(qn_obj, yt.len, &qn) != 0) ||
        (mc_obj != Py_None && cs_py_acquire_u8_rw(mc_obj, yt.len, &mc) != 0)) {
        cs_py_u8_release(&mc);
        cs_py_f64_release(&qn);
        cs_py_f64_release(&yp);
        cs_py_f64_release(&yt);
        return NULL;
    }

    size_t n_steps = yt.len / self->n_channels;
    cs_status_t st;
    Py_BEGIN_ALLOW_THREADS
    st = cs_conformal_soa_step_batch(self->soa, yt.data, yp.data, n_steps, qn.data, mc.data);
    Py_END_ALLOW_THREADS

    cs_py_u8_release(&mc);
    cs_py_f64_release(&qn);
    cs_py_f64_release(&yp);
    cs_py_f64_release(&yt);

    if (st != CS_OK) {
        cs_set_py_error(st, "ConformalBatchPID.ingest_batch");
        return NULL;
    }
    return PyLong_FromSize_t(n_steps);
}

static PyObject *PyConformalSoA_reset(PyConformalSoAObject *self, PyObject *Py_UNUSED(ignored)) {
    CS_CHECK_INIT(self->soa, "ConformalBatchPID");
    cs_conformal_soa_reset(self->soa);
    Py_RETURN_NONE;
}

static PyMethodDef PyConformalSoA_methods[] = {
    {"ingest_batch", (PyCFunction)(void (*)(void))PyConformalSoA_ingest_batch,
     METH_VARARGS | METH_KEYWORDS, "Zero-copy SoA SIMD batch update across C channels."},
    {"reset", (PyCFunction)PyConformalSoA_reset, METH_NOARGS, "Reset all C channels."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject PyConformalSoAType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "chronosieve._core.ConformalBatchPID",
    .tp_basicsize = sizeof(PyConformalSoAObject),
    .tp_dealloc = (destructor)PyConformalSoA_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Multi-channel SoA SIMD Conformal PID batch kernel.",
    .tp_methods = PyConformalSoA_methods,
    .tp_init = (initproc)PyConformalSoA_init,
    .tp_new = PyType_GenericNew,
};

/* =========================================================================
 * 4. SleepingBOA Python Type
 * ========================================================================= */

typedef struct {
    PyObject_HEAD
    cs_fuser_t *fus;
    size_t n_agents;
} PySleepingBOAObject;

static void PySleepingBOA_dealloc(PySleepingBOAObject *self) {
    cs_fuser_destroy(self->fus);
    self->fus = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int PySleepingBOA_init(PySleepingBOAObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {
        "n_agents", "share_base", "share_shift", "shift_var_decay",
        "rogue_threshold", "rogue_mu0", "rogue_sigma", "cal_window",
        "use_abs_loss", "quarantine_rogue", NULL
    };
    Py_ssize_t n_agents = 0;
    double share_base = 0.01;
    double share_shift = 0.35;
    double shift_var_decay = 0.25;
    double rogue_threshold = 200.0;
    double rogue_mu0 = NAN;
    double rogue_sigma = NAN;
    long long cal_window_ll = 48LL;
    int use_abs_loss = 0;
    int quarantine_rogue = 1;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "n|ddddddLpp", kwlist,
            &n_agents, &share_base, &share_shift, &shift_var_decay,
            &rogue_threshold, &rogue_mu0, &rogue_sigma, &cal_window_ll,
            &use_abs_loss, &quarantine_rogue)) {
        return -1;
    }
    if (n_agents <= 0) {
        PyErr_SetString(PyExc_ValueError, "n_agents must be > 0");
        return -1;
    }

    cs_fuser_config_t cfg;
    cs_fuser_config_default(&cfg, (size_t)n_agents);
    if (cs_check_u32(cal_window_ll, "cal_window", &cfg.cal_window) != 0) {
        return -1;
    }
    cfg.share_base = share_base;
    cfg.share_shift = share_shift;
    cfg.shift_var_decay = shift_var_decay;
    cfg.rogue_threshold = rogue_threshold;
    cfg.rogue_mu0 = rogue_mu0;
    cfg.rogue_sigma = rogue_sigma;
    cfg.use_abs_loss = (use_abs_loss != 0);
    cfg.quarantine_rogue = (quarantine_rogue != 0);

    cs_fuser_t *new_fus = cs_fuser_create(&cfg);
    if (!new_fus) {
        PyErr_SetString(PyExc_ValueError, "invalid SleepingBOA configuration");
        return -1;
    }
    cs_fuser_destroy(self->fus);
    self->fus = new_fus;
    self->n_agents = (size_t)n_agents;
    return 0;
}

static PyObject *fuser_result_to_dict(const cs_fuser_result_t *r) {
    return Py_BuildValue(
        "{s:d,s:d,s:d,s:i,s:n,s:n,s:K,s:O}",
        "prediction", r->prediction,
        "fused_loss", r->fused_loss,
        "max_rogue_stat", r->max_rogue_stat,
        "worst_agent", (int)r->worst_agent,
        "active_count", (Py_ssize_t)r->active_count,
        "rogue_count", (Py_ssize_t)r->rogue_count,
        "step_count", (unsigned long long)r->step_count,
        "any_rogue", r->any_rogue ? Py_True : Py_False);
}

static PyObject *PySleepingBOA_predict(PySleepingBOAObject *self, PyObject *args, PyObject *kwds) {
    CS_CHECK_INIT(self->fus, "SleepingBOA");
    static char *kwlist[] = {"preds", "active", NULL};
    PyObject *p_obj = NULL, *a_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &p_obj, &a_obj)) {
        return NULL;
    }

    cs_py_f64_buf_t pbuf;
    cs_py_u8_buf_t abuf;
    memset(&abuf, 0, sizeof(abuf));

    if (cs_py_acquire_f64_ro(p_obj, &pbuf) != 0) return NULL;
    if (pbuf.len != self->n_agents) {
        cs_py_f64_release(&pbuf);
        PyErr_SetString(PyExc_ValueError, "preds length must equal n_agents");
        return NULL;
    }
    if (a_obj != Py_None) {
        if (cs_py_acquire_u8_ro(a_obj, &abuf) != 0) {
            cs_py_f64_release(&pbuf);
            return NULL;
        }
        if (abuf.len != self->n_agents) {
            cs_py_u8_release(&abuf);
            cs_py_f64_release(&pbuf);
            PyErr_SetString(PyExc_ValueError, "active length must equal n_agents");
            return NULL;
        }
    }

    double pred = 0.0;
    cs_status_t st = cs_fuser_predict(self->fus, pbuf.data, a_obj != Py_None ? abuf.data : NULL, &pred);
    cs_py_u8_release(&abuf);
    cs_py_f64_release(&pbuf);
    if (st != CS_OK) {
        cs_set_py_error(st, "SleepingBOA.predict");
        return NULL;
    }
    return PyFloat_FromDouble(pred);
}

static PyObject *PySleepingBOA_step(PySleepingBOAObject *self, PyObject *args, PyObject *kwds) {
    CS_CHECK_INIT(self->fus, "SleepingBOA");
    static char *kwlist[] = {"preds", "y_true", "active", "shift_detected", NULL};
    PyObject *p_obj = NULL, *a_obj = Py_None;
    double y_true = 0.0;
    int shift_detected = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Od|Op", kwlist, &p_obj, &y_true, &a_obj, &shift_detected)) {
        return NULL;
    }

    cs_py_f64_buf_t pbuf;
    cs_py_u8_buf_t abuf;
    memset(&abuf, 0, sizeof(abuf));

    if (cs_py_acquire_f64_ro(p_obj, &pbuf) != 0) return NULL;
    if (pbuf.len != self->n_agents) {
        cs_py_f64_release(&pbuf);
        PyErr_SetString(PyExc_ValueError, "preds length must equal n_agents");
        return NULL;
    }
    if (a_obj != Py_None) {
        if (cs_py_acquire_u8_ro(a_obj, &abuf) != 0) {
            cs_py_f64_release(&pbuf);
            return NULL;
        }
        if (abuf.len != self->n_agents) {
            cs_py_u8_release(&abuf);
            cs_py_f64_release(&pbuf);
            PyErr_SetString(PyExc_ValueError, "active length must equal n_agents");
            return NULL;
        }
    }

    cs_fuser_result_t res;
    cs_status_t st = cs_fuser_step(
        self->fus, pbuf.data, a_obj != Py_None ? abuf.data : NULL,
        y_true, shift_detected != 0, &res);
    cs_py_u8_release(&abuf);
    cs_py_f64_release(&pbuf);
    if (st != CS_OK) {
        cs_set_py_error(st, "SleepingBOA.step");
        return NULL;
    }
    return fuser_result_to_dict(&res);
}

static PyObject *PySleepingBOA_fuse_batch(PySleepingBOAObject *self, PyObject *args, PyObject *kwds) {
    CS_CHECK_INIT(self->fus, "SleepingBOA");
    static char *kwlist[] = {
        "preds_matrix", "y_true", "active_matrix", "shifts",
        "out_preds", "out_rogue_any", NULL
    };
    PyObject *pm_obj = NULL, *yt_obj = NULL;
    PyObject *am_obj = Py_None, *sh_obj = Py_None, *op_obj = Py_None, *or_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OO|OOOO", kwlist,
            &pm_obj, &yt_obj, &am_obj, &sh_obj, &op_obj, &or_obj)) {
        return NULL;
    }

    cs_py_f64_buf_t pm, yt, op;
    cs_py_u8_buf_t am, sh, rog;
    memset(&yt, 0, sizeof(yt));
    memset(&op, 0, sizeof(op));
    memset(&am, 0, sizeof(am));
    memset(&sh, 0, sizeof(sh));
    memset(&rog, 0, sizeof(rog));

    if (cs_py_acquire_f64_ro(pm_obj, &pm) != 0) return NULL;
    if (cs_py_acquire_f64_ro(yt_obj, &yt) != 0) {
        cs_py_f64_release(&pm);
        return NULL;
    }
    if (pm.len % self->n_agents != 0U || pm.len / self->n_agents != yt.len) {
        cs_py_f64_release(&yt);
        cs_py_f64_release(&pm);
        PyErr_SetString(PyExc_ValueError, "preds_matrix must have len(y_true) * n_agents elements");
        return NULL;
    }
    if (am_obj != Py_None) {
        if (cs_py_acquire_u8_ro(am_obj, &am) != 0 || am.len != pm.len) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_ValueError, "active_matrix size mismatch");
            }
            goto fail;
        }
    }
    if (sh_obj != Py_None) {
        if (cs_py_acquire_u8_ro(sh_obj, &sh) != 0 || sh.len != yt.len) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_ValueError, "shifts size mismatch");
            }
            goto fail;
        }
    }
    if (op_obj != Py_None && cs_py_acquire_f64_rw(op_obj, yt.len, &op) != 0) goto fail;
    if (or_obj != Py_None && cs_py_acquire_u8_rw(or_obj, yt.len, &rog) != 0) goto fail;

    cs_fuser_result_t last;
    memset(&last, 0, sizeof(last));
    cs_status_t st;
    Py_BEGIN_ALLOW_THREADS
    st = cs_fuser_step_batch(
        self->fus, pm.data, am.data, yt.data, sh.data, yt.len,
        op.data, NULL, rog.data, &last);
    Py_END_ALLOW_THREADS

    cs_py_u8_release(&rog);
    cs_py_f64_release(&op);
    cs_py_u8_release(&sh);
    cs_py_u8_release(&am);
    cs_py_f64_release(&yt);
    cs_py_f64_release(&pm);

    if (st != CS_OK) {
        cs_set_py_error(st, "SleepingBOA.fuse_batch");
        return NULL;
    }
    return fuser_result_to_dict(&last);

fail:
    cs_py_u8_release(&rog);
    cs_py_f64_release(&op);
    cs_py_u8_release(&sh);
    cs_py_u8_release(&am);
    cs_py_f64_release(&yt);
    cs_py_f64_release(&pm);
    return NULL;
}

static PyObject *PySleepingBOA_weights(PySleepingBOAObject *self, PyObject *Py_UNUSED(ignored)) {
    CS_CHECK_INIT(self->fus, "SleepingBOA");
    double *w = (double *)malloc(self->n_agents * sizeof(double));
    if (!w) return PyErr_NoMemory();
    cs_fuser_get_weights(self->fus, w, self->n_agents);
    PyObject *lst = PyList_New((Py_ssize_t)self->n_agents);
    if (!lst) {
        free(w);
        return NULL;
    }
    for (size_t i = 0; i < self->n_agents; ++i) {
        PyObject *v = PyFloat_FromDouble(w[i]);
        if (!v) {
            Py_DECREF(lst);
            free(w);
            return NULL;
        }
        PyList_SET_ITEM(lst, (Py_ssize_t)i, v);
    }
    free(w);
    return lst;
}

static PyObject *PySleepingBOA_rogue_stats(PySleepingBOAObject *self, PyObject *Py_UNUSED(ignored)) {
    CS_CHECK_INIT(self->fus, "SleepingBOA");
    double *stats = (double *)malloc(self->n_agents * sizeof(double));
    uint8_t *flags = (uint8_t *)malloc(self->n_agents);
    if (!stats || !flags) {
        free(stats);
        free(flags);
        return PyErr_NoMemory();
    }
    cs_fuser_get_rogue_stats(self->fus, stats, flags, self->n_agents);
    PyObject *slst = PyList_New((Py_ssize_t)self->n_agents);
    PyObject *flst = PyList_New((Py_ssize_t)self->n_agents);
    if (!slst || !flst) {
        Py_XDECREF(slst);
        Py_XDECREF(flst);
        free(stats);
        free(flags);
        return NULL;
    }
    for (size_t i = 0; i < self->n_agents; ++i) {
        PyObject *sv = PyFloat_FromDouble(stats[i]);
        if (!sv) {
            Py_DECREF(slst);
            Py_DECREF(flst);
            free(stats);
            free(flags);
            return NULL;
        }
        PyList_SET_ITEM(slst, (Py_ssize_t)i, sv);
        PyList_SET_ITEM(flst, (Py_ssize_t)i, PyBool_FromLong(flags[i] != 0U));
    }
    free(stats);
    free(flags);
    PyObject *tup = PyTuple_New(2);
    if (!tup) {
        Py_DECREF(slst);
        Py_DECREF(flst);
        return NULL;
    }
    PyTuple_SET_ITEM(tup, 0, slst);
    PyTuple_SET_ITEM(tup, 1, flst);
    return tup;
}

static PyObject *PySleepingBOA_clear_rogue(PySleepingBOAObject *self, PyObject *args) {
    CS_CHECK_INIT(self->fus, "SleepingBOA");
    int agent_idx = -1;
    if (!PyArg_ParseTuple(args, "|i", &agent_idx)) {
        return NULL;
    }
    cs_fuser_clear_rogue(self->fus, (int32_t)agent_idx);
    Py_RETURN_NONE;
}

static PyObject *PySleepingBOA_reset(PySleepingBOAObject *self, PyObject *Py_UNUSED(ignored)) {
    CS_CHECK_INIT(self->fus, "SleepingBOA");
    cs_fuser_reset(self->fus);
    Py_RETURN_NONE;
}

static PyMethodDef PySleepingBOA_methods[] = {
    {"predict", (PyCFunction)(void (*)(void))PySleepingBOA_predict, METH_VARARGS | METH_KEYWORDS,
     "Compute fused prediction from active agents without updating weights."},
    {"step", (PyCFunction)(void (*)(void))PySleepingBOA_step, METH_VARARGS | METH_KEYWORDS,
     "Fuse active agent predictions, update Sleeping Fixed-Share BOA weights, and check rogue drift."},
    {"fuse_batch", (PyCFunction)(void (*)(void))PySleepingBOA_fuse_batch,
     METH_VARARGS | METH_KEYWORDS, "Zero-copy batch multi-agent fusion."},
    {"weights", (PyCFunction)PySleepingBOA_weights, METH_NOARGS, "Return current agent weights."},
    {"rogue_stats", (PyCFunction)PySleepingBOA_rogue_stats, METH_NOARGS,
     "Return (catoni_stats, rogue_flags) per agent."},
    {"clear_rogue", (PyCFunction)PySleepingBOA_clear_rogue, METH_VARARGS,
     "Clear rogue flag for agent_idx (or -1 for all agents)."},
    {"reset", (PyCFunction)PySleepingBOA_reset, METH_NOARGS, "Reset weights, variance, and drift detectors."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject PySleepingBOAType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "chronosieve._core.SleepingBOA",
    .tp_basicsize = sizeof(PySleepingBOAObject),
    .tp_dealloc = (destructor)PySleepingBOA_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "SIMD Sleeping Fixed-Share BOA with per-agent residual Catoni drift attribution.",
    .tp_methods = PySleepingBOA_methods,
    .tp_init = (initproc)PySleepingBOA_init,
    .tp_new = PyType_GenericNew,
};

/* =========================================================================
 * 5. StreamDigest Python Type (Swing PLA + Wake-Up Gate + Ring-Buffer zoom)
 * ========================================================================= */

typedef struct {
    PyObject_HEAD
    cs_digest_t *dig;
} PyStreamDigestObject;

static void PyStreamDigest_dealloc(PyStreamDigestObject *self) {
    cs_digest_destroy(self->dig);
    self->dig = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int PyStreamDigest_init(PyStreamDigestObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {
        "epsilon", "ring_capacity", "max_segments", "max_anomalies",
        "wake_cooldown", "anomaly_burst", NULL
    };
    cs_digest_config_t cfg;
    cs_digest_config_default(&cfg);

    Py_ssize_t ring_cap = (Py_ssize_t)cfg.ring_capacity;
    Py_ssize_t max_segs = (Py_ssize_t)cfg.max_segments;
    Py_ssize_t max_anoms = (Py_ssize_t)cfg.max_anomalies;
    long long wake_cd = (long long)cfg.wake_cooldown;
    long long anom_burst = (long long)cfg.anomaly_burst;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "|dnnnLL", kwlist,
            &cfg.epsilon, &ring_cap, &max_segs, &max_anoms, &wake_cd, &anom_burst)) {
        return -1;
    }
    if (ring_cap <= 0 || max_segs <= 0 || max_anoms <= 0) {
        PyErr_SetString(PyExc_ValueError, "capacities must be > 0");
        return -1;
    }
    if (cs_check_u32(wake_cd, "wake_cooldown", &cfg.wake_cooldown) != 0 ||
        cs_check_u32(anom_burst, "anomaly_burst", &cfg.anomaly_burst) != 0) {
        return -1;
    }
    cfg.ring_capacity = (size_t)ring_cap;
    cfg.max_segments = (size_t)max_segs;
    cfg.max_anomalies = (size_t)max_anoms;

    cs_digest_t *new_dig = cs_digest_create(&cfg);
    if (!new_dig) {
        PyErr_SetString(PyExc_ValueError, "invalid StreamDigest configuration");
        return -1;
    }
    cs_digest_destroy(self->dig);
    self->dig = new_dig;
    return 0;
}

static PyObject *digest_step_to_dict(const cs_digest_step_t *s) {
    return Py_BuildValue(
        "{s:K,s:K,s:K,s:K,s:O,s:O}",
        "total_points", (unsigned long long)s->total_points,
        "total_segments", (unsigned long long)s->total_segments,
        "total_anomalies", (unsigned long long)s->total_anomalies,
        "wake_count", (unsigned long long)s->wake_count,
        "should_wake", s->should_wake ? Py_True : Py_False,
        "segment_emitted", s->segment_emitted ? Py_True : Py_False);
}

static PyObject *PyStreamDigest_push(PyStreamDigestObject *self, PyObject *args, PyObject *kwds) {
    CS_CHECK_INIT(self->dig, "StreamDigest");
    static char *kwlist[] = {"t", "y", "y_hat", "q_t", "edet_stat", "flags", NULL};
    double t = 0.0, y = 0.0;
    PyObject *yh_obj = Py_None;
    double q_t = NAN, edet_stat = 0.0;
    long long flags_ll = 0LL;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "dd|OddL", kwlist, &t, &y, &yh_obj, &q_t, &edet_stat, &flags_ll)) {
        return NULL;
    }
    uint32_t flags = 0U;
    if (cs_check_u32(flags_ll, "flags", &flags) != 0) {
        return NULL;
    }
    double y_hat = y;
    if (yh_obj != Py_None) {
        y_hat = PyFloat_AsDouble(yh_obj);
        if (y_hat == -1.0 && PyErr_Occurred()) {
            return NULL;
        }
    }

    cs_digest_step_t step;
    cs_status_t st = cs_digest_push(self->dig, t, y, y_hat, q_t, edet_stat, flags, &step);
    if (st != CS_OK) {
        cs_set_py_error(st, "StreamDigest.push");
        return NULL;
    }
    return digest_step_to_dict(&step);
}

static PyObject *PyStreamDigest_digest_batch(PyStreamDigestObject *self, PyObject *args, PyObject *kwds) {
    CS_CHECK_INIT(self->dig, "StreamDigest");
    static char *kwlist[] = {
        "ts", "ys", "y_hats", "qs", "edet_stats", "flags", "out_wakes", NULL
    };
    PyObject *ts_obj = NULL, *ys_obj = NULL;
    PyObject *yh_obj = Py_None, *qs_obj = Py_None, *es_obj = Py_None;
    PyObject *fl_obj = Py_None, *ow_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OO|OOOOO", kwlist,
            &ts_obj, &ys_obj, &yh_obj, &qs_obj, &es_obj, &fl_obj, &ow_obj)) {
        return NULL;
    }

    cs_py_f64_buf_t ts, ys, yh, qs, es;
    cs_py_u32_buf_t fl;
    cs_py_u8_buf_t ow;
    memset(&ys, 0, sizeof(ys));
    memset(&yh, 0, sizeof(yh));
    memset(&qs, 0, sizeof(qs));
    memset(&es, 0, sizeof(es));
    memset(&fl, 0, sizeof(fl));
    memset(&ow, 0, sizeof(ow));

    if (cs_py_acquire_f64_ro(ts_obj, &ts) != 0) return NULL;
    if (cs_py_acquire_f64_ro(ys_obj, &ys) != 0) {
        cs_py_f64_release(&ts);
        return NULL;
    }
    if (ts.len != ys.len) {
        PyErr_SetString(PyExc_ValueError, "ts and ys must have equal length");
        goto fail;
    }
    if (yh_obj != Py_None && (cs_py_acquire_f64_ro(yh_obj, &yh) != 0 || yh.len != ts.len)) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "y_hats length mismatch");
        goto fail;
    }
    if (qs_obj != Py_None && (cs_py_acquire_f64_ro(qs_obj, &qs) != 0 || qs.len != ts.len)) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "qs length mismatch");
        goto fail;
    }
    if (es_obj != Py_None && (cs_py_acquire_f64_ro(es_obj, &es) != 0 || es.len != ts.len)) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "edet_stats length mismatch");
        goto fail;
    }
    if (fl_obj != Py_None && (cs_py_acquire_u32_ro(fl_obj, &fl) != 0 || fl.len != ts.len)) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "flags length mismatch");
        goto fail;
    }
    if (ow_obj != Py_None && cs_py_acquire_u8_rw(ow_obj, ts.len, &ow) != 0) goto fail;

    cs_digest_step_t last;
    memset(&last, 0, sizeof(last));
    cs_status_t st;
    Py_BEGIN_ALLOW_THREADS
    st = cs_digest_push_batch(self->dig, ts.data, ys.data, yh.data, qs.data, es.data, fl.data, ts.len, ow.data, &last);
    Py_END_ALLOW_THREADS

    cs_py_u8_release(&ow);
    cs_py_u32_release(&fl);
    cs_py_f64_release(&es);
    cs_py_f64_release(&qs);
    cs_py_f64_release(&yh);
    cs_py_f64_release(&ys);
    cs_py_f64_release(&ts);

    if (st != CS_OK) {
        cs_set_py_error(st, "StreamDigest.digest_batch");
        return NULL;
    }
    return digest_step_to_dict(&last);

fail:
    cs_py_u8_release(&ow);
    cs_py_u32_release(&fl);
    cs_py_f64_release(&es);
    cs_py_f64_release(&qs);
    cs_py_f64_release(&yh);
    cs_py_f64_release(&ys);
    cs_py_f64_release(&ts);
    return NULL;
}

static PyObject *PyStreamDigest_flush(PyStreamDigestObject *self, PyObject *Py_UNUSED(ignored)) {
    CS_CHECK_INIT(self->dig, "StreamDigest");
    cs_digest_flush(self->dig);
    Py_RETURN_NONE;
}

static PyObject *PyStreamDigest_segments(PyStreamDigestObject *self, PyObject *args) {
    CS_CHECK_INIT(self->dig, "StreamDigest");
    Py_ssize_t max_count = 0;
    if (!PyArg_ParseTuple(args, "|n", &max_count)) {
        return NULL;
    }
    size_t total = 0U;
    cs_digest_get_segments(self->dig, NULL, 0U, &total);
    size_t want = (max_count > 0 && (size_t)max_count < total) ? (size_t)max_count : total;
    if (want == 0U) {
        return PyList_New(0);
    }
    cs_segment_t *arr = (cs_segment_t *)malloc(want * sizeof(*arr));
    if (!arr) return PyErr_NoMemory();
    size_t got = 0U;
    cs_digest_get_segments(self->dig, arr, want, &got);

    PyObject *lst = PyList_New((Py_ssize_t)got);
    if (!lst) {
        free(arr);
        return NULL;
    }
    for (size_t i = 0; i < got; ++i) {
        PyObject *d = Py_BuildValue(
            "{s:d,s:d,s:d,s:d,s:d,s:k,s:k}",
            "t_start", arr[i].t_start,
            "y_start", arr[i].y_start,
            "t_end", arr[i].t_end,
            "y_end", arr[i].y_end,
            "slope", arr[i].slope,
            "count", (unsigned long)arr[i].count,
            "flags", (unsigned long)arr[i].flags);
        if (!d) {
            Py_DECREF(lst);
            free(arr);
            return NULL;
        }
        PyList_SET_ITEM(lst, (Py_ssize_t)i, d);
    }
    free(arr);
    return lst;
}

static PyObject *PyStreamDigest_anomalies(PyStreamDigestObject *self, PyObject *args) {
    CS_CHECK_INIT(self->dig, "StreamDigest");
    Py_ssize_t max_count = 0;
    if (!PyArg_ParseTuple(args, "|n", &max_count)) {
        return NULL;
    }
    size_t total = 0U;
    cs_digest_get_anomalies(self->dig, NULL, 0U, &total);
    size_t want = (max_count > 0 && (size_t)max_count < total) ? (size_t)max_count : total;
    if (want == 0U) {
        return PyList_New(0);
    }
    cs_anomaly_t *arr = (cs_anomaly_t *)malloc(want * sizeof(*arr));
    if (!arr) return PyErr_NoMemory();
    size_t got = 0U;
    cs_digest_get_anomalies(self->dig, arr, want, &got);

    PyObject *lst = PyList_New((Py_ssize_t)got);
    if (!lst) {
        free(arr);
        return NULL;
    }
    for (size_t i = 0; i < got; ++i) {
        PyObject *d = Py_BuildValue(
            "{s:d,s:d,s:d,s:d,s:d,s:d,s:k}",
            "t", arr[i].t,
            "y", arr[i].y,
            "y_hat", arr[i].y_hat,
            "residual", arr[i].residual,
            "q_t", arr[i].q_t,
            "edet_stat", arr[i].edet_stat,
            "flags", (unsigned long)arr[i].flags);
        if (!d) {
            Py_DECREF(lst);
            free(arr);
            return NULL;
        }
        PyList_SET_ITEM(lst, (Py_ssize_t)i, d);
    }
    free(arr);
    return lst;
}

static PyObject *PyStreamDigest_zoom(PyStreamDigestObject *self, PyObject *args) {
    CS_CHECK_INIT(self->dig, "StreamDigest");
    double t0 = 0.0, t1 = 0.0;
    if (!PyArg_ParseTuple(args, "dd", &t0, &t1)) {
        return NULL;
    }
    size_t n_rec = 0U;
    bool truncated = false;
    cs_status_t st = cs_digest_zoom(self->dig, t0, t1, NULL, 0U, &n_rec, &truncated);
    if (st != CS_OK) {
        cs_set_py_error(st, "StreamDigest.zoom");
        return NULL;
    }
    PyObject *lst = PyList_New((Py_ssize_t)n_rec);
    if (!lst) return NULL;

    if (n_rec > 0U) {
        cs_ring_record_t *recs = (cs_ring_record_t *)malloc(n_rec * sizeof(*recs));
        if (!recs) {
            Py_DECREF(lst);
            return PyErr_NoMemory();
        }
        cs_digest_zoom(self->dig, t0, t1, recs, n_rec, &n_rec, &truncated);
        for (size_t i = 0; i < n_rec; ++i) {
            PyObject *d = Py_BuildValue(
                "{s:d,s:d,s:d,s:d,s:k}",
                "t", recs[i].t,
                "y", recs[i].y,
                "y_hat", recs[i].y_hat,
                "q_t", recs[i].q_t,
                "flags", (unsigned long)recs[i].flags);
            if (!d) {
                free(recs);
                Py_DECREF(lst);
                return NULL;
            }
            PyList_SET_ITEM(lst, (Py_ssize_t)i, d);
        }
        free(recs);
    }

    PyObject *res = Py_BuildValue("{s:O,s:O}", "records", lst, "truncated", truncated ? Py_True : Py_False);
    Py_DECREF(lst);
    return res;
}

static PyObject *PyStreamDigest_reconstruct(PyStreamDigestObject *self, PyObject *args, PyObject *kwds) {
    CS_CHECK_INIT(self->dig, "StreamDigest");
    static char *kwlist[] = {"ts", "out_y", NULL};
    PyObject *ts_obj = NULL, *oy_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &ts_obj, &oy_obj)) {
        return NULL;
    }
    cs_py_f64_buf_t ts, oy;
    memset(&oy, 0, sizeof(oy));
    if (cs_py_acquire_f64_ro(ts_obj, &ts) != 0) return NULL;

    if (oy_obj != Py_None) {
        if (cs_py_acquire_f64_rw(oy_obj, ts.len, &oy) != 0) {
            cs_py_f64_release(&ts);
            return NULL;
        }
        cs_status_t st;
        Py_BEGIN_ALLOW_THREADS
        st = cs_digest_reconstruct(self->dig, ts.data, ts.len, oy.data);
        Py_END_ALLOW_THREADS
        cs_py_f64_release(&oy);
        cs_py_f64_release(&ts);
        if (st != CS_OK) {
            cs_set_py_error(st, "StreamDigest.reconstruct");
            return NULL;
        }
        Py_RETURN_NONE;
    }

    if (ts.len == 0U) {
        cs_py_f64_release(&ts);
        return PyList_New(0);
    }
    double *out = (double *)malloc(ts.len * sizeof(double));
    if (!out) {
        cs_py_f64_release(&ts);
        return PyErr_NoMemory();
    }
    cs_status_t st = cs_digest_reconstruct(self->dig, ts.data, ts.len, out);
    size_t n = ts.len;
    cs_py_f64_release(&ts);
    if (st != CS_OK) {
        free(out);
        cs_set_py_error(st, "StreamDigest.reconstruct");
        return NULL;
    }
    PyObject *lst = PyList_New((Py_ssize_t)n);
    if (!lst) {
        free(out);
        return NULL;
    }
    for (size_t i = 0; i < n; ++i) {
        PyObject *v = PyFloat_FromDouble(out[i]);
        if (!v) {
            Py_DECREF(lst);
            free(out);
            return NULL;
        }
        PyList_SET_ITEM(lst, (Py_ssize_t)i, v);
    }
    free(out);
    return lst;
}

static PyObject *PyStreamDigest_format_markdown(PyStreamDigestObject *self, PyObject *args, PyObject *kwds) {
    CS_CHECK_INIT(self->dig, "StreamDigest");
    static char *kwlist[] = {
        "channel", "emp_coverage", "current_q", "edet_stat", "edet_threshold",
        "rogue_agent", "max_segments", "max_anomalies", NULL
    };
    const char *channel = "stream";
    double emp_cov = 1.0, cur_q = 0.0, edet_stat = 0.0, edet_thr = 500.0;
    int rogue_agent = -1;
    Py_ssize_t max_segs = 6, max_anoms = 5;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "|sddddinn", kwlist,
            &channel, &emp_cov, &cur_q, &edet_stat, &edet_thr,
            &rogue_agent, &max_segs, &max_anoms)) {
        return NULL;
    }
    if (max_segs < 0) max_segs = 6;
    if (max_anoms < 0) max_anoms = 5;

    size_t req = 0U;
    cs_status_t st = cs_digest_format_markdown(
        self->dig, channel, emp_cov, cur_q, edet_stat, edet_thr,
        (int32_t)rogue_agent, (size_t)max_segs, (size_t)max_anoms,
        NULL, 0U, &req);
    if (st != CS_OK) {
        cs_set_py_error(st, "StreamDigest.format_markdown");
        return NULL;
    }

    char *buf = (char *)malloc(req + 2U);
    if (!buf) return PyErr_NoMemory();
    size_t actual = 0U;
    st = cs_digest_format_markdown(
        self->dig, channel, emp_cov, cur_q, edet_stat, edet_thr,
        (int32_t)rogue_agent, (size_t)max_segs, (size_t)max_anoms,
        buf, req + 2U, &actual);
    if (st != CS_OK) {
        free(buf);
        cs_set_py_error(st, "StreamDigest.format_markdown");
        return NULL;
    }

    PyObject *s = PyUnicode_FromStringAndSize(buf, (Py_ssize_t)actual);
    free(buf);
    return s;
}

static PyObject *PyStreamDigest_reset(PyStreamDigestObject *self, PyObject *Py_UNUSED(ignored)) {
    CS_CHECK_INIT(self->dig, "StreamDigest");
    cs_digest_reset(self->dig);
    Py_RETURN_NONE;
}

static PyObject *PyStreamDigest_get_eps(PyStreamDigestObject *self, void *Py_UNUSED(c)) {
    CS_CHECK_INIT(self->dig, "StreamDigest");
    return PyFloat_FromDouble(cs_digest_epsilon(self->dig));
}
static PyObject *PyStreamDigest_get_seg_cnt(PyStreamDigestObject *self, void *Py_UNUSED(c)) {
    CS_CHECK_INIT(self->dig, "StreamDigest");
    return PyLong_FromSize_t(cs_digest_segment_count(self->dig));
}
static PyObject *PyStreamDigest_get_anom_cnt(PyStreamDigestObject *self, void *Py_UNUSED(c)) {
    CS_CHECK_INIT(self->dig, "StreamDigest");
    return PyLong_FromSize_t(cs_digest_anomaly_count(self->dig));
}
static PyObject *PyStreamDigest_get_ring_sz(PyStreamDigestObject *self, void *Py_UNUSED(c)) {
    CS_CHECK_INIT(self->dig, "StreamDigest");
    return PyLong_FromSize_t(cs_digest_ring_size(self->dig));
}

static PyMethodDef PyStreamDigest_methods[] = {
    {"push", (PyCFunction)(void (*)(void))PyStreamDigest_push, METH_VARARGS | METH_KEYWORDS,
     "Push a single tick into Swing PLA, ring buffer, and wake-up gate."},
    {"digest_batch", (PyCFunction)(void (*)(void))PyStreamDigest_digest_batch,
     METH_VARARGS | METH_KEYWORDS, "Zero-copy batch Swing PLA + ring buffer + wake-up gate."},
    {"flush", (PyCFunction)PyStreamDigest_flush, METH_NOARGS, "Flush currently open Swing PLA segment."},
    {"segments", (PyCFunction)PyStreamDigest_segments, METH_VARARGS, "Return recent Swing PLA segments."},
    {"anomalies", (PyCFunction)PyStreamDigest_anomalies, METH_VARARGS, "Return verbatim anomaly records."},
    {"zoom", (PyCFunction)PyStreamDigest_zoom, METH_VARARGS, "Lossless O(log N) zoom over [t0, t1]."},
    {"reconstruct", (PyCFunction)(void (*)(void))PyStreamDigest_reconstruct,
     METH_VARARGS | METH_KEYWORDS, "Reconstruct Swing PLA values y_hat(t) with guaranteed L_inf <= eps."},
    {"format_markdown", (PyCFunction)(void (*)(void))PyStreamDigest_format_markdown,
     METH_VARARGS | METH_KEYWORDS, "Format compact ~100-250 token Markdown digest packet."},
    {"reset", (PyCFunction)PyStreamDigest_reset, METH_NOARGS, "Reset segments, anomalies, and ring buffer."},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef PyStreamDigest_getset[] = {
    {"epsilon", (getter)PyStreamDigest_get_eps, NULL, "Guaranteed L_inf error bound.", NULL},
    {"segment_count", (getter)PyStreamDigest_get_seg_cnt, NULL, "Live Swing PLA segment count.", NULL},
    {"anomaly_count", (getter)PyStreamDigest_get_anom_cnt, NULL, "Retained verbatim anomaly count.", NULL},
    {"ring_size", (getter)PyStreamDigest_get_ring_sz, NULL, "Retained raw records in ring buffer.", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject PyStreamDigestType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "chronosieve._core.StreamDigest",
    .tp_basicsize = sizeof(PyStreamDigestObject),
    .tp_dealloc = (destructor)PyStreamDigest_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Anomaly-preserving Swing PLA + Wake-Up Gate + lossless ring-buffer zoom().",
    .tp_methods = PyStreamDigest_methods,
    .tp_getset = PyStreamDigest_getset,
    .tp_init = (initproc)PyStreamDigest_init,
    .tp_new = PyType_GenericNew,
};

/* =========================================================================
 * Module-Level Functions (coma_merge, version, simd_backend)
 * ========================================================================= */

static PyObject *py_coma_merge(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"lowers", "uppers", "weights", "tau", "randomized", "seed", NULL};
    PyObject *lo_obj = NULL, *hi_obj = NULL, *w_obj = Py_None;
    double tau = 0.5;
    int randomized = 0;
    unsigned long long seed = 0x243f6a8885a308d3ULL;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OO|OdpK", kwlist,
            &lo_obj, &hi_obj, &w_obj, &tau, &randomized, &seed)) {
        return NULL;
    }

    cs_py_f64_buf_t lo, hi, w;
    memset(&hi, 0, sizeof(hi));
    memset(&w, 0, sizeof(w));

    if (cs_py_acquire_f64_ro(lo_obj, &lo) != 0) return NULL;
    if (cs_py_acquire_f64_ro(hi_obj, &hi) != 0) {
        cs_py_f64_release(&lo);
        return NULL;
    }
    if (lo.len != hi.len || lo.len == 0U) {
        cs_py_f64_release(&hi);
        cs_py_f64_release(&lo);
        PyErr_SetString(PyExc_ValueError, "lowers and uppers must be non-empty and equal length");
        return NULL;
    }
    if (w_obj != Py_None) {
        if (cs_py_acquire_f64_ro(w_obj, &w) != 0) {
            cs_py_f64_release(&hi);
            cs_py_f64_release(&lo);
            return NULL;
        }
        if (w.len != lo.len) {
            cs_py_f64_release(&w);
            cs_py_f64_release(&hi);
            cs_py_f64_release(&lo);
            PyErr_SetString(PyExc_ValueError, "weights must match lowers length");
            return NULL;
        }
    }

    size_t k = lo.len;
    cs_interval_t *ivs = (cs_interval_t *)malloc(k * sizeof(*ivs));
    if (!ivs) {
        cs_py_f64_release(&w);
        cs_py_f64_release(&hi);
        cs_py_f64_release(&lo);
        return PyErr_NoMemory();
    }

    uint64_t rng_state = (uint64_t)seed;
    cs_coma_result_t res;
    cs_status_t st = cs_coma_merge(
        lo.data, hi.data, w_obj != Py_None ? w.data : NULL, k,
        tau, randomized != 0, randomized ? &rng_state : NULL,
        ivs, k, &res);

    cs_py_f64_release(&w);
    cs_py_f64_release(&hi);
    cs_py_f64_release(&lo);

    if (st != CS_OK) {
        free(ivs);
        cs_set_py_error(st, "coma_merge");
        return NULL;
    }

    PyObject *iv_list = PyList_New((Py_ssize_t)res.n_intervals);
    if (!iv_list) {
        free(ivs);
        return NULL;
    }
    for (size_t i = 0; i < res.n_intervals; ++i) {
        PyObject *tup = Py_BuildValue("(dd)", ivs[i].lo, ivs[i].hi);
        if (!tup) {
            Py_DECREF(iv_list);
            free(ivs);
            return NULL;
        }
        PyList_SET_ITEM(iv_list, (Py_ssize_t)i, tup);
    }
    free(ivs);

    PyObject *hull = NULL;
    if (res.n_intervals > 0U) {
        hull = Py_BuildValue("(dd)", res.hull_lo, res.hull_hi);
        if (!hull) {
            Py_DECREF(iv_list);
            return NULL;
        }
    } else {
        Py_INCREF(Py_None);
        hull = Py_None;
    }

    PyObject *out = Py_BuildValue(
        "{s:O,s:O,s:d,s:d}",
        "intervals", iv_list,
        "hull", hull,
        "max_agreement", res.max_agreement,
        "used_tau", res.used_tau);
    Py_DECREF(iv_list);
    Py_DECREF(hull);
    return out;
}

static PyObject *py_version(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(ignored)) {
    return PyUnicode_FromString(cs_version_string());
}

static PyObject *py_simd_backend(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(ignored)) {
    return PyUnicode_FromString(cs_simd_backend_name());
}

static PyMethodDef core_methods[] = {
    {"coma_merge", (PyCFunction)(void (*)(void))py_coma_merge, METH_VARARGS | METH_KEYWORDS,
     "O(K log K) COMA weighted majority-vote interval consensus (Gasparin & Ramdas UAI 2024)."},
    {"version", py_version, METH_NOARGS, "Return ChronoSieve C core version string."},
    {"simd_backend", py_simd_backend, METH_NOARGS, "Return compiled SIMD backend name."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef core_module = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "ChronoSieve zero-dependency C11/SIMD extension.",
    -1,
    core_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__core(void) {
    if (PyType_Ready(&PyEDetectorType) < 0 ||
        PyType_Ready(&PyEFusionType) < 0 ||
        PyType_Ready(&PyConformalPIDType) < 0 ||
        PyType_Ready(&PyConformalSoAType) < 0 ||
        PyType_Ready(&PySleepingBOAType) < 0 ||
        PyType_Ready(&PyStreamDigestType) < 0) {
        return NULL;
    }

    PyObject *m = PyModule_Create(&core_module);
    if (!m) {
        return NULL;
    }

    Py_INCREF(&PyEDetectorType);
    Py_INCREF(&PyEFusionType);
    Py_INCREF(&PyConformalPIDType);
    Py_INCREF(&PyConformalSoAType);
    Py_INCREF(&PySleepingBOAType);
    Py_INCREF(&PyStreamDigestType);

    PyModule_AddObject(m, "EDetector", (PyObject *)&PyEDetectorType);
    PyModule_AddObject(m, "EFusion", (PyObject *)&PyEFusionType);
    PyModule_AddObject(m, "ConformalPID", (PyObject *)&PyConformalPIDType);
    PyModule_AddObject(m, "ConformalBatchPID", (PyObject *)&PyConformalSoAType);
    PyModule_AddObject(m, "SleepingBOA", (PyObject *)&PySleepingBOAType);
    PyModule_AddObject(m, "StreamDigest", (PyObject *)&PyStreamDigestType);

    PyModule_AddIntConstant(m, "FLAG_ANOMALY", (long)CS_FLAG_ANOMALY);
    PyModule_AddIntConstant(m, "FLAG_SHIFT", (long)CS_FLAG_SHIFT);
    PyModule_AddIntConstant(m, "FLAG_ROGUE", (long)CS_FLAG_ROGUE);
    PyModule_AddIntConstant(m, "FLAG_PINNED", (long)CS_FLAG_PINNED);

    return m;
}
