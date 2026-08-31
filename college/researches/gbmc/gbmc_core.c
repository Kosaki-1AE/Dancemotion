
#include <Python.h>

// ここにアセンブリ直書きの「Motion」がある
static PyObject* gbmc_add_asm(PyObject* self, PyObject* args) {
    int a, b;
    int result;

    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;
    }

    // インラインアセンブリ (x86_64)
    // Pythonから受け取った数値を、CPUレジスタで直接演算
    __asm__ volatile (
        "movl %1, %%eax;"
        "movl %2, %%ebx;"
        "addl %%ebx, %%eax;"
        "movl %%eax, %0;"
        : "=r" (result)
        : "r" (a), "r" (b)
        : "%eax", "%ebx"
    );

    return Py_BuildValue("i", result);
}

// モジュール定義
static PyMethodDef GbmcMethods[] = {
    {"add_asm", gbmc_add_asm, METH_VARARGS, "Add via Assembly"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef gbmcmodule = {
    PyModuleDef_HEAD_INIT,
    "gbmc_core",
    "GBMC Core Module",
    -1,
    GbmcMethods
};

PyMODINIT_FUNC PyInit_gbmc_core(void) {
    return PyModule_Create(&gbmcmodule);
}
