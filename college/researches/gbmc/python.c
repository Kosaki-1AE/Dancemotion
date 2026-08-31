#include <Python.h>

// 【Motion Phase】
// 実際にCPUを叩く関数（C言語の中にアセンブリを埋め込む）
static PyObject* gbmc_add_asm(PyObject* self, PyObject* args) {
    int a, b;
    int result;

    // 1. Pythonからの入力（Genesys）をCの変数（Buffer）に変換
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;
    }

    // 2. インラインアセンブリ（Motion）
    // ここでC言語の変数をレジスタに流し込み、CPUの命令で足し算を行う
    // "addl %%ebx, %%eax" : ebxの内容をeaxに足す
    __asm__ volatile (
        "movl %1, %%eax;"  // 変数aをEAXレジスタへ
        "movl %2, %%ebx;"  // 変数bをEBXレジスタへ
        "addl %%ebx, %%eax;" // 足し算実行 (EAX = EAX + EBX)
        "movl %%eax, %0;"  // 結果を変数resultに戻す
        : "=r" (result)    // 出力
        : "r" (a), "r" (b) // 入力
        : "%eax", "%ebx"   // 破壊されるレジスタの申告
    );

    // 3. 結果をPythonに返す
    return Py_BuildValue("i", result);
}

// モジュールの定義（Pythonに「こういう関数があるよ」と教える設定）
static PyMethodDef GbmcMethods[] = {
    {"add_asm", gbmc_add_asm, METH_VARARGS, "Add two numbers using Assembly."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef gbmcmodule = {
    PyModuleDef_HEAD_INIT,
    "gbmc_core",
    "GBMC Core Module connecting Python to Asm",
    -1,
    GbmcMethods
};

// モジュールの初期化
PyMODINIT_FUNC PyInit_gbmc_core(void) {
    return PyModule_Create(&gbmcmodule);
}