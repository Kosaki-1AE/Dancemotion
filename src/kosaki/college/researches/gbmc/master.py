import importlib.util
import os
import subprocess
import sys

# ==========================================
# Phase 1: Genesys (思考・生成)
# Pythonの中に「Cとアセンブリの設計図」を持っておく
# ==========================================

# C言語のソースコード（アセンブリ含む）を文字列として定義
c_source_code = """
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
"""

# ビルド用の setup.py もPythonが勝手に書く
setup_py_code = """
from setuptools import setup, Extension
module = Extension('gbmc_core', sources=['gbmc_core.c'])
setup(
    name='gbmc_core',
    version='1.0',
    ext_modules=[module]
)
"""

def generate_files():
    """ソースコードをファイルとして具現化する"""
    print("[-] Genesys Phase: ソースコードを生成中...")
    with open('gbmc_core.c', 'w') as f:
        f.write(c_source_code)
    with open('setup_gbmc.py', 'w') as f:
        f.write(setup_py_code)

# ==========================================
# Phase 2: Buffer (構造化・コンパイル)
# Pythonがコンパイラ(gcc等)を呼び出してビルドさせる
# ==========================================

def compile_extension():
    """C言語をコンパイルして共有ライブラリ(.so/.pyd)にする"""
    print("[-] Buffer Phase: コンパイルを実行中...")
    
    # コマンド: python setup_gbmc.py build_ext --inplace
    try:
        subprocess.check_call(
            [sys.executable, 'setup_gbmc.py', 'build_ext', '--inplace'],
            stdout=subprocess.DEVNULL, # ログを黙らせる（エラー時のみ表示）
            stderr=subprocess.STDOUT
        )
        print("    -> コンパイル成功。強固な構造を獲得しました。")
    except subprocess.CalledProcessError:
        print("!!! コンパイルエラー発生 !!!")
        sys.exit(1)

# ==========================================
# Phase 3: Motion (動作・実行)
# 生成された筋肉(Asm)を使って動く
# ==========================================

def load_and_run():
    """生成されたモジュールを動的にロードして実行する"""
    print("[-] Motion Phase: アセンブリを実行します...")
    
    # コンパイルされたファイル名を探す（環境によって名前が変わるため）
    # 例: gbmc_core.cpython-310-x86_64-linux-gnu.so
    so_file = None
    for file in os.listdir('.'):
        if file.startswith('gbmc_core') and (file.endswith('.so') or file.endswith('.pyd')):
            so_file = file
            break
            
    if not so_file:
        print("モジュールが見つかりません。")
        return

    # 動的インポート（import gbmc_core と同じことを手動でやる）
    spec = importlib.util.spec_from_file_location("gbmc_core", so_file)
    gbmc_core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gbmc_core)

    # 実行！
    val1, val2 = 100, 55
    result = gbmc_core.add_asm(val1, val2)
    
    print(f"\n    Result: {val1} + {val2} = {result} (Calculated by CPU Register)")
    print("    -> 正常動作確認完了。\n")

# ==========================================
# Main Loop
# ==========================================

if __name__ == "__main__":
    # 1. 生成
    generate_files()
    
    # 2. コンパイル
    compile_extension()
    
    # 3. 実行
    load_and_run()
    
    # (オプション) 終わったら生成物を消す「完全な動中静」にするならここでお掃除
    # os.remove('gbmc_core.c')
    # os.remove('setup_gbmc.py')