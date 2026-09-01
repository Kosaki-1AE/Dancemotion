/*
 * GBMC Hybrid Kernel v1.0
 * Architecture: C (Body/Motion) + Python (Mind/Genesys)
 * Concept: Embedded Intelligence in Quantum Swarm
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

// ==========================================
// 1. Buffer Phase (構造定義)
// ==========================================
#define GRID_SIZE 9
#define Q_COUNT (GRID_SIZE * GRID_SIZE)

typedef struct {
    int8_t state;           // 1(前進), -1(後退), 0(停滞)
    int8_t responsibility;  // Pythonから与えられる意思 (-1 ~ 1)
    long energy;            // 蓄積エネルギー
} Quantum;

Quantum matrix[Q_COUNT];
volatile int keep_running = 1;
long total_system_energy = 0;

// ==========================================
// 2. Motion Phase (物理動作 - C言語の領域)
// ==========================================
void* physics_loop(void* arg) {
    while (keep_running) {
        long temp_energy = 0;
        for (int i = 0; i < Q_COUNT; i++) {
            // カオス(乱数) + 意思(責任ベクトル)
            int noise = (rand() % 3) - 1; 
            int intention = matrix[i].responsibility;
            
            // 物理法則: 意思が強くてもノイズで揺らぐ
            int action = noise + intention;
            
            // クランプ (-1 ~ 1)
            if (action > 1) action = 1;
            if (action < -1) action = -1;

            matrix[i].state = action;
            matrix[i].energy += action;
            temp_energy += matrix[i].energy;
        }
        total_system_energy = temp_energy;
        
        // 人間の目に見える速度に調整 (10ms)
        usleep(10000); 
    }
    return NULL;
}

// ==========================================
// 3. Genesys Phase (思考 - Pythonの領域)
// ==========================================

// Pythonの脳（スクリプト）を定義
// ここを書き換えれば、OSの性格が即座に変わる
const char* python_brain_script = 
"import random\n"
"\n"
"def think(current_energy):\n"
"    # --- ここが思考の核心 (Genesys) ---\n"
"    # エネルギーが溜まりすぎたら放出(後退)させ、\n"
"    # 少なすぎたらチャージ(前進)させるホメオスタシス機能\n"
"\n"
"    decision = 0\n"
"    reason = '停滞(観測中)'\n"
"\n"
"    if current_energy > 500:\n"
"        decision = -1\n"
"        reason = '過剰エネルギー：冷却します(▼)'\n"
"    elif current_energy < -500:\n"
"        decision = 1\n"
"        reason = 'エネルギー不足：加速します(▲)'\n"
"    else:\n"
"        # 安定しているときはランダムに揺らぐ\n"
"        if random.random() > 0.8:\n"
"            decision = random.choice([-1, 1])\n"
"            reason = '気まぐれな探索'\n"
"\n"
"    return decision, reason\n";

int main(int argc, char *argv[]) {
    srand(time(NULL));

    // --- A. 身体の起動 (C Setup) ---
    printf("[-] System Boot: Initializing Quantum Field...\n");
    for(int i=0; i<Q_COUNT; i++) {
        matrix[i].state = 0;
        matrix[i].responsibility = 0;
        matrix[i].energy = 0;
    }

    // --- B. 脳の覚醒 (Python Init) ---
    printf("[-] System Boot: Awakening Python Mind...\n");
    Py_Initialize();
    
    // Pythonスクリプトを読み込ませて関数を定義する
    PyRun_SimpleString(python_brain_script);

    // 定義した関数オブジェクトを取得しておく
    PyObject *pName = PyUnicode_DecodeFSDefault("__main__");
    PyObject *pModule = PyImport_Import(pName);
    PyObject *pFunc = PyObject_GetAttrString(pModule, "think");

    if (!pFunc || !PyCallable_Check(pFunc)) {
        fprintf(stderr, "Critical Error: Python brain function 'think' not found.\n");
        return 1;
    }

    // --- C. 生命活動開始 (Start Threads) ---
    pthread_t p_thread;
    pthread_create(&p_thread, NULL, physics_loop, NULL);

    printf("[-] System Ready. Chaos engine running.\n\n");

    // --- メインループ: 思考と観測 ---
    while (keep_running) {
        // 1. CがPythonに現在の状況(エネルギー)を報告
        PyObject *pArgs = PyTuple_New(1);
        PyTuple_SetItem(pArgs, 0, PyLong_FromLong(total_system_energy));

        // 2. Pythonが思考する (関数呼び出し)
        PyObject *pValue = PyObject_CallObject(pFunc, pArgs);

        // 3. 結果の解析 (Python -> C)
        if (pValue != NULL) {
            // 戻り値は (decision, reason) のタプル
            PyObject *pDecision = PyTuple_GetItem(pValue, 0);
            PyObject *pReason = PyTuple_GetItem(pValue, 1);

            int new_vector = (int)PyLong_AsLong(pDecision);
            
            // UTF-8文字列として理由を取得
            PyObject* pStrObj = PyUnicode_AsUTF8String(pReason);
            const char* reason_str = PyBytes_AsString(pStrObj);

            // 4. 責任ベクトルの適用 (脳からの指令を身体に伝える)
            for(int i=0; i<Q_COUNT; i++) {
                matrix[i].responsibility = new_vector;
            }

            // --- 可視化 (Visualization) ---
            printf("\033[H\033[J"); // 画面クリア
            printf("=== GBMC HYBRID KERNEL ===\n");
            printf("Energy Level: %ld\n", total_system_energy);
            printf("Mind State  : %s\n\n", reason_str); // Pythonの思考を表示

            // 量子グリッド描画
            for (int y = 0; y < GRID_SIZE; y++) {
                for (int x = 0; x < GRID_SIZE; x++) {
                    int idx = y * GRID_SIZE + x;
                    if (matrix[idx].state == 1) printf("\033[32m▲\033[0m ");
                    else if (matrix[idx].state == -1) printf("\033[31m▼\033[0m ");
                    else printf("\033[90m·\033[0m ");
                }
                printf("\n");
            }
            
            Py_DECREF(pStrObj);
            Py_DECREF(pValue);
        } else {
            PyErr_Print(); // エラーがあれば表示
        }
        
        Py_DECREF(pArgs);
        
        // 思考サイクル (0.1秒ごとに判断)
        usleep(100000); 
    }

    // --- 終了処理 ---
    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
    Py_DECREF(pName);
    Py_Finalize();
    return 0;
}