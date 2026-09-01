/*
 * GBMC Quantum Core v0.1
 * Concept: Stillness in Motion (動中静)
 * Author: GBMC Architect
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

// 定数定義
#define GRID_SIZE 9
#define QUANTUM_COUNT (GRID_SIZE * GRID_SIZE) // 81個
#define CYCLE_MICROSEC 50000 // 更新速度（50ms = 人間の目にヌルヌル見える速度）

// 量子構造体 (L1キャッシュラインに最適化)
// サイズ: 4 bytes (非常に小さい)
typedef struct {
    int8_t state;           // 現在の状態: 1(前進), -1(後退), 0(停滞)
    int8_t responsibility;  // 責任ベクトル: Pythonから与えられるバイアス (-1 ~ 1)
    int16_t energy;         // 蓄積されたエネルギー（値）
} Quantum;

// 81個の量子群 (Global Field)
Quantum matrix[QUANTUM_COUNT];
int system_running = 1;

// ==========================================
// Motion Phase: アセンブリの領域
// 常に振動し続ける生命維持装置
// ==========================================
void* quantum_pulse(void* arg) {
    while (system_running) {
        for (int i = 0; i < QUANTUM_COUNT; i++) {
            // 1. 純粋なカオス (ノイズ) の生成
            // ここはアセンブリなら RDSEED 命令などを使う場所
            int noise = (rand() % 3) - 1; // -1, 0, 1 のいずれか

            // 2. 責任ベクトルの適用 (バイアス)
            // ノイズ + 意思 = 実際の動き
            int determination = noise + matrix[i].responsibility;

            // 3. 物理的な制約 (クランプ処理)
            // 状態は -1, 0, 1 の範囲を超えない
            if (determination > 1) determination = 1;
            if (determination < -1) determination = -1;

            // 4. 状態の更新
            matrix[i].state = determination;
            
            // 5. エネルギーの蓄積 (積分)
            // これが「移動量」や「計算結果」になる
            matrix[i].energy += determination;
        }
        
        // 鼓動のサイクル待ち
        usleep(CYCLE_MICROSEC);
    }
    return NULL;
}

// ==========================================
// Visualization: 観測フェーズ
// システムの状態を可視化する (OSの画面描画に相当)
// ==========================================
void render_field() {
    // 画面クリア (ANSIエスケープシーケンス)
    printf("\033[H\033[J");
    
    printf("=== GBMC QUANTUM FIELD (9x9) ===\n");
    printf("Controls: [0] Reset, [1] Forward Bias, [2] Backward Bias, [q] Quit\n\n");

    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            int idx = y * GRID_SIZE + x;
            int s = matrix[idx].state;
            
            // 状態に応じた文字を描画
            // 前進(+): '▲', 後退(-): '▼', 停滞(0): '·'
            if (s == 1)      printf(" \033[32m▲\033[0m "); // 緑
            else if (s == -1) printf(" \033[31m▼\033[0m "); // 赤
            else             printf(" \033[90m·\033[0m "); // グレー
        }
        printf("\n");
    }
    printf("\nEnergy Sum: ");
    long total_energy = 0;
    for(int i=0; i<QUANTUM_COUNT; i++) total_energy += matrix[i].energy;
    printf("%ld\n", total_energy);
}

// ==========================================
// Genesys Phase: メイン制御
// Pythonが本来やるべき「責任の付与」をシミュレート
// ==========================================
int main() {
    srand(time(NULL));
    pthread_t thread_id;

    // 量子の初期化
    for (int i = 0; i < QUANTUM_COUNT; i++) {
        matrix[i].state = 0;
        matrix[i].responsibility = 0;
        matrix[i].energy = 0;
    }

    // 鼓動(スレッド)の開始
    if (pthread_create(&thread_id, NULL, quantum_pulse, NULL) != 0) {
        perror("Failed to create heart beat");
        return 1;
    }

    // 入力ループ (非ブロッキング風に簡易実装)
    // 本来はPythonからここを操作する
    char input;
    system("stty -icanon -echo"); // 入力を即座に受け取る設定

    while (system_running) {
        render_field();
        
        // キー入力待ち（簡易的ポーリング）
        // 実際はselectなどを使うが、ここでは見た目重視
        if (read(STDIN_FILENO, &input, 1) > 0) {
            if (input == 'q') system_running = 0;
            
            // 責任ベクトルの注入
            int new_bias = 0;
            if (input == '1') new_bias = 1;  // 全員前進せよ
            if (input == '2') new_bias = -1; // 全員後退せよ
            if (input == '0') new_bias = 0;  // 自由になれ (カオス)

            // ベクトルを一斉送信 (ブロードキャスト)
            for(int i=0; i<QUANTUM_COUNT; i++) {
                matrix[i].responsibility = new_bias;
            }
        }
        usleep(50000); // 描画更新レート
    }

    system("stty cooked echo"); // 設定戻し
    pthread_join(thread_id, NULL);
    return 0;
}