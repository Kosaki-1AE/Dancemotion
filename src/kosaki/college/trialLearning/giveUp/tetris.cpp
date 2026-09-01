#include <iostream>
#include <thread>
#include <chrono>

char field[20][10];

void drawField() {
    system("cls");  // mac/linuxなら system("clear")
    for (int y = 0; y < 20; ++y) {
        for (int x = 0; x < 10; ++x) {
            std::cout << (field[y][x] ? "■" : " ");
        }
        std::cout << std::endl;
    }
}

int main() {
    while (true) {
        drawField();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        // フィールドのどこかに1を入れてみよう
        field[rand() % 20][rand() % 10] = 1;
    }
}
