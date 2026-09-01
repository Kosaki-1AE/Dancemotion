#include <iostream>
#include <string>
#include <vector>

class Room {
public:
    std::string description;
    std::vector<int> exits;
    
    Room(std::string desc) : description(desc) {}
};

class Game {
private:
    std::vector<Room> rooms;
    int currentRoom;

public:
    Game() {
        // 部屋を初期化
        rooms.push_back(Room("あなたは暗い洞窟の入り口にいます。"));
        rooms.push_back(Room("宝物が散らばる広間に到着しました。"));
        rooms.push_back(Room("深い淵の前に立っています。"));

        // 出口を設定
        rooms[0].exits = {1};
        rooms[1].exits = {0, 2};
        rooms[2].exits = {1};

        currentRoom = 0;
    }

    void play() {
        while (true) {
            std::cout << rooms[currentRoom].description << std::endl;
            std::cout << "どちらに進みますか？ (0: 戻る, 1: 進む, 2: 終了): ";
            int choice;
            std::cin >> choice;

            if (choice == 2) break;
            if (choice >= 0 && choice < rooms[currentRoom].exits.size()) {
                currentRoom = rooms[currentRoom].exits[choice];
            } else {
                std::cout << "その方向には進めません。" << std::endl;
            }
        }
    }
};

int main() {
    Game game;
    game.play();
    return 0;
}
