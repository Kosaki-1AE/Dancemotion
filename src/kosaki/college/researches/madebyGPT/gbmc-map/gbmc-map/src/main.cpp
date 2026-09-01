#include "gbmc_map.hpp"

int main(int argc, char** argv) {
    auto map = gbmc::default_map();

    if (argc <= 1) {
        std::cout << "GBMC Map\n\n";
        std::cout << "Usage:\n";
        std::cout << "  ./gbmc_map <concept>\n";
        std::cout << "  ./gbmc_map --nodes\n\n";
        std::cout << "Examples:\n";
        std::cout << "  ./gbmc_map Groove\n";
        std::cout << "  ./gbmc_map Attention\n";
        std::cout << "  ./gbmc_map Pipe\n";
        std::cout << "  ./gbmc_map Translator\n";
        return 0;
    }

    std::string query = argv[1];

    if (query == "--nodes") {
        map.print_all_nodes();
        return 0;
    }

    map.print_position(query);

    return 0;
}
