#include "gbmc_concept.hpp"

int main(int argc, char** argv) {
    auto translator = gbmc::default_translator();

    if (argc <= 1) {
        std::cout << "GBMC Concept Translator\n\n";
        std::cout << "Usage:\n";
        std::cout << "  ./gbmc_translate <concept>\n";
        std::cout << "  ./gbmc_translate --all\n\n";
        std::cout << "Examples:\n";
        std::cout << "  ./gbmc_translate Attention\n";
        std::cout << "  ./gbmc_translate Groove\n";
        std::cout << "  ./gbmc_translate Pipe\n";
        return 0;
    }

    std::string query = argv[1];

    if (query == "--all") {
        translator.print_all();
        return 0;
    }

    translator.print_query(query);

    return 0;
}
