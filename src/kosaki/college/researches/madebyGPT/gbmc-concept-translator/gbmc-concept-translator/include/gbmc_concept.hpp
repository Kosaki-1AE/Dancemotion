#ifndef GBMC_CONCEPT_HPP
#define GBMC_CONCEPT_HPP

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace gbmc {

inline std::string lower(std::string s) {
    for (auto& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

struct Mapping {
    std::string source_domain;
    std::string source_concept;
    std::string gbmc_concept;
    double confidence = 0.0;
    std::string reason;
};

class ConceptTranslator {
public:
    void add(Mapping mapping) {
        mappings_.push_back(std::move(mapping));
    }

    std::vector<Mapping> query(const std::string& concept) const {
        const auto q = lower(concept);
        std::vector<Mapping> result;

        for (const auto& m : mappings_) {
            const auto source = lower(m.source_concept);
            const auto gbmc = lower(m.gbmc_concept);

            if (source.find(q) != std::string::npos ||
                q.find(source) != std::string::npos ||
                gbmc.find(q) != std::string::npos) {
                result.push_back(m);
            }
        }

        std::sort(result.begin(), result.end(),
            [](const Mapping& a, const Mapping& b) {
                return a.confidence > b.confidence;
            });

        return result;
    }

    void print_query(const std::string& concept) const {
        auto result = query(concept);

        std::cout << "=== GBMC Concept Translator ===\n";
        std::cout << "Query: " << concept << "\n\n";

        if (result.empty()) {
            std::cout << "No mapping found.\n";
            std::cout << "Hint: try Attention, Observer, Groove, State, Pipe, Feedback.\n";
            return;
        }

        for (std::size_t i = 0; i < result.size(); ++i) {
            const auto& m = result[i];

            std::cout << i + 1 << ". "
                      << m.source_domain << "." << m.source_concept
                      << " -> GBMC." << m.gbmc_concept << "\n";
            std::cout << "   confidence: " << m.confidence << "\n";
            std::cout << "   reason    : " << m.reason << "\n\n";
        }
    }

    void print_all() const {
        std::cout << "=== GBMC Mapping Dictionary ===\n\n";

        for (const auto& m : mappings_) {
            std::cout << m.source_domain << "." << m.source_concept
                      << " -> GBMC." << m.gbmc_concept
                      << " (" << m.confidence << ")\n";
        }
    }

private:
    std::vector<Mapping> mappings_;
};

inline ConceptTranslator default_translator() {
    ConceptTranslator t;

    t.add({
        "Transformer",
        "Attention",
        "Port",
        0.72,
        "Attention mediates which information can connect to the next representation."
    });

    t.add({
        "Transformer",
        "Query-Key Matching",
        "Resonance",
        0.66,
        "Query and key become meaningful when their relation produces a usable connection."
    });

    t.add({
        "Transformer",
        "Embedding",
        "Module",
        0.55,
        "An embedding isolates a token as a manipulable representation."
    });

    t.add({
        "ControlTheory",
        "Observer",
        "Translator",
        0.76,
        "An observer translates measured state into an internal estimate."
    });

    t.add({
        "ControlTheory",
        "Feedback",
        "ResponsibilityRange",
        0.70,
        "Feedback constrains behavior into a predictable and correctable range."
    });

    t.add({
        "ControlTheory",
        "State",
        "ModuleState",
        0.69,
        "A state describes where a module currently is inside a transition."
    });

    t.add({
        "StreetDance",
        "Groove",
        "Resonance",
        0.84,
        "Groove emerges when body, rhythm, and space transition together."
    });

    t.add({
        "StreetDance",
        "Isolation",
        "Module",
        0.88,
        "Isolation separates body parts so that each part can be reconnected freely."
    });

    t.add({
        "StreetDance",
        "Represent",
        "ReferenceFrame",
        0.91,
        "Represent works as the reference frame from which meaning and responsibility are defined."
    });

    t.add({
        "Unix",
        "Pipe",
        "Port",
        0.86,
        "A pipe is a common mediation interface between isolated commands."
    });

    t.add({
        "Unix",
        "Small Tool",
        "Module",
        0.82,
        "A small tool is an isolated module that does one thing and connects through a standard interface."
    });

    t.add({
        "Unix",
        "stdin/stdout",
        "Port",
        0.89,
        "stdin/stdout define the connection surface that allows tools to compose."
    });

    t.add({
        "HumanPoweredAircraft",
        "Servo Limit",
        "ResponsibilityRange",
        0.87,
        "A servo limit defines the range where the physical output can be guaranteed."
    });

    t.add({
        "HumanPoweredAircraft",
        "AngleToPWM",
        "Translator",
        0.93,
        "AngleToPWM translates pilot intention into an executable servo signal."
    });

    return t;
}

} // namespace gbmc

#endif // GBMC_CONCEPT_HPP
