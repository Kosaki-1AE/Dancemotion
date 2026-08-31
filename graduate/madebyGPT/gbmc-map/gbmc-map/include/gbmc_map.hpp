#ifndef GBMC_MAP_HPP
#define GBMC_MAP_HPP

#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <set>
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

struct ConceptNode {
    std::string name;
    std::string domain;
    std::string role;
    std::string description;
};

struct Edge {
    std::string from;
    std::string to;
    std::string relation;
    double confidence = 1.0;
};

struct Mapping {
    std::string external;
    std::string gbmc;
    std::string reason;
    double confidence = 1.0;
};

class GBMCMap {
public:
    void add_node(ConceptNode node) {
        nodes_[node.name] = std::move(node);
    }

    void add_edge(Edge edge) {
        edges_.push_back(std::move(edge));
    }

    void add_mapping(Mapping mapping) {
        mappings_.push_back(std::move(mapping));
    }

    std::vector<Mapping> find_mapping(const std::string& query) const {
        std::vector<Mapping> result;
        const auto q = lower(query);

        for (const auto& m : mappings_) {
            const auto external = lower(m.external);
            const auto gbmc = lower(m.gbmc);

            if (external.find(q) != std::string::npos ||
                q.find(external) != std::string::npos ||
                gbmc.find(q) != std::string::npos ||
                q.find(gbmc) != std::string::npos) {
                result.push_back(m);
            }
        }

        std::sort(result.begin(), result.end(),
            [](const Mapping& a, const Mapping& b) {
                return a.confidence > b.confidence;
            });

        return result;
    }

    bool has_node(const std::string& name) const {
        return nodes_.find(name) != nodes_.end();
    }

    const ConceptNode* node(const std::string& name) const {
        auto it = nodes_.find(name);
        if (it == nodes_.end()) return nullptr;
        return &it->second;
    }

    std::vector<Edge> neighbors(const std::string& name) const {
        std::vector<Edge> result;

        for (const auto& e : edges_) {
            if (e.from == name || e.to == name) {
                result.push_back(e);
            }
        }

        std::sort(result.begin(), result.end(),
            [](const Edge& a, const Edge& b) {
                return a.confidence > b.confidence;
            });

        return result;
    }

    void print_position(const std::string& query) const {
        std::cout << "=== GBMC Map ===\n";
        std::cout << "Query: " << query << "\n\n";

        auto mappings = find_mapping(query);

        if (mappings.empty()) {
            if (has_node(query)) {
                print_node_position(query);
                return;
            }

            std::cout << "No position found.\n";
            std::cout << "Try: Groove, Attention, Pipe, Observer, Isolation, ServoLimit, Port, Resonance.\n";
            return;
        }

        const auto& best = mappings.front();

        std::cout << "Best Position:\n";
        std::cout << "  " << best.external << " -> GBMC." << best.gbmc << "\n";
        std::cout << "  confidence: " << best.confidence << "\n";
        std::cout << "  reason    : " << best.reason << "\n\n";

        print_node_position(best.gbmc);

        if (mappings.size() > 1) {
            std::cout << "\nOther candidate positions:\n";
            for (std::size_t i = 1; i < mappings.size(); ++i) {
                std::cout << "  - " << mappings[i].external
                          << " -> GBMC." << mappings[i].gbmc
                          << " (" << mappings[i].confidence << ")\n";
            }
        }
    }

    void print_node_position(const std::string& name) const {
        const auto* n = node(name);

        if (!n) {
            std::cout << "GBMC node not found: " << name << "\n";
            return;
        }

        std::cout << "Current GBMC Node:\n";
        std::cout << "  name  : " << n->name << "\n";
        std::cout << "  domain: " << n->domain << "\n";
        std::cout << "  role  : " << n->role << "\n";
        std::cout << "  desc  : " << n->description << "\n\n";

        std::cout << "Nearby Nodes:\n";

        auto ns = neighbors(name);

        if (ns.empty()) {
            std::cout << "  none\n";
            return;
        }

        for (const auto& e : ns) {
            std::string other = (e.from == name) ? e.to : e.from;
            std::cout << "  " << name << " --[" << e.relation << "]--> "
                      << other << " (" << e.confidence << ")\n";
        }

        std::cout << "\nLocal Interpretation:\n";
        interpret(name);
    }

    void interpret(const std::string& name) const {
        if (name == "Module") {
            std::cout << "  Module is an isolated part. It becomes useful only when ports allow it to connect.\n";
        } else if (name == "Port") {
            std::cout << "  Port is the mediation surface. It decides what can be interpreted and connected.\n";
        } else if (name == "Translator") {
            std::cout << "  Translator converts incompatible types or concepts so that a flow can continue.\n";
        } else if (name == "ResponsibilityRange") {
            std::cout << "  ResponsibilityRange defines what a module can guarantee without collapsing.\n";
        } else if (name == "Resonance") {
            std::cout << "  Resonance is not just connection. It is a bidirectional transition that can continue.\n";
        } else if (name == "ReferenceFrame") {
            std::cout << "  ReferenceFrame is the represent. It fixes where meaning and responsibility are measured from.\n";
        } else if (name == "Flow") {
            std::cout << "  Flow is the ordered transition across connected modules.\n";
        } else if (name == "MeaningWall") {
            std::cout << "  MeaningWall appears when port, translator, or responsibility range is undefined.\n";
        } else {
            std::cout << "  This node is part of the GBMC local structure.\n";
        }
    }

    void print_all_nodes() const {
        std::cout << "=== GBMC Nodes ===\n\n";
        for (const auto& pair : nodes_) {
            std::cout << "- " << pair.second.name
                      << " : " << pair.second.role << "\n";
        }
    }

private:
    std::map<std::string, ConceptNode> nodes_;
    std::vector<Edge> edges_;
    std::vector<Mapping> mappings_;
};

inline GBMCMap default_map() {
    GBMCMap map;

    map.add_node({"Module", "GBMC", "isolated part", "A part that can exist independently before being connected."});
    map.add_node({"Port", "GBMC", "mediation surface", "The surface through which another module can be interpreted."});
    map.add_node({"Translator", "GBMC", "type/concept converter", "Converts incompatible signals or concepts into a connectable form."});
    map.add_node({"ResponsibilityRange", "GBMC", "guarantee boundary", "The range where a module can guarantee transition."});
    map.add_node({"Resonance", "GBMC", "continuable connection", "A connection whose transition can continue without collapse."});
    map.add_node({"ReferenceFrame", "GBMC", "represent/origin", "The represent that defines where meaning starts."});
    map.add_node({"Flow", "GBMC", "transition order", "The ordered sequence of connected modules."});
    map.add_node({"MeaningWall", "GBMC", "undefined boundary", "A wall that appears when port, translator, or responsibility is undefined."});
    map.add_node({"ModuleState", "GBMC", "current condition", "The current state of a module inside a transition."});

    map.add_edge({"Module", "Port", "exposes", 0.95});
    map.add_edge({"Port", "Translator", "requires-when-incompatible", 0.88});
    map.add_edge({"Translator", "Flow", "allows-continuation", 0.84});
    map.add_edge({"Port", "Resonance", "enables", 0.90});
    map.add_edge({"Resonance", "Flow", "stabilizes", 0.86});
    map.add_edge({"ResponsibilityRange", "MeaningWall", "defines-boundary-of", 0.91});
    map.add_edge({"ReferenceFrame", "ResponsibilityRange", "anchors", 0.87});
    map.add_edge({"ReferenceFrame", "MeaningWall", "reduces", 0.80});
    map.add_edge({"ModuleState", "Flow", "moves-through", 0.78});
    map.add_edge({"Translator", "MeaningWall", "resolves", 0.83});

    map.add_mapping({"Attention", "Port", "Attention mediates which information can connect to the next representation.", 0.72});
    map.add_mapping({"Query-Key Matching", "Resonance", "Query and key form a usable relation when their states align.", 0.66});
    map.add_mapping({"Embedding", "Module", "Embedding isolates an item as a manipulable representation.", 0.55});
    map.add_mapping({"Observer", "Translator", "An observer translates measured state into an internal estimate.", 0.76});
    map.add_mapping({"Feedback", "ResponsibilityRange", "Feedback keeps behavior within a correctable range.", 0.70});
    map.add_mapping({"State", "ModuleState", "State is the current condition of a system or module.", 0.69});
    map.add_mapping({"Groove", "Resonance", "Groove emerges when body, rhythm, and space continue together.", 0.84});
    map.add_mapping({"Isolation", "Module", "Isolation separates parts so they can be reconnected freely.", 0.88});
    map.add_mapping({"Represent", "ReferenceFrame", "Represent fixes the coordinate system for meaning and responsibility.", 0.91});
    map.add_mapping({"Pipe", "Port", "A pipe is a standard mediation surface between isolated commands.", 0.86});
    map.add_mapping({"stdin/stdout", "Port", "stdin/stdout define the shared surface for command composition.", 0.89});
    map.add_mapping({"Small Tool", "Module", "A small tool is an isolated module with a standard interface.", 0.82});
    map.add_mapping({"Servo Limit", "ResponsibilityRange", "A servo limit defines the physical range that can be guaranteed.", 0.87});
    map.add_mapping({"AngleToPWM", "Translator", "AngleToPWM converts pilot intention into executable servo signal.", 0.93});

    return map;
}

} // namespace gbmc

#endif // GBMC_MAP_HPP
