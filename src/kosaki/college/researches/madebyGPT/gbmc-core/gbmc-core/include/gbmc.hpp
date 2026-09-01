#ifndef GBMC_HPP
#define GBMC_HPP

#include <iostream>
#include <string>
#include <vector>
#include <utility>

namespace gbmc {

/*
  GBMC Core Concepts

  Module:
    An isolated part. It does not know other modules directly.

  Port:
    A mediation surface. Modules connect only through compatible ports.

  ResponsibilityRange:
    The range in which a module can guarantee a state transition.

  Connection:
    A proposed relationship between two modules.

  Resonance:
    A connection is resonant when port compatibility and responsibility
    range compatibility are both satisfied.
*/

struct Port {
    std::string name;
    std::string type;

    Port() = default;
    Port(std::string n, std::string t)
        : name(std::move(n)), type(std::move(t)) {}
};

struct ResponsibilityRange {
    double min_value = 0.0;
    double max_value = 0.0;

    ResponsibilityRange() = default;
    ResponsibilityRange(double min_v, double max_v)
        : min_value(min_v), max_value(max_v) {}

    bool contains(double value) const {
        return min_value <= value && value <= max_value;
    }

    bool overlaps(const ResponsibilityRange& other) const {
        return !(max_value < other.min_value || other.max_value < min_value);
    }
};

struct Module {
    std::string name;
    std::string represent;
    std::vector<Port> inputs;
    std::vector<Port> outputs;
    ResponsibilityRange responsibility;

    Module() = default;

    Module(std::string n, std::string rep, ResponsibilityRange r)
        : name(std::move(n)), represent(std::move(rep)), responsibility(r) {}

    void add_input(const Port& port) { inputs.push_back(port); }
    void add_output(const Port& port) { outputs.push_back(port); }
};

struct ResonanceResult {
    bool port_compatible = false;
    bool responsibility_compatible = false;
    bool resonant = false;
    std::string message;
};

struct Connection {
    Module* from = nullptr;
    Module* to = nullptr;
    Port out_port;
    Port in_port;

    Connection(Module& f, Module& t, Port out, Port in)
        : from(&f), to(&t), out_port(std::move(out)), in_port(std::move(in)) {}

    ResonanceResult evaluate() const {
        ResonanceResult result;

        if (!from || !to) {
            result.message = "Invalid connection: module pointer is null.";
            return result;
        }

        result.port_compatible = (out_port.type == in_port.type);
        result.responsibility_compatible = from->responsibility.overlaps(to->responsibility);
        result.resonant = result.port_compatible && result.responsibility_compatible;

        if (result.resonant) result.message = "Resonance: connection can flow.";
        else if (!result.port_compatible) result.message = "Meaning wall: port types are not compatible.";
        else result.message = "Meaning wall: responsibility ranges do not overlap.";

        return result;
    }
};

inline void print_module(const Module& m) {
    std::cout << "Module: " << m.name << "\n";
    std::cout << "  Represent: " << m.represent << "\n";
    std::cout << "  ResponsibilityRange: [" << m.responsibility.min_value << ", " << m.responsibility.max_value << "]\n";

    std::cout << "  Inputs:\n";
    for (const auto& p : m.inputs) std::cout << "    - " << p.name << " : " << p.type << "\n";

    std::cout << "  Outputs:\n";
    for (const auto& p : m.outputs) std::cout << "    - " << p.name << " : " << p.type << "\n";
}

inline void print_connection(const Connection& c) {
    auto r = c.evaluate();
    std::cout << "\nConnection:\n";
    std::cout << "  " << c.from->name << "." << c.out_port.name << " -> " << c.to->name << "." << c.in_port.name << "\n";
    std::cout << "  Port compatible: " << (r.port_compatible ? "yes" : "no") << "\n";
    std::cout << "  Responsibility compatible: " << (r.responsibility_compatible ? "yes" : "no") << "\n";
    std::cout << "  Result: " << (r.resonant ? "RESONANT" : "BLOCKED") << "\n";
    std::cout << "  Message: " << r.message << "\n";
}

} // namespace gbmc

#endif // GBMC_HPP
