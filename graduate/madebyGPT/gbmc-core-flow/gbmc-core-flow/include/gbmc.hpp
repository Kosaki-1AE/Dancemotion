#ifndef GBMC_HPP
#define GBMC_HPP

#include <iostream>
#include <string>
#include <vector>
#include <utility>

namespace gbmc {

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

    void add_input(const Port& port) {
        inputs.push_back(port);
    }

    void add_output(const Port& port) {
        outputs.push_back(port);
    }
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
        result.responsibility_compatible =
            from->responsibility.overlaps(to->responsibility);

        result.resonant =
            result.port_compatible && result.responsibility_compatible;

        if (result.resonant) {
            result.message = "Resonance: connection can flow.";
        } else if (!result.port_compatible) {
            result.message = "Meaning wall: port types are not compatible.";
        } else {
            result.message = "Meaning wall: responsibility ranges do not overlap.";
        }

        return result;
    }
};

struct FlowResult {
    bool complete = true;
    std::vector<ResonanceResult> steps;
};

class Flow {
public:
    explicit Flow(std::string n) : name_(std::move(n)) {}

    void add(const Connection& connection) {
        connections_.push_back(connection);
    }

    FlowResult evaluate() const {
        FlowResult result;

        for (const auto& connection : connections_) {
            auto step = connection.evaluate();
            if (!step.resonant) {
                result.complete = false;
            }
            result.steps.push_back(step);
        }

        return result;
    }

    void print() const {
        std::cout << "=== GBMC Flow: " << name_ << " ===\n";

        auto result = evaluate();

        for (std::size_t i = 0; i < connections_.size(); ++i) {
            const auto& c = connections_[i];
            const auto& r = result.steps[i];

            std::cout << "\nStep " << i + 1 << "\n";
            std::cout << "  " << c.from->name << "." << c.out_port.name
                      << " -> "
                      << c.to->name << "." << c.in_port.name << "\n";
            std::cout << "  Port: "
                      << (r.port_compatible ? "OK" : "NG") << "\n";
            std::cout << "  Responsibility: "
                      << (r.responsibility_compatible ? "OK" : "NG") << "\n";
            std::cout << "  State: "
                      << (r.resonant ? "RESONANT" : "MEANING_WALL") << "\n";
            std::cout << "  Message: " << r.message << "\n";
        }

        std::cout << "\nFlow Result: "
                  << (result.complete ? "COMPLETE" : "BLOCKED") << "\n";
    }

private:
    std::string name_;
    std::vector<Connection> connections_;
};

inline void print_module(const Module& m) {
    std::cout << "Module: " << m.name << "\n";
    std::cout << "  Represent: " << m.represent << "\n";
    std::cout << "  ResponsibilityRange: ["
              << m.responsibility.min_value << ", "
              << m.responsibility.max_value << "]\n";
}

} // namespace gbmc

#endif // GBMC_HPP
