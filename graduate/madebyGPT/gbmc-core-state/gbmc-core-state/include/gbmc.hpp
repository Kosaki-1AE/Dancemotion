#ifndef GBMC_HPP
#define GBMC_HPP

#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace gbmc {

struct Signal {
    std::string type;
    double value = 0.0;

    Signal() = default;
    Signal(std::string t, double v)
        : type(std::move(t)), value(v) {}
};

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

    double clamp(double value) const {
        return std::max(min_value, std::min(max_value, value));
    }
};

struct TransitionResult {
    Signal input;
    Signal output;
    bool clipped = false;
    std::string message;
};

class Module {
public:
    using Transform = std::function<Signal(const Signal&)>;

    Module() = default;

    Module(std::string n, std::string rep, ResponsibilityRange r, Transform fn)
        : name(std::move(n)),
          represent(std::move(rep)),
          responsibility(r),
          transform(std::move(fn)) {}

    void add_input(const Port& port) {
        inputs.push_back(port);
    }

    void add_output(const Port& port) {
        outputs.push_back(port);
    }

    TransitionResult run(const Signal& signal) const {
        TransitionResult result;
        result.input = signal;

        Signal raw = transform ? transform(signal) : signal;
        result.output = raw;

        if (!responsibility.contains(raw.value)) {
            result.clipped = true;
            result.output.value = responsibility.clamp(raw.value);
            result.message = "Responsibility boundary: output was clipped.";
        } else {
            result.message = "Within responsibility range.";
        }

        return result;
    }

    std::string name;
    std::string represent;
    std::vector<Port> inputs;
    std::vector<Port> outputs;
    ResponsibilityRange responsibility;
    Transform transform;
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

class Flow {
public:
    explicit Flow(std::string n) : name_(std::move(n)) {}

    void add(const Connection& connection) {
        connections_.push_back(connection);
    }

    bool can_flow() const {
        for (const auto& c : connections_) {
            if (!c.evaluate().resonant) {
                return false;
            }
        }
        return true;
    }

    Signal run(Signal signal) const {
        std::cout << "=== GBMC State Flow: " << name_ << " ===\n";
        std::cout << "Initial Signal: " << signal.type << " = " << signal.value << "\n";

        for (std::size_t i = 0; i < connections_.size(); ++i) {
            const auto& c = connections_[i];
            const auto resonance = c.evaluate();

            std::cout << "\nStep " << i + 1 << ": "
                      << c.from->name << " -> " << c.to->name << "\n";

            if (!resonance.resonant) {
                std::cout << "  BLOCKED: " << resonance.message << "\n";
                return signal;
            }

            auto from_result = c.from->run(signal);

            std::cout << "  " << c.from->name
                      << " input  : " << from_result.input.type
                      << " = " << from_result.input.value << "\n";
            std::cout << "  " << c.from->name
                      << " output : " << from_result.output.type
                      << " = " << from_result.output.value << "\n";
            if (from_result.clipped) {
                std::cout << "  Note   : " << from_result.message << "\n";
            }

            signal = from_result.output;
            signal.type = c.in_port.type;
        }

        if (!connections_.empty()) {
            const auto& last = connections_.back();
            auto final_result = last.to->run(signal);

            std::cout << "\nFinal Module: " << last.to->name << "\n";
            std::cout << "  input  : " << final_result.input.type
                      << " = " << final_result.input.value << "\n";
            std::cout << "  output : " << final_result.output.type
                      << " = " << final_result.output.value << "\n";
            if (final_result.clipped) {
                std::cout << "  Note   : " << final_result.message << "\n";
            }

            signal = final_result.output;
        }

        std::cout << "\nFinal Signal: " << signal.type << " = " << signal.value << "\n";
        return signal;
    }

private:
    std::string name_;
    std::vector<Connection> connections_;
};

} // namespace gbmc

#endif // GBMC_HPP
