#ifndef GBMC_HPP
#define GBMC_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
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

    double violation(double value) const {
        if (value < min_value) return min_value - value;
        if (value > max_value) return value - max_value;
        return 0.0;
    }
};

struct TransitionResult {
    Signal input;
    Signal raw_output;
    Signal output;
    bool clipped = false;
    double violation = 0.0;
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
        result.raw_output = raw;
        result.output = raw;

        result.violation = responsibility.violation(raw.value);

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

struct Translation {
    std::string from_type;
    std::string to_type;
    double confidence = 1.0;
    std::string note;
};

class Translator : public Module {
public:
    using TranslateFn = std::function<Signal(const Signal&)>;

    Translator(
        std::string n,
        std::string rep,
        std::string from_t,
        std::string to_t,
        ResponsibilityRange r,
        double conf,
        TranslateFn fn
    )
        : Module(
            n,
            rep,
            r,
            [fn](const Signal& s) {
                return fn(s);
            }
          ),
          translation{std::move(from_t), std::move(to_t), conf, ""}
    {
        add_input(Port("in", translation.from_type));
        add_output(Port("out", translation.to_type));
    }

    Translation translation;
};

struct ConceptMapping {
    std::string source_domain;
    std::string source_concept;
    std::string gbmc_concept;
    double confidence = 0.0;
    std::string reason;
};

class ConceptTranslator {
public:
    void add(const ConceptMapping& mapping) {
        mappings_.push_back(mapping);
    }

    std::vector<ConceptMapping> find(const std::string& source_concept) const {
        std::vector<ConceptMapping> result;
        for (const auto& m : mappings_) {
            if (m.source_concept == source_concept) {
                result.push_back(m);
            }
        }

        std::sort(result.begin(), result.end(),
            [](const ConceptMapping& a, const ConceptMapping& b) {
                return a.confidence > b.confidence;
            });

        return result;
    }

    void print_find(const std::string& source_concept) const {
        auto result = find(source_concept);

        std::cout << "=== Concept Translator ===\n";
        std::cout << "Query: " << source_concept << "\n";

        if (result.empty()) {
            std::cout << "No mapping found.\n";
            return;
        }

        for (const auto& m : result) {
            std::cout << "\n" << m.source_domain << "." << m.source_concept
                      << " -> GBMC." << m.gbmc_concept << "\n";
            std::cout << "  confidence: " << m.confidence << "\n";
            std::cout << "  reason    : " << m.reason << "\n";
        }
    }

private:
    std::vector<ConceptMapping> mappings_;
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

        // Responsibility compatibility is meaningful only when the signal type
        // is the same. If a Translator changes the type, the next module's
        // responsibility is checked when the translated signal arrives.
        result.responsibility_compatible =
            result.port_compatible ? from->responsibility.overlaps(to->responsibility) : false;

        result.resonant =
            result.port_compatible &&
            (result.responsibility_compatible || out_port.type != "numeric-range-strict");

        if (result.port_compatible) {
            result.resonant = true;
            result.message = "Resonance: port types match. Responsibility will be checked at transition time.";
        } else {
            result.resonant = false;
            result.message = "Meaning wall: port types are not compatible. Translator required.";
        }

        return result;
    }
};

struct FlowRunResult {
    Signal final_signal;
    bool blocked = false;
    double total_violation = 0.0;
    int clipped_count = 0;
};

class Flow {
public:
    explicit Flow(std::string n) : name_(std::move(n)) {}

    const std::string& name() const {
        return name_;
    }

    void add(const Connection& connection) {
        connections_.push_back(connection);
    }

    FlowRunResult run(Signal signal, bool verbose = true) const {
        FlowRunResult result;
        result.final_signal = signal;

        if (verbose) {
            std::cout << "=== GBMC Translator Flow: " << name_ << " ===\n";
            std::cout << "Initial Signal: " << signal.type << " = " << signal.value << "\n";
        }

        for (std::size_t i = 0; i < connections_.size(); ++i) {
            const auto& c = connections_[i];
            const auto resonance = c.evaluate();

            if (verbose) {
                std::cout << "\nStep " << i + 1 << ": "
                          << c.from->name << " -> " << c.to->name << "\n";
            }

            if (!resonance.resonant) {
                result.blocked = true;
                if (verbose) {
                    std::cout << "  BLOCKED: " << resonance.message << "\n";
                }
                result.final_signal = signal;
                return result;
            }

            auto from_result = c.from->run(signal);

            result.total_violation += from_result.violation;
            if (from_result.clipped) {
                result.clipped_count++;
            }

            if (verbose) {
                std::cout << "  input      : " << from_result.input.type
                          << " = " << from_result.input.value << "\n";
                std::cout << "  raw output : " << from_result.raw_output.type
                          << " = " << from_result.raw_output.value << "\n";
                std::cout << "  output     : " << from_result.output.type
                          << " = " << from_result.output.value << "\n";
                std::cout << "  state      : "
                          << (from_result.clipped ? "CLIPPED" : "OK") << "\n";
            }

            signal = from_result.output;
            signal.type = c.in_port.type;
        }

        if (!connections_.empty()) {
            const auto& last = connections_.back();
            auto final_result = last.to->run(signal);

            result.total_violation += final_result.violation;
            if (final_result.clipped) {
                result.clipped_count++;
            }

            if (verbose) {
                std::cout << "\nFinal Module: " << last.to->name << "\n";
                std::cout << "  input      : " << final_result.input.type
                          << " = " << final_result.input.value << "\n";
                std::cout << "  raw output : " << final_result.raw_output.type
                          << " = " << final_result.raw_output.value << "\n";
                std::cout << "  output     : " << final_result.output.type
                          << " = " << final_result.output.value << "\n";
                std::cout << "  state      : "
                          << (final_result.clipped ? "CLIPPED" : "OK") << "\n";
            }

            signal = final_result.output;
        }

        result.final_signal = signal;

        if (verbose) {
            std::cout << "\nFinal Signal: " << signal.type << " = " << signal.value << "\n";
            std::cout << "Total violation: " << result.total_violation << "\n";
            std::cout << "Clipped count   : " << result.clipped_count << "\n";
        }

        return result;
    }

private:
    std::string name_;
    std::vector<Connection> connections_;
};

} // namespace gbmc

#endif // GBMC_HPP
