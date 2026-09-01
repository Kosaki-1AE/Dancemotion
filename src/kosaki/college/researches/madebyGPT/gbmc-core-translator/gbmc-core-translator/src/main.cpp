#include "gbmc.hpp"

int main() {
    using namespace gbmc;

    Module joystick(
        "Joystick",
        "pilot-intention",
        ResponsibilityRange(-90.0, 90.0),
        [](const Signal& s) {
            return Signal("angle", s.value);
        }
    );
    joystick.add_output(Port("angle_out", "angle"));

    Module pico(
        "Pico2W",
        "candidate-generator",
        ResponsibilityRange(-45.0, 45.0),
        [](const Signal& s) {
            return Signal("angle", s.value);
        }
    );
    pico.add_input(Port("angle_in", "angle"));
    pico.add_output(Port("candidate_out", "angle"));

    Translator angle_to_pwm(
        "AngleToPWM",
        "translator",
        "angle",
        "pwm",
        ResponsibilityRange(1200.0, 1800.0),
        0.95,
        [](const Signal& s) {
            return Signal("pwm", 1500.0 + s.value * 10.0);
        }
    );

    Module servo(
        "Servo",
        "physical-output",
        ResponsibilityRange(1250.0, 1750.0),
        [](const Signal& s) {
            return Signal("pwm", s.value);
        }
    );
    servo.add_input(Port("pwm_in", "pwm"));
    servo.add_output(Port("surface_out", "pwm"));

    Flow control("human-powered-aircraft-translator-control");

    control.add(Connection(
        joystick, pico,
        Port("angle_out", "angle"),
        Port("angle_in", "angle")
    ));

    control.add(Connection(
        pico, angle_to_pwm,
        Port("candidate_out", "angle"),
        Port("in", "angle")
    ));

    control.add(Connection(
        angle_to_pwm, servo,
        Port("out", "pwm"),
        Port("pwm_in", "pwm")
    ));

    std::cout << "\n--- Case 1: safe input ---\n";
    control.run(Signal("angle", 20.0), true);

    std::cout << "\n\n--- Case 2: excessive input ---\n";
    control.run(Signal("angle", 80.0), true);

    std::cout << "\n\n";
    ConceptTranslator concepts;

    concepts.add({
        "Transformer",
        "Attention",
        "Port",
        0.72,
        "Attention mediates which information can connect to the next representation."
    });

    concepts.add({
        "ControlTheory",
        "Observer",
        "Translator",
        0.68,
        "An observer translates measured state into an internal estimate."
    });

    concepts.add({
        "StreetDance",
        "Groove",
        "Resonance",
        0.81,
        "Groove emerges when body, rhythm, and space transition together."
    });

    concepts.print_find("Attention");
    concepts.print_find("Groove");

    return 0;
}
