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

    Module esp32(
        "ESP32",
        "execution-filter",
        ResponsibilityRange(1200.0, 1800.0),
        [](const Signal& s) {
            return Signal("pwm", 1500.0 + s.value * 10.0);
        }
    );
    esp32.add_input(Port("candidate_in", "angle"));
    esp32.add_output(Port("pwm_out", "pwm"));

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

    Flow control("human-powered-aircraft-final-control");

    control.add(Connection(
        joystick, pico,
        Port("angle_out", "angle"),
        Port("angle_in", "angle")
    ));

    control.add(Connection(
        pico, esp32,
        Port("candidate_out", "angle"),
        Port("candidate_in", "angle")
    ));

    control.add(Connection(
        esp32, servo,
        Port("pwm_out", "pwm"),
        Port("pwm_in", "pwm")
    ));

    std::vector<Candidate> candidates = {
        {"soft-left", Signal("angle", -15.0)},
        {"center", Signal("angle", 0.0)},
        {"soft-right", Signal("angle", 20.0)},
        {"hard-right", Signal("angle", 60.0)},
        {"panic-right", Signal("angle", 85.0)}
    };

    Planner planner(control);
    planner.print_choice(candidates);

    auto best = planner.choose(candidates);

    std::cout << "\n\n=== Execute Chosen Candidate ===\n";
    control.run(best.candidate.signal, true);

    return 0;
}
