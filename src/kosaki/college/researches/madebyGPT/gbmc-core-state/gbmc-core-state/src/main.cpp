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
        ResponsibilityRange(-30.0, 30.0),
        [](const Signal& s) {
            // angle candidate -> PWM-like command
            return Signal("pwm", 1500.0 + s.value * 10.0);
        }
    );
    esp32.add_input(Port("candidate_in", "angle"));
    esp32.add_output(Port("pwm_out", "pwm"));

    Module servo(
        "Servo",
        "physical-output",
        ResponsibilityRange(1200.0, 1800.0),
        [](const Signal& s) {
            // PWM is physically accepted by the servo.
            return Signal("pwm", s.value);
        }
    );
    servo.add_input(Port("pwm_in", "pwm"));
    servo.add_output(Port("surface_out", "pwm"));

    Flow control("human-powered-aircraft-state-transition");

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

    std::cout << "\n--- Case 1: safe input ---\n";
    control.run(Signal("angle", 20.0));

    std::cout << "\n\n--- Case 2: excessive input ---\n";
    control.run(Signal("angle", 80.0));

    return 0;
}
