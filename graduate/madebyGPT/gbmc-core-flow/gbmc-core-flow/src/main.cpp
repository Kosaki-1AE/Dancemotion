#include "gbmc.hpp"

int main() {
    using namespace gbmc;

    Module joystick(
        "Joystick",
        "pilot-intention",
        ResponsibilityRange(-60.0, 60.0)
    );
    joystick.add_output(Port("angle_out", "angle"));

    Module pico(
        "Pico2W",
        "candidate-generator",
        ResponsibilityRange(-45.0, 45.0)
    );
    pico.add_input(Port("angle_in", "angle"));
    pico.add_output(Port("candidate_out", "angle"));

    Module esp32(
        "ESP32",
        "execution-filter",
        ResponsibilityRange(-30.0, 30.0)
    );
    esp32.add_input(Port("candidate_in", "angle"));
    esp32.add_output(Port("pwm_out", "pwm"));

    Module servo(
        "Servo",
        "physical-output",
        ResponsibilityRange(-25.0, 25.0)
    );
    servo.add_input(Port("pwm_in", "pwm"));
    servo.add_output(Port("surface_angle", "angle"));

    Flow safe_flow("human-powered-aircraft-safe-control");

    safe_flow.add(Connection(
        joystick,
        pico,
        Port("angle_out", "angle"),
        Port("angle_in", "angle")
    ));

    safe_flow.add(Connection(
        pico,
        esp32,
        Port("candidate_out", "angle"),
        Port("candidate_in", "angle")
    ));

    safe_flow.add(Connection(
        esp32,
        servo,
        Port("pwm_out", "pwm"),
        Port("pwm_in", "pwm")
    ));

    safe_flow.print();

    Flow broken_flow("direct-pico-to-servo-broken");

    broken_flow.add(Connection(
        pico,
        servo,
        Port("candidate_out", "angle"),
        Port("pwm_in", "pwm")
    ));

    std::cout << "\n\n";
    broken_flow.print();

    return 0;
}
