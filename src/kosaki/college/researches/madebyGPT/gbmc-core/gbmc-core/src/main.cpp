#include "gbmc.hpp"

int main() {
    using namespace gbmc;

    Module pico("Pico2W", "pilot-intention", ResponsibilityRange(0.0, 60.0));
    pico.add_input(Port("joystick", "angle"));
    pico.add_output(Port("control_candidate", "angle"));

    Module esp32("ESP32", "execution-filter", ResponsibilityRange(0.0, 45.0));
    esp32.add_input(Port("candidate_in", "angle"));
    esp32.add_output(Port("safe_pwm", "pwm"));

    Module servo("Servo", "physical-output", ResponsibilityRange(0.0, 30.0));
    servo.add_input(Port("pwm_in", "pwm"));
    servo.add_output(Port("surface_angle", "angle"));

    print_module(pico);
    print_module(esp32);
    print_module(servo);

    Connection c1(pico, esp32, Port("control_candidate", "angle"), Port("candidate_in", "angle"));
    Connection c2(esp32, servo, Port("safe_pwm", "pwm"), Port("pwm_in", "pwm"));
    Connection c3(pico, servo, Port("control_candidate", "angle"), Port("pwm_in", "pwm"));

    print_connection(c1);
    print_connection(c2);
    print_connection(c3);

    return 0;
}
