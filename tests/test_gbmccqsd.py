import unittest

from gbmccqsd import GBMCCQSDMachine, GBMCCQSDState, RoutingMode, TableRouter, classify_mode


class StateTests(unittest.TestCase):
    def test_byte_round_trip(self):
        state = GBMCCQSDState.from_byte(0xAB)
        self.assertEqual(state.room, 0xA)
        self.assertEqual(state.responsibility, 0b10)
        self.assertEqual(state.operation, 0b11)
        self.assertEqual(state.byte, 0xAB)
        self.assertEqual(state.bits, "10101011")

    def test_operation_labels(self):
        self.assertEqual(GBMCCQSDState.from_parts(0, 0x0).operation_label, "打つ")
        self.assertEqual(GBMCCQSDState.from_parts(0, 0x1).operation_label, "弾く")
        self.assertEqual(GBMCCQSDState.from_parts(0, 0x2).operation_label, "外す")
        self.assertEqual(GBMCCQSDState.from_parts(0, 0x3).operation_label, "置く")


class ModeTests(unittest.TestCase):
    def test_all_lower_nibbles(self):
        expected = {
            0x0: RoutingMode.OBSERVE,
            0x1: RoutingMode.OBSERVE,
            0x2: RoutingMode.STRAIGHT,
            0x3: RoutingMode.STRAIGHT,
            0x4: RoutingMode.OBSERVE,
            0x5: RoutingMode.OBSERVE,
            0x6: RoutingMode.STRAIGHT,
            0x7: RoutingMode.STRAIGHT,
            0x8: RoutingMode.JAB,
            0x9: RoutingMode.JAB,
            0xA: RoutingMode.TWIST,
            0xB: RoutingMode.TWIST,
            0xC: RoutingMode.JAB,
            0xD: RoutingMode.JAB,
            0xE: RoutingMode.TWIST,
            0xF: RoutingMode.TWIST,
        }
        for lower, mode in expected.items():
            state = GBMCCQSDState.from_parts(0, lower)
            self.assertEqual(classify_mode(state.responsibility, state.operation), mode)


class MachineTests(unittest.TestCase):
    def test_missing_route_is_conservative(self):
        machine = GBMCCQSDMachine(initial_room=3)
        record = machine.step(0xA)
        self.assertEqual(record.mode, RoutingMode.TWIST)
        self.assertEqual(record.after.room, 3)
        self.assertFalse(record.routing_rule_hit)

    def test_table_route(self):
        router = TableRouter({(3, RoutingMode.TWIST): 9})
        machine = GBMCCQSDMachine(initial_room=3, router=router)
        record = machine.step(0xA)
        self.assertEqual(record.after.room, 9)
        self.assertTrue(record.routing_rule_hit)


if __name__ == "__main__":
    unittest.main()
