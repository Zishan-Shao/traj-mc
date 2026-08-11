import unittest

from utils.abmp import allocate_block_orders


class ABMPAllocationTest(unittest.TestCase):
    def test_llada_ratio_preserves_two_bit_average(self):
        orders = allocate_block_orders(range(32), base_order=2, ratio=0.05)
        self.assertEqual(orders.count(1), 1)
        self.assertEqual(orders.count(3), 1)
        self.assertEqual(sum(orders) / len(orders), 2.0)

    def test_dream_ratio_preserves_two_bit_average(self):
        orders = allocate_block_orders(range(28), base_order=2, ratio=0.10)
        self.assertEqual(orders.count(1), 2)
        self.assertEqual(orders.count(3), 2)
        self.assertEqual(sum(orders) / len(orders), 2.0)

    def test_zero_ratio_keeps_uniform_order(self):
        orders = allocate_block_orders(range(4), base_order=2, ratio=0.0)
        self.assertEqual(orders, [2, 2, 2, 2])

    def test_invalid_ratio_is_rejected(self):
        with self.assertRaises(ValueError):
            allocate_block_orders(range(32), base_order=2, ratio=0.6)


if __name__ == "__main__":
    unittest.main()
