import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from intermediary_matrices import IntermediaryMatrices, unify_read

import logging

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)


class IntermediateMatricesTest(unittest.TestCase):
    def test_constructor(self):
        mats = IntermediaryMatrices({2: 1}, 5, 3)
        self.assertDictEqual(mats.ins_map, {2: 1})
        self.assertEqual(mats.mutation_pos, 5)
        self.assertEqual(mats.mutation_length, 3)
        self.assertListEqual(mats.nuc, [])
        self.assertListEqual(mats.bse, [])
        self.assertListEqual(mats.map, [])
        self.assertListEqual(mats.str, [])
        self.assertListEqual(mats.xtt, [])
        self.assertListEqual(mats.nmt, [])
        self.assertListEqual(mats.flg, [])

    def test_append_read(self):
        nuc_lst = ['*', '*', '-', '*', 'A', 'A', 'C']
        baseq_lst = [0, 0, 0, 0, 30, 40, 20]
        mats = IntermediaryMatrices({2: 1}, 5, 3)
        mats.append_read(nuc_lst, baseq_lst, 50, True, 'U', 5, 60, '', 60, 147)
        self.assertListEqual(
            mats.nuc,
            [['*', '*', '*', '*', 'A', 'A', 'C']]
        )
        self.assertListEqual(
            mats.bse,
            [[0, 0, 0, 0, 30, 40, 20]]
        )
        self.assertListEqual(mats.map, [50])
        self.assertListEqual(mats.str, [True])
        self.assertListEqual(mats.xtt, ['U'])
        self.assertListEqual(mats.nmt, [5])
        self.assertListEqual(mats.flg, [147])

        nuc_lst = ['*', '*', '-', 'A', 'A', '-', 'C']
        baseq_lst = [0, 0, 0, 20, 40, 0, 20]
        mats.append_read(nuc_lst, baseq_lst, 40, False, 'M', 0, 60, '', 60, 99)
        self.assertListEqual(
            mats.nuc,
            [
                ['*', '*', '*', '*', 'A', 'A', 'C'],
                ['*', '*', '*', 'A', 'A', '-', 'C']
            ]
        )
        self.assertListEqual(
            mats.bse,
            [
                [0, 0, 0, 0, 30, 40, 20],
                [0, 0, 0, 20, 40, 0, 20],
            ]
        )
        self.assertListEqual(mats.map, [50, 40])
        self.assertListEqual(mats.str, [True, False])
        self.assertListEqual(mats.xtt, ['U', 'M'])
        self.assertListEqual(mats.nmt, [5, 0])
        self.assertListEqual(mats.flg, [147, 99])

    def test_complete_rows(self):
        mats = IntermediaryMatrices({2: 1}, 5, 3)
        mats.nuc = [['A', 'A'], ['A', 'A', 'A']]
        mats.bse = [[20, 30], [20, 30, 10]]
        mats.map = [40, 45]
        mats.str = [True, True]
        mats.xtt = ['U', 'U']
        mats.nmt = [0, 0]
        mats.flg = [147, 147]

        mats.complete_rows(3)

        self.assertListEqual(mats.nuc, [['A', 'A', '*'], ['A', 'A', 'A']])
        self.assertListEqual(mats.bse, [[20, 30, 0], [20, 30, 10]])

    def test_get_transpose(self):
        mats = IntermediaryMatrices({2: 1}, 5, 3)
        mats.nuc = [['A', 'A', 'C'], ['A', 'A', 'A']]
        mats.bse = [[20, 30, 30], [20, 30, 10]]

        nuc_t, bse_t = mats.get_transpose()

        self.assertListEqual(
            [['A', 'A'], ['A', 'A'], ['C', 'A']],
            nuc_t
        )

        self.assertListEqual(
            [[20, 20], [30, 30], [30, 10]],
            bse_t
        )

    def test_is_empty(self):
        mats = IntermediaryMatrices({2: 1}, 5, 3)
        self.assertEqual(mats.is_empty(), True)
        mats.nuc = [['A', 'A'], ['A', 'A', 'A']]
        self.assertEqual(mats.is_empty(), False)

    def test_fill_with_dummy(self):
        mats = IntermediaryMatrices({2: 1}, 5, 3)
        mats.fill_with_dummy(2)
        self.assertEqual(mats.is_empty(), False)

    def test_get_mutation_range(self):
        mats = IntermediaryMatrices({2: 1}, 5, 3)
        rng = mats.get_mutation_range(7, 'ins')
        self.assertEqual(rng, range(6, 9))
        rng = mats.get_mutation_range(3, 'ins')
        self.assertEqual(rng, range(6, 7))

    def test_scale_matrices_horizontal(self):
        mats = IntermediaryMatrices({2: 1}, 1, 0)
        mats.nuc = [['A', 'C'], ['A', 'C', 'A']]
        mats.bse = [[20, 30], [20, 30, 10]]
        mats.map = [40, 45]
        mats.str = [True, True]
        mats.xtt = ['U', 'U']
        mats.nmt = [0, 0]
        mats.flg = [147, 147]

        mats.complete_rows(15)

        mats.nuc = np.array(mats.nuc)
        mats.bse = np.array(mats.bse)

        ref_seq = ['A', 'C', 'A']
        ref_seq = mats.scale_matrices_horizontal(ref_seq, 2, 1)

        np.testing.assert_array_equal(ref_seq, ['C', 'A'])
        np.testing.assert_array_equal(mats.nuc, [['C', '*'], ['C', 'A']])
        np.testing.assert_array_equal(mats.bse, [[30, 0], [30, 10]])

    def test_scale_matrices_horizontal_2(self):
        mats = IntermediaryMatrices({2: 1}, 1, 0)
        mats.nuc = [['A', 'C'], ['A', 'C', 'A']]
        mats.bse = [[20, 30], [20, 30, 10]]
        mats.map = [40, 45]
        mats.str = [True, True]
        mats.xtt = ['U', 'U']
        mats.nmt = [0, 0]
        mats.flg = [147, 147]

        mats.complete_rows(15)

        mats.nuc = np.array(mats.nuc)
        mats.bse = np.array(mats.bse)

        ref_seq = ['A', 'C', 'A']
        ref_seq = mats.scale_matrices_horizontal(ref_seq, 7, 1)

        np.testing.assert_array_equal(ref_seq,
                                      ['*', '*', 'A', 'C', 'A', '*', '*'])
        np.testing.assert_array_equal(
            mats.nuc,
            [['*', '*', 'A', 'C', '*', '*', '*'],
             ['*', '*', 'A', 'C', 'A', '*', '*']]
        )
        np.testing.assert_array_equal(
            mats.bse,
            [[0, 0, 20, 30, 0, 0, 0], [0, 0, 20, 30, 10, 0, 0]]
        )

    def test_calculate_padding_pattern(self):
        self.assertTupleEqual(
            IntermediaryMatrices.calculate_padding_pattern(7, 3),
            ((0, 0), (2, 2))
        )
        self.assertTupleEqual(
            IntermediaryMatrices.calculate_padding_pattern(7, 6),
            ((0, 0), (0, 1))
        )
        self.assertTupleEqual(
            IntermediaryMatrices.calculate_padding_pattern(7, 7),
            ((0, 0), (0, 0))
        )
        self.assertTupleEqual(
            IntermediaryMatrices.calculate_padding_pattern(6, 5),
            ((0, 0), (0, 1))
        )

    def test_calculate_start_end(self):
        self.assertTupleEqual(
            IntermediaryMatrices.calculate_start_end(7, 9),
            (1, 8)
        )
        self.assertTupleEqual(
            IntermediaryMatrices.calculate_start_end(6, 9),
            (2, 8)
        )
        self.assertTupleEqual(
            IntermediaryMatrices.calculate_start_end(7, 7),
            (0, 7)
        )
        self.assertTupleEqual(
            IntermediaryMatrices.calculate_start_end(8, 9),
            (1, 9)
        )

    def test_sample(self):
        mats = IntermediaryMatrices({2: 1}, 5, 3)
        mats.nuc = np.array(
            [['A', 'A', 'C'], ['A', 'A', 'A'], ['*', 'A', 'C']])
        mats.bse = np.array([[20, 30, 30], [20, 30, 10], [0, 30, 10]])
        mats.map = [20, 30, 10]
        mats.str = [True, True, False]
        mats.xtt = ['U', 'U', 'M']
        mats.nmt = [0, 1, 2]
        mats.ast = [0, 1, 2]
        mats.mdt = ['U', 'U', 'M']
        mats.xst = [0, 1, 2]
        mats.flg = [147, 99, 147]

        mats = IntermediaryMatrices.sample(mats, np.array([0, 2]))
        np.testing.assert_array_equal(
            mats.nuc,
            [['A', 'A', 'C'], ['*', 'A', 'C']]
        )
        np.testing.assert_array_equal(
            mats.bse,
            [[20, 30, 30], [0, 30, 10]]
        )
        np.testing.assert_array_equal(mats.map, [20, 10])
        np.testing.assert_array_equal(mats.str, [True, False])
        np.testing.assert_array_equal(mats.xtt, ['U', 'M'])
        np.testing.assert_array_equal(mats.nmt, [0, 2])
        np.testing.assert_array_equal(mats.flg, [147, 147])

    def test_unify_read(self):
        self.assertListEqual(
            unify_read(['*', '*', '-', '*', 'A', 'A', 'C']),
            ['*', '*', '*', '*', 'A', 'A', 'C']
        )
        self.assertListEqual(
            unify_read(['*', '*', '-', 'A', 'A', '-', 'C']),
            ['*', '*', '*', 'A', 'A', '-', 'C']
        )
        self.assertListEqual(
            unify_read(['*', '-', '-', '*', 'A', '-', 'C']),
            ['*', '*', '*', '*', 'A', '-', 'C']
        )
        self.assertListEqual(
            unify_read(['*', '-', '-', 'A', 'A', '-', 'C']),
            ['*', '*', '*', 'A', 'A', '-', 'C']
        )
        self.assertListEqual(
            unify_read(['*', '-', '-', 'A', 'A', '-', '*']),
            ['*', '*', '*', 'A', 'A', '*', '*']
        )


if __name__ == '__main__':
    unittest.main()
