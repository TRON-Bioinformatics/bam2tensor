import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import pysam
from aligner_utils import *
from regions import get_candidates, compute_regions
from intermediary_matrices import IntermediaryMatrices
import test_constants

import logging

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)


class AlignerUtilsTest(unittest.TestCase):
    def test_get_sequence_and_baseq(self):
        read = pysam.AlignedSegment()
        read.reference_start = 20
        read.query_name = 'read1'
        read.mapping_quality = 20
        read.query_sequence = 'AAAAATGGGCAAAANG'
        read.query_qualities = [30] * 16
        read.cigarstring = '4S10M2S'
        # read.is_reverse = True #TODO: no effect here, why?
        seq, bqs = get_sequence_and_baseq(read)
        self.assertSequenceEqual(seq, 'AAAAATGGGCAAAA-G')
        self.assertListEqual(bqs, [30] * 16)

    def test_get_reference_seq(self):
        candidates = get_candidates(test_constants.CANDIDATES)
        regions = compute_regions(candidates, window=7)
        ref_path = test_constants.REFERENCE
        seq = get_reference_seq(ref_path, regions[0], window=7)
        self.assertSequenceEqual(
            ['T', 'G', 'G', 'G', 'C', 'A', 'G', 'C', 'G', 'G', 'T', 'A', 'A',
             'T', 'C'],
            seq
        )
        seq = get_reference_seq(ref_path, regions[1], window=7)
        self.assertSequenceEqual(
            ['A', 'G', 'C', 'G', 'G', 'T', 'A', 'A', 'T', 'C', 'N', 'N', 'N',
             'N'],
            seq
        )

    def test_get_soft_clipping_info(self):
        read = pysam.AlignedSegment()
        read.reference_start = 20
        read.query_name = 'read1'
        read.mapping_quality = 20
        read.query_sequence = 'AAAAATAAAACAAAAT'
        read.query_qualities = [30] * 16
        read.cigarstring = '4S10M2S'

        pos, len_at_start = get_soft_clipping_info(read)

        self.assertListEqual(pos, [0, 1, 2, 3, 14, 15])
        self.assertEqual(len_at_start, 4)

    def test_extend_alignment_matrices(self):
        alt = IntermediaryMatrices(ins_map={}, mutation_pos=5,
                                   mutation_length=0)
        alt.nuc = np.array([
            ['A', 'C', 'A', 'C', 'A', 'C', 'C', 'C', 'C', 'C'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-', 'C'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-', 'C'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-', 'C'],
            ['A', 'G', 'A', 'G', 'A', '-', '-', 'G', '-', 'G'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-', 'C'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-', 'C'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-', 'C']
        ])
        alt.bse = np.array([
            [50, 40, 50, 40, 50, 40, 40, 40, 40, 40],
            [50, 40, 50, 40, 50, 40, 0, 40, 0, 40],
            [50, 40, 50, 40, 50, 40, 0, 40, 0, 40],
            [50, 40, 50, 40, 50, 40, 0, 40, 0, 40],
            [50, 45, 50, 45, 50, 0, 0, 45, 0, 45],
            [50, 40, 50, 40, 50, 40, 0, 40, 0, 40],
            [50, 40, 50, 40, 50, 40, 0, 40, 0, 40],
            [50, 40, 50, 40, 50, 40, 0, 40, 0, 40]
        ])
        aln = IntermediaryMatrices(ins_map={}, mutation_pos=5,
                                   mutation_length=0)
        aln.nuc = np.array([
            ['A', 'C', 'A', 'C', 'A', 'C', 'C', 'C', 'C'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-'],
            ['A', 'G', 'A', 'G', 'A', '-', '-', 'G', '-'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-']
        ])
        aln.bse = np.array([
            [50, 40, 50, 40, 50, 40, 40, 40, 40],
            [50, 40, 50, 40, 50, 40, 0, 40, 0],
            [50, 40, 50, 40, 50, 40, 0, 40, 0],
            [50, 40, 50, 40, 50, 40, 0, 40, 0],
            [50, 45, 50, 45, 50, 0, 0, 45, 0],
            [50, 40, 50, 40, 50, 40, 0, 40, 0],
            [50, 40, 50, 40, 50, 40, 0, 40, 0],
            [50, 40, 50, 40, 50, 40, 0, 40, 0]
        ])
        extend_alignment_matrices(alt, aln, [''])  # TODO: test ref as well
        self.assertEqual(alt.shape_of_one()[1], aln.shape_of_one()[1])

    def test_get_ins_pos_to_length_mapping(self):
        candidates = get_candidates(test_constants.CANDIDATES)
        regions = compute_regions(candidates, window=7)
        expected = {11: 3, 9: 1, 18: 1, 20: 3}

        sam_file = pysam.AlignmentFile(test_constants.TUMOR_BAM, 'rb')
        output = get_ins_pos_to_length_mapping(sam_file, regions[0])

        self.assertDictEqual(expected, output)

    def test_compute_single_insertion_mappings(self):
        candidates = get_candidates(test_constants.CANDIDATES)
        regions = compute_regions(candidates, window=7)

        expected = [
            {}, {}, {11: 1}, {11: 1}, {11: 3}, {9: 1}, {18: 1}, {}, {18: 1},
            {}, {}, {11: 3, 18: 1},
            {20: 3}
        ]

        sam_file = pysam.AlignmentFile(test_constants.TUMOR_BAM, 'rb')
        output = compute_single_insertion_mappings(sam_file, regions[0])

        self.assertListEqual(expected, output)

    def test_merge_insertion_mappings(self):
        all_ins_maps = [
            {1: 1, 6: 2},
            {1: 1},
            {6: 1},
            {1: 3, 5: 1, 6: 1},
            {13: 1, 6: 2}
        ]
        expected = {1: 3, 5: 1, 6: 2, 13: 1}
        output = merge_insertion_mappings(all_ins_maps)
        self.assertDictEqual(expected, output)

    def test_get_read_indexed_ins_map(self):
        candidates = get_candidates(test_constants.CANDIDATES)
        region = compute_regions(candidates, window=7)[0]
        region.start = 1257323
        region.end = 1257353

        ins_map = {1257333: 2, 1257346: 1}

        expected = {11: 2, 24: 1}

        output = get_read_indexed_ins_map(ins_map, region)

        self.assertDictEqual(expected, output)

    def test_get_different_insertions(self):
        ins_map1 = {1: 2, 3: 1, 5: 1}
        ins_map2 = {1: 1, 5: 1}

        expected = {1: 1, 3: 1}

        output = get_different_insertions(ins_map1, ins_map2)

        self.assertDictEqual(expected, output)

    def test_compute_added_no_1(self):
        ins_map = {1: 3, 4: 2, 7: 1}
        self_ins_pos = [1, 4, 7]
        ins_pos = 1
        added_no = compute_added_no(ins_map, self_ins_pos, ins_pos)

        self.assertEqual(added_no, 3)
        np.testing.assert_array_equal(self_ins_pos, [4, 7])

    def test_compute_added_no_2(self):
        ins_map = {1: 3, 4: 2, 7: 1}
        self_ins_pos = [4, 7]
        ins_pos = 4
        added_no = compute_added_no(ins_map, self_ins_pos, ins_pos)

        self.assertEqual(added_no, 2)
        np.testing.assert_array_equal(self_ins_pos, [7])

    def test_add_default_cols(self):
        cols = [
            ['A', 'A', 'A', 'A'],
            ['A', 'A', 'A', 'A'],
            ['A', 'A', 'A', 'A'],
            ['A', 'A', 'A', 'A'],
        ]
        ins_start_pos = 2
        ins_diff = 3
        default_col = ['-'] * 4

        expected = [
            ['A', 'A', 'A', 'A'],
            ['A', 'A', 'A', 'A'],
            ['-', '-', '-', '-'],
            ['-', '-', '-', '-'],
            ['-', '-', '-', '-'],
            ['A', 'A', 'A', 'A'],
            ['A', 'A', 'A', 'A'],
        ]

        output = add_default_cols(cols, ins_start_pos, ins_diff, default_col)

        np.testing.assert_array_equal(expected, output)


if __name__ == '__main__':
    unittest.main()
