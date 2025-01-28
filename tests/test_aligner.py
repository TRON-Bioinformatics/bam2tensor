import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import numpy as np
import pysam
from aligner import *
from regions import compute_regions, get_candidates
import test_constants

import logging

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)


class AlignerTest(unittest.TestCase):
    def test_compute_alignment_1(self):
        """Test compute_alignment(): The entire alignment pipeline for region 1."""
        candidates = get_candidates(test_constants.CANDIDATES)
        regions = compute_regions(candidates, window=7)
        inter_matst, inter_matsn, ref_seq = compute_alignment(
            test_constants.TUMOR_BAM,
            test_constants.NORMAL_BAM,
            test_constants.REFERENCE,
            regions[0],
            window=7
        )
        al_tum = inter_matst.nuc
        al_nor = inter_matsn.nuc
        np.testing.assert_array_equal(al_tum, np.loadtxt(test_constants.AL_TUM1, dtype=str))
        np.testing.assert_array_equal(al_nor, np.loadtxt(test_constants.AL_NOR1, dtype=str))

    def test_compute_alignment_2(self):
        """Test compute_alignment(): The entire alignment pipeline for region 2."""
        candidates = get_candidates(test_constants.CANDIDATES)
        regions = compute_regions(candidates, window=7)
        inter_matst, inter_matsn, ref_seq = compute_alignment(
            test_constants.TUMOR_BAM,
            test_constants.NORMAL_BAM,
            test_constants.REFERENCE,
            regions[1],
            window=7
        )
        al_tum = inter_matst.nuc
        al_nor = inter_matsn.nuc
        np.testing.assert_array_equal(al_tum, np.loadtxt(test_constants.AL_TUM2, dtype=str))
        np.testing.assert_array_equal(al_nor, np.loadtxt(test_constants.AL_NOR2, dtype=str))


    def test_align_sequence_for_read(self):
        # expected outputs list
        expected_seqs = [
            'A---GTGGTAA-TC---',
            'A---GCGGTAA-TC---',
            'AA--GCGGTAA-TC---',
            'AC--GTGGTAA-TC---',
            'ACCCGTGGTAA-TC---',
            'A---GCGGTA--TC---',
            'A---GTGGTAAATC---',
            'A---GTGGTAA-TC---',
            'A---GTGGTAAATC---',
            'A---GTGGTAA--C---',
            'A---GTGGTA---C---',
            'ACCCGTGGTAAATC---',
            'A---GTGGTAA-TCAAA'
        ]

        candidates = get_candidates(test_constants.CANDIDATES)
        region = compute_regions(candidates, window=7)[1]
        sam_file = pysam.AlignmentFile(test_constants.TUMOR_BAM, 'rb')
        ins_map = get_ins_pos_to_length_mapping(sam_file, region)
        for i, read_record in enumerate(sam_file.fetch(region.chrom, region.start, region.end)):
            nuc_lst, bse_list, mapq, strand, xt_tag, nm_tag, _, _, _, flag = align_sequence_for_read(
                read_record,
                region,
                ins_map
            )
            assert_error = '{} not same as {}'.format(''.join(nuc_lst), expected_seqs[i])
            assert ''.join(nuc_lst) == expected_seqs[i], assert_error

    def test_start_read_1(self):
        candidates = get_candidates(test_constants.CANDIDATES)
        region = compute_regions(candidates, window=7)[0]
        region.start = 1
        region.end = 8
        seq = ['A', 'C', 'G', 'T', 'A', 'C', 'G', 'T']
        bqs = [22, 20, 30, 15, 22, 20, 30, 15]
        ref_start = 3
        entire_ins_map = {}
        soft_clipped_len_at_start = 0
        new_seq, new_bqs = start_read(
            seq,
            bqs,
            ref_start,
            region,
            entire_ins_map,
            soft_clipped_len_at_start
        )
        np.testing.assert_array_equal(new_seq, ['*', '*', '*'])

    def test_start_read_2(self):
        candidates = get_candidates(test_constants.CANDIDATES)
        region = compute_regions(candidates, window=7)[0]
        region.start = 1
        region.end = 8
        seq = ['A', 'C', 'G', 'T', 'A', 'C', 'G', 'T']
        bqs = [22, 20, 30, 15, 22, 20, 30, 15]
        ref_start = 3
        entire_ins_map = {1: 4}
        soft_clipped_len_at_start = 0
        new_seq, new_bqs = start_read(
            seq,
            bqs,
            ref_start,
            region,
            entire_ins_map,
            soft_clipped_len_at_start
        )
        np.testing.assert_array_equal(new_seq, ['*', '*', '*', '*', '*', '*', '*'])

    def test_start_read_3(self):
        candidates = get_candidates(test_constants.CANDIDATES)
        region = compute_regions(candidates, window=7)[0]
        region.start = 1
        region.end = 8
        seq = ['A', 'C', 'G', 'T', 'A', 'C', 'G', 'T']
        bqs = [22, 20, 30, 15, 22, 20, 30, 15]
        ref_start = 3
        entire_ins_map = {1: 4}
        soft_clipped_len_at_start = 2
        new_seq, new_bqs = start_read(
            seq,
            bqs,
            ref_start,
            region,
            entire_ins_map,
            soft_clipped_len_at_start
        )
        np.testing.assert_array_equal(new_seq, ['*', '*', '*', '*', '*', 'A', 'C'])

    def test_start_read_4(self):
        candidates = get_candidates(test_constants.CANDIDATES)
        region = compute_regions(candidates, window=7)[0]
        region.start = 1
        region.end = 8
        seq = ['A', 'C', 'G', 'T', 'A', 'C', 'G', 'T']
        bqs = [22, 20, 30, 15, 22, 20, 30, 15]
        ref_start = 3
        entire_ins_map = {1: 4, 2: 1}
        soft_clipped_len_at_start = 2
        new_seq, new_bqs = start_read(
            seq,
            bqs,
            ref_start,
            region,
            entire_ins_map,
            soft_clipped_len_at_start
        )
        np.testing.assert_array_equal(new_seq, ['*', '*', '*', '*', '*', '*', 'A', 'C'])

    def test_compute_alignment_for_bam_1(self):
        """Test compute_alignment_for_bam(): Alignment in one BAM file for region 1."""
        # region 1
        candidates = get_candidates(test_constants.CANDIDATES)
        regions = compute_regions(candidates, window=7)

        # tumor
        alignment_matrices = compute_alignment_for_bam(
            test_constants.TUMOR_BAM, regions[0], 7
        )
        al_tum = alignment_matrices.nuc
        np.testing.assert_array_equal(al_tum, np.loadtxt(test_constants.AL_TUM1_BAM, dtype=str))
        self.assertDictEqual(alignment_matrices.ins_map, test_constants.INS_MAP_TUM1)

        # normal
        alignment_matrices = compute_alignment_for_bam(
            test_constants.NORMAL_BAM, regions[0], 7
        )
        al_nor = alignment_matrices.nuc
        np.testing.assert_array_equal(al_nor, np.loadtxt(test_constants.AL_NOR1_BAM, dtype=str))
        self.assertDictEqual(alignment_matrices.ins_map, test_constants.INS_MAP_NOR1)

    def test_compute_alignment_for_bam_2(self):
        """Test compute_alignment_for_bam(): Alignment in one BAM file for region 2."""
        # region 2
        candidates = get_candidates(test_constants.CANDIDATES)
        regions = compute_regions(candidates, window=7)

        # tumor
        alignment_matrices = compute_alignment_for_bam(
            test_constants.TUMOR_BAM, regions[1], 7
        )
        al_tum = alignment_matrices.nuc

        np.testing.assert_array_equal(al_tum, np.loadtxt(test_constants.AL_TUM2_BAM, dtype=str))
        self.assertDictEqual(alignment_matrices.ins_map, test_constants.INS_MAP_TUM2)

        # normal
        alignment_matrices = compute_alignment_for_bam(
            test_constants.NORMAL_BAM, regions[1], 7
        )
        al_nor = alignment_matrices.nuc
        np.testing.assert_array_equal(al_nor, np.loadtxt(test_constants.AL_NOR2_BAM, dtype=str))
        self.assertDictEqual(alignment_matrices.ins_map, test_constants.INS_MAP_NOR2)

    def test_align_tumor_normal_1(self):
        """Test align(): Re-alignment of two alignments (tumor and normal) for region 1."""
        # region 1
        candidates = get_candidates(test_constants.CANDIDATES)
        regions = compute_regions(candidates, window=7)
        al_matst = compute_alignment_for_bam(test_constants.TUMOR_BAM, regions[0], 7)
        al_matsn = compute_alignment_for_bam(test_constants.NORMAL_BAM, regions[0], 7)

        align_tumor_normal(al_matst, al_matsn, [''])  # TODO: test ref as well

        np.testing.assert_array_equal(al_matst.nuc, np.loadtxt(test_constants.AL_TUM1, dtype=str))
        np.testing.assert_array_equal(al_matsn.nuc, np.loadtxt(test_constants.AL_NOR1, dtype=str))

    def test_align_tumor_normal_2(self):
        """Test align(): Re-alignment of two alignments (tumor and normal) for region 2."""
        # region 2
        candidates = get_candidates(test_constants.CANDIDATES)
        regions = compute_regions(candidates, window=7)
        al_matst = compute_alignment_for_bam(test_constants.TUMOR_BAM, regions[1], 7)
        al_matsn = compute_alignment_for_bam(test_constants.NORMAL_BAM, regions[1], 7)

        align_tumor_normal(al_matst, al_matsn, [''])
        np.testing.assert_array_equal(al_matst.nuc, np.loadtxt(test_constants.AL_TUM2, dtype=str))
        np.testing.assert_array_equal(al_matsn.nuc, np.loadtxt(test_constants.AL_NOR2, dtype=str))

    def test_align_ref(self):
        # create test variables
        ref_seq = ['A', 'C', 'A', 'C', 'A', 'C', 'C', 'C']
        ins_map = {5: 2, 6: 1, 1: 3}

        ref_seq_after = ['A', 'C', '-', '-', '-', 'A', 'C', 'A', 'C', '-', '-', 'C', '-', 'C']

        # call function
        ref_seq_new = align_ref(ref_seq, ins_map, 14)

        # test_function_output
        np.testing.assert_array_equal(ref_seq_after, ref_seq_new)

    def test_align_for_one_1(self):
        # create test variables
        other_ins_map = {5: 2, 6: 1, 1: 3}
        self_ins_map = {5: 1, 6: 1}
        inter_mats = IntermediaryMatrices(self_ins_map, 5, 0)
        inter_mats.nuc = np.array([
            ['A', 'C', 'A', 'C', 'A', 'C', 'C', 'C', 'C', 'C'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-', 'C'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-', 'C'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-', 'C'],
            ['A', 'G', 'A', 'G', 'A', '-', '-', 'G', '-', 'G'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-', 'C'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-', 'C'],
            ['A', 'C', 'A', 'C', 'A', 'C', '-', 'C', '-', 'C']
        ])
        nuc_after = np.array([
            ['A', 'C', '-', '-', '-', 'A', 'C', 'A', 'C', 'C', '-', 'C', 'C', 'C'],
            ['A', 'C', '-', '-', '-', 'A', 'C', 'A', 'C', '-', '-', 'C', '-', 'C'],
            ['A', 'C', '-', '-', '-', 'A', 'C', 'A', 'C', '-', '-', 'C', '-', 'C'],
            ['A', 'C', '-', '-', '-', 'A', 'C', 'A', 'C', '-', '-', 'C', '-', 'C'],
            ['A', 'G', '-', '-', '-', 'A', 'G', 'A', '-', '-', '-', 'G', '-', 'G'],
            ['A', 'C', '-', '-', '-', 'A', 'C', 'A', 'C', '-', '-', 'C', '-', 'C'],
            ['A', 'C', '-', '-', '-', 'A', 'C', 'A', 'C', '-', '-', 'C', '-', 'C'],
            ['A', 'C', '-', '-', '-', 'A', 'C', 'A', 'C', '-', '-', 'C', '-', 'C']
        ])
        inter_mats.bse = np.array([
            [50, 40, 50, 40, 50, 40, 40, 40, 40, 40],
            [50, 40, 50, 40, 50, 40, 0, 40, 0, 40],
            [50, 40, 50, 40, 50, 40, 0, 40, 0, 40],
            [50, 40, 50, 40, 50, 40, 0, 40, 0, 40],
            [50, 45, 50, 45, 50, 0, 0, 45, 0, 45],
            [50, 40, 50, 40, 50, 40, 0, 40, 0, 40],
            [50, 40, 50, 40, 50, 40, 0, 40, 0, 40],
            [50, 40, 50, 40, 50, 40, 0, 40, 0, 40]
        ])
        bse_after = np.array([
            [50, 40, 0, 0, 0, 50, 40, 50, 40, 40, 0, 40, 40, 40],
            [50, 40, 0, 0, 0, 50, 40, 50, 40, 0, 0, 40, 0, 40],
            [50, 40, 0, 0, 0, 50, 40, 50, 40, 0, 0, 40, 0, 40],
            [50, 40, 0, 0, 0, 50, 40, 50, 40, 0, 0, 40, 0, 40],
            [50, 45, 0, 0, 0, 50, 45, 50, 0, 0, 0, 45, 0, 45],
            [50, 40, 0, 0, 0, 50, 40, 50, 40, 0, 0, 40, 0, 40],
            [50, 40, 0, 0, 0, 50, 40, 50, 40, 0, 0, 40, 0, 40],
            [50, 40, 0, 0, 0, 50, 40, 50, 40, 0, 0, 40, 0, 40],

        ])

        # call function
        nuc_new, bse_new = align_for_one(other_ins_map, inter_mats)

        # test the output of the function
        np.testing.assert_array_equal(
            nuc_new,
            nuc_after
        )
        np.testing.assert_array_equal(
            bse_new,
            bse_after
        )

    def test_align_for_one_2(self):
        candidates = get_candidates(test_constants.CANDIDATES)
        regions = compute_regions(candidates, window=7)
        inter_matst = compute_alignment_for_bam(test_constants.TUMOR_BAM, regions[0], 7)
        inter_matsn = compute_alignment_for_bam(test_constants.NORMAL_BAM, regions[0], 7)

        inter_matsn.nuc, inter_matsn.bse = align_for_one(inter_matst.ins_map, inter_matsn)
        inter_matst.nuc, inter_matst.bse = align_for_one(inter_matsn.ins_map, inter_matst)

        np.testing.assert_array_equal(
            inter_matsn.nuc,
            np.loadtxt(test_constants.AL_NOR1, dtype=str)
        )
        np.testing.assert_array_equal(
            inter_matst.nuc,
            np.loadtxt(test_constants.AL_TUM1, dtype=str)
        )


if __name__ == '__main__':
    unittest.main()
