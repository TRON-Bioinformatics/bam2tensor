"""Helper functions to compute insertion position-length dictionaries."""

import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Text
import numpy as np
import pybedtools as pybed
import pysam
import sys

sys.path.append('./bam2tensor')

from intermediary_matrices import IntermediaryMatrices

logger = logging.getLogger(__name__)


def get_sequence_and_baseq(
        read: pysam.AlignedSegment
) -> Tuple[List[Text], List[Text]]:
    """Retrieve the sequence and base quality information from read object,
    reverse complement if in reverse strand.

    Base qualities are set to 0 if they are not found.

    :param read: AlignedSegment object of pysam.
    :return: A list of nucleotides and a list of base qualities.
    """
    reverse_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', '-': '-'}

    seq = read.get_forward_sequence().upper().replace('N', '-') # TODO
    bqs = read.get_forward_qualities()
    if read.is_reverse:
        seq = ''.join(reverse_dict[base] for base in seq)[::-1]
        bqs = bqs[::-1]

    bqs = bqs if bqs else [0] * len(seq)
    return seq, bqs


def get_reference_seq(
        ref_path: str, region: pybed.cbedtools.Interval, window: int
) -> List[Text]:
    """Retrieve reference sequence in the given region from fasta file.

    :param ref_path: Path to the reference fasta file.
    :param region: The genomic region of interest.
    :param window: Window size of the final tensor.
    :return: seq - Reference sequence as a list of capital letters.
    """
    file = pysam.FastaFile(ref_path)
    seq = []
    s = region.start - 1
    e = region.start + (window * 2)
    for read in file.fetch(region.chrom, s, e):
        seq.append(read.upper())
    return seq


def get_ins_pos_to_length_mapping(
        sam_file: pysam.AlignmentFile,
        region: pybed.cbedtools.Interval
) -> Dict[int, int]:
    """ Create a dictionary that contains the insertion positions and lengths

    :param sam_file: Alignment file.
    :param region: The genomic region of interest.
    :return: A dictionary with containing {position: length} of insertions in
    the alignment file within the given region
    """
    all_ins_maps = compute_single_insertion_mappings(sam_file, region)
    return merge_insertion_mappings(all_ins_maps)


def compute_single_insertion_mappings(
        sam_file: pysam.AlignmentFile,
        region: pybed.cbedtools.Interval
) -> List[Dict[int, int]]:
    """Compute a dictionary that maps positions of insertions to their lengths.

    :param sam_file: Alignment file for which the mappings will be calculated.
    :param region: Region of interest in the genome
    :return: A list of dictionaries containing insertion position-length
    mappings for all the reads in the region.
    """
    all_ins_maps = []
    for read in sam_file.fetch(region.chrom, region.start, region.end):
        try:
            ins_map = {}
            prev_ref_pos = read.reference_start
            soft_clipped_positions, _ = get_soft_clipping_info(read)

            for seq_pos, ref_pos in read.get_aligned_pairs():
                if not ref_pos and seq_pos not in soft_clipped_positions:
                    if prev_ref_pos not in ins_map:
                        ins_map[prev_ref_pos] = 1
                    else:
                        ins_map[prev_ref_pos] += 1
                else:
                    prev_ref_pos = ref_pos
            all_ins_maps.append(ins_map)
        except:
            logger.debug(read)
    sam_file.seek(0)
    return all_ins_maps


def merge_insertion_mappings(
        all_ins_maps: List[Dict[int, int]]
) -> Dict:
    """Merge the dictionaries of insertion position:length mappings such that
    the mapping with the longest length is kept.

    :param all_ins_maps: All insertion position:length mappings from all the
    reads.
    :return: Merged insertion position:length mapping.
    """
    entire_ins_map = defaultdict(list)

    for ins_map in all_ins_maps:
        for k, v in ins_map.items():
            if k is not None:  # if key is not none
                entire_ins_map[k].append(v)
    for pos, length_list in entire_ins_map.items():
        entire_ins_map[pos] = max(length_list) if length_list else 0
    return entire_ins_map


def get_read_indexed_ins_map(
        ins_map: Dict[int, int],
        region: pybed.cbedtools.Interval
) -> Dict[int, int]:
    """Get the insertion position:length dictionary where the position is
    indexed based on the current region.

    e.g. from dictionary {1257333: 2, 1257346: 1}
         to dictionary {11: 2, 24: 1}
    :param ins_map: Insertion position-length mappings
    :param region: Region of interest.
    :return: Insertion position-length mappings indexed on the region.
    """
    read_indexed_ins_map = {}
    for pos, length in ins_map.items():
        pos_in_read = pos - region.start + 1
        if pos_in_read >= 0:
            read_indexed_ins_map[pos_in_read] = length
    return read_indexed_ins_map


def get_soft_clipping_info(read: pysam.AlignedSegment) -> Tuple[List, int]:
    """Retrieve the positions where soft clipping occurs and the length of soft
    clipping in the beginning of the read.

    :param read: AlignedSegment object from pysam.
    :return: Soft clipped starting positions as a list and the length of the
    soft clipping in the beginning of the read.
    """
    current_pos = 0
    soft_clipped_positions = []
    soft_clipped_len_at_start = 0
    for tpl in read.cigartuples:
        if tpl[0] == 4:  # if it's soft clipped
            start, end = current_pos, current_pos + tpl[1]
            soft_clipped_positions.extend(range(start, end))
        # if it's soft clipped & in the beginning of the read:
        if tpl[0] == 4 and current_pos == 0:
            soft_clipped_len_at_start += tpl[1]
        current_pos += tpl[1]
    return soft_clipped_positions, soft_clipped_len_at_start


def extend_alignment_matrices(
        alignment_matricest: IntermediaryMatrices,
        alignment_matricesn: IntermediaryMatrices,
        reference: List[Text]
):
    """Extend the alignment matrices such that tumor, normal, and reference
    sequence lengths match (in case they didn't before)

    :param alignment_matricest: IntermediateMatrices of tumor reads
    :param alignment_matricesn: IntermediateMatrices of normal reads
    :param reference: Reference sequence.
    """
    rt, ct = alignment_matricest.shape_of_one()
    rn, cn = alignment_matricesn.shape_of_one()
    diff = ct - cn
    if diff > 0:
        alignment_matricesn.nuc = np.hstack(
            (alignment_matricesn.nuc, np.array([['*'] * abs(diff)] * rn))
        )
        alignment_matricesn.bse = np.hstack(
            (alignment_matricesn.bse, np.array([[0] * abs(diff)] * rn))
        )
    elif diff < 0:
        alignment_matricest.nuc = np.hstack(
            (alignment_matricest.nuc, np.array([['*'] * abs(diff)] * rt))
        )
        alignment_matricest.bse = np.hstack(
            (alignment_matricest.bse, np.array([[0] * abs(diff)] * rt))
        )
    rt, ct = alignment_matricest.shape_of_one()
    diff = ct - len(reference)
    if diff > 0:
        reference.extend(['*'] * diff)


def get_different_insertions(
        ins_map1: Dict[int, int], ins_map2: Dict[int, int]
) -> Dict[int, int]:
    """Get the difference between insertion positions (ins_map1 - ins_map2),
    also when the alignment length is different.

    :param ins_map1: Map of insertions in one alignment, position:length.
    :param ins_map2: Map of insertions in the other alignment, position:length.
    :return: A dictionary of insertion_pos: insertion_length_difference.
    """
    different_insertions = {}
    for pos, length in ins_map1.items():
        if pos not in ins_map2:
            different_insertions[pos] = length
        elif pos in ins_map2 and ins_map2[pos] < ins_map1[pos]:
            diff = length - ins_map2[pos]
            different_insertions[pos] = diff
    return different_insertions


def compute_added_no(
        self_ins_map: Dict[int, int],
        self_ins_pos: List[int],
        ins_pos: int
) -> int:
    """Compute the last number of added bases/gaps before the current insertion
     position.

    Side effects: Removes values from self_ins_pos list

    :param self_ins_map: Map of insertions in the alignment, position:length.
    :param self_ins_pos: List of sorted insertion positions, the processed
    positions are removed afterwards.
    :param ins_pos: The original position of the insertion.
    :return: The number of added bases/gaps before the current insertion pos.
    """
    added_no = 0
    before_pos = []
    for pos in self_ins_pos:
        if ins_pos >= pos:
            added_no += self_ins_map[pos]
            before_pos.append(pos)
    for pos in before_pos:
        self_ins_pos.remove(pos)
    return added_no


def add_default_cols(
        cols: List[object],
        ins_start_pos: int,
        ins_diff: int,
        default_col: List[object]
) -> List[object]:
    """Add default columns to the new insertion positions.

    :param cols: Columns before the insertion.
    :param ins_start_pos: Start position of the insertion.
    :param ins_diff: The number of columns to be added in between.
    :param default_col: Default/empty column, e.g. ['-', '-', ..., '-'] for
    sequence columns.
    :return: The new columns.
    """
    new_cols = cols[:ins_start_pos]
    [new_cols.append(default_col) for _ in range(ins_diff)]
    new_cols.extend(cols[ins_start_pos:])
    return new_cols
