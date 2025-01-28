import logging

import sys

logger = logging.getLogger(__name__)
sys.path.append('./bam2tensor')

import pybedtools as pybed
import pysam
import traceback
from collections import OrderedDict
from aligner_utils import *
from intermediary_matrices import IntermediaryMatrices, unify_read


def compute_alignment(
        tumor_path: Text,
        normal_path: Text,
        ref_path: Text,
        region: pybed.cbedtools.Interval,
        window: int
) -> Tuple[IntermediaryMatrices, IntermediaryMatrices, List]:
    """Realign BAMs by adding gaps to the reference sequence at positions of
    insertions in tumor or normal sequences.

    :param tumor_path: The path to the tumor BAM file.
    :param normal_path: The path to the normal BAM file.
    :param ref_path: The path to the reference sequence.
    :param region: The region to be aligned.
    :param window: Window size around the candidate.
    :return: IntermediateMatrices object for tumor, normal, and a list of
    reference bases.
    """
    try:
        logger.debug('region: {}'.format(region))

        # Compute the alignments for tumor and normal
        inter_matst = compute_alignment_for_bam(tumor_path, region, window)
        inter_matsn = compute_alignment_for_bam(normal_path, region, window)
        ref_seq = get_reference_seq(ref_path, region, window)

        if check_if_empty(inter_matst, inter_matsn, region):
            return tuple()

        align_tumor_normal(inter_matst, inter_matsn, ref_seq)
        max_ref_len = np.array(inter_matst.nuc).shape[1]
        ref_seq = align_ref(ref_seq, inter_matst.ins_map, max_ref_len)

        inter_matst, inter_matsn = adjust_position(inter_matst, inter_matsn)

        return inter_matst, inter_matsn, ref_seq
    except Exception as e:
        logger.warning(('Couldn\'t parse region: {}'.format(region)))
        logger.warning(e)
        traceback.print_exc()
        return tuple()


def compute_alignment_for_bam(
        bam_path: Text,
        region: pybed.cbedtools.Interval,
        window: int,
) -> IntermediaryMatrices:
    """Redo the alignment of BAM by virtually adding gaps to the reference
    sequence in positions of insertions.

    :param bam_path: The path to the BAM file.
    :param region: The region where the alignment is to be computed.
    :param window: length of window
    :return: IntermediateMatrices object for the given BAM and region.
    """
    mutation_pos = int(region[3]) - region.start
    if mutation_pos < 0:
        logger.info(
            'Mutation position: {} Region: {}'.format(mutation_pos, region))
        raise Exception('Mutation not in region. Skipping this region')

    sam_file = pysam.AlignmentFile(bam_path, 'rb')
    ins_map = get_ins_pos_to_length_mapping(sam_file, region)
    ins_map = OrderedDict(sorted(ins_map.items()))
    inter_mats = IntermediaryMatrices(ins_map, mutation_pos, region[5])

    # parse each read_record in the region and add the related information to
    # inter_matrices object.
    for read_record in sam_file.fetch(region.chrom, region.start, region.end):
        if read_record.mapping_quality <= 0:
            continue
        if read_record.flag >= 256:
            continue
        if read_record.has_tag('AS') and read_record.has_tag('XS'):
            if read_record.get_tag('XS') >= read_record.get_tag('AS'):
                continue
        try:
            read_info = align_sequence_for_read(read_record, region, ins_map)
            nuc, bsq, mapq, strand, xtt, nmt, ast, mdt, xst, flag = read_info
            inter_mats.append_read(
                nuc, bsq, mapq, strand, xtt, nmt, ast, mdt, xst, flag
            )
        except TypeError:
            logger.debug('TypeError in read_record:\n{}'.format(read_record))
        except AttributeError:
            logger.debug('Read record doesn\'t have a sequence')
    sam_file.close()
    # complete the rows in inter_matrices if they fall short.
    max_read_length = window * 2 + 1
    inter_mats.complete_rows(max_read_length)

    inter_mats.ins_map = get_read_indexed_ins_map(ins_map, region)
    return inter_mats


def align_sequence_for_read(
        read: pysam.AlignedSegment,
        region: pybed.cbedtools.Interval,
        entire_ins_map: Dict[int, int]
) -> Tuple[List, List, float, bool, Text, float, float, Text, float, int]:
    """Align the sequence such that all reads show the same position in genome
    in same index.

    :param read: The current read, a pysam.AlignedSegment object
    :param region: Region of interest in the genome
    :param entire_ins_map: Insertion position: length mappings
    :return: Sequence and base qualities aligned to others
    """
    ref_start = read.reference_start
    seq, bqs = get_sequence_and_baseq(read)
    soft_clipped_positions, start_soft_clip_len = get_soft_clipping_info(read)

    aligned_seq, aligned_bqs = start_read(
        seq, bqs, ref_start, region, entire_ins_map, start_soft_clip_len
    )

    ins_len = 0
    prev_ref_pos = ref_start
    for seq_pos, ref_pos in read.get_aligned_pairs():
        if ref_pos is not None and ref_pos < region.start - 1:
            # skip if the base is not within the region
            continue
        if seq_pos is not None and seq_pos in soft_clipped_positions:
            if seq_pos > start_soft_clip_len:
                # Add the soft clipped region at the end of the sequence.
                aligned_seq.append(seq[seq_pos])
                aligned_bqs.append(bqs[seq_pos])
            # skip if the position is within soft clipped positions at start
            # (already added)
            continue
        if (ref_pos and seq_pos is not None):
            # Add the base to the position if all planets have aligned
            # properly.
            aligned_seq.append(seq[seq_pos])
            aligned_bqs.append(bqs[seq_pos])
        if seq_pos is None:
            # Add a gap if there is a deletion at the current position.
            aligned_seq.append('-')
            aligned_bqs.append(0)
        if ref_pos in entire_ins_map.keys():
            # If there is an insertion in any of the sequences but not the
            # current one, add gaps.
            ins_len = entire_ins_map[ref_pos]
            aligned_seq.extend(['-'] * ins_len)
            aligned_bqs.extend([0] * ins_len)
        if prev_ref_pos in entire_ins_map.keys() \
                and ref_pos is None \
                and prev_ref_pos >= region.start - 1 \
                and len(aligned_seq) > 0 \
                and ins_len > 0:
            # If there was an insertion in the previous position at the current
            # sequence
            aligned_seq[-ins_len] = seq[seq_pos]
            aligned_bqs[-ins_len] = bqs[seq_pos]
            ins_len -= 1
        if ref_pos is not None:
            prev_ref_pos = ref_pos
    return (
        aligned_seq,
        aligned_bqs,
        read.mapping_quality,
        read.is_reverse,
        read.get_tag('XT') if read.has_tag('XT') else '',
        read.get_tag('NM') if read.has_tag('NM') else 0.,
        read.get_tag('AS') if read.has_tag('AS') else 0.,
        read.get_tag('MD') if read.has_tag('MD') else '',
        read.get_tag('XS') if read.has_tag('XS') else 0.,
        read.flag
    )


def start_read(
        seq: List[Text],
        bqs: List[float],
        ref_start: int,
        region: pybed.cbedtools.Interval,
        entire_ins_map: Dict[int, int],
        soft_clipped_len_at_start: int
) -> Tuple[List, List]:
    """Pad the beginning of the read such that it starts with gap
    characters/soft clipped bases.

    e.g.
    Read starts before/at the region start:
    padded_seq = []
    Read starts 3 bases after the region start:
    padded_seq = ['*', '*', '*']
    Read starts 3 bases after the region start with soft clipped sequence 'ACT'
    padded_seq = ['A', 'C', 'T']

    :param seq: The read sequence
    :param bqs: Base qualities
    :param ref_start: Start position of the read
    :param region: Region of interest
    :param entire_ins_map: Map of insertions
    :param soft_clipped_len_at_start: Length of the soft clipped region at the
    start of the read
    :return: Padded sequence and padded base qualities.
    """
    padded_seq, padded_bqs = [], []

    # Step 1: Determine whether there has been an insertion (and its length)
    # before the current sequence started.
    difference = 0
    for pos, ins_len in entire_ins_map.items():
        if ref_start - 1 >= pos >= region.start - 1:
            difference += ins_len

    # Step 2: Determine whether the read starts from the start of the region,
    # and if not the difference between
    if ref_start > region.start - 1:
        difference += ref_start - region.start + 1

    # Step 3: Fill in the read start in necessary, incorporating the soft
    # clipped region.
    if difference > 0:
        if soft_clipped_len_at_start == 0:
            # if there was no soft clipping, simply pad with default values
            padded_seq.extend(['*'] * difference)
            padded_bqs.extend([0] * difference)
        else:
            # if there was soft clipping, determine the difference and pad
            # accordingly.
            if soft_clipped_len_at_start >= difference:
                padded_seq.extend(seq[:difference])
                padded_bqs.extend(bqs[:difference])
            else:
                missing_len = difference - soft_clipped_len_at_start
                padded_seq.extend(['*'] * missing_len)
                padded_seq.extend(seq[:soft_clipped_len_at_start])
                padded_bqs.extend([0] * missing_len)
                padded_bqs.extend(bqs[:soft_clipped_len_at_start])
    return padded_seq, padded_bqs


def check_if_empty(inter_matst, inter_matsn, region):
    if inter_matst.is_empty() and inter_matsn.is_empty():
        logger.info('Both matrices are empty in region {}'.format(region))
        return True
    if inter_matsn.is_empty():
        logger.info('Normal matrix is empty in region {}.'.format(region))
        return True
    if inter_matst.is_empty():
        logger.info('Tumor matrix is empty. Filling it in!')
        logger.info(len(inter_matsn.nuc[0]))
        inter_matst.fill_with_dummy(len(inter_matsn.nuc[0]))
    return False


def align_tumor_normal(
        inter_mats_tum: IntermediaryMatrices,
        inter_mats_nor: IntermediaryMatrices,
        reference: List[Text],
):
    """Restructure two alignments such that they have gaps in positions where
    the other has insertions.

    :param inter_mats_tum: Map of insertions in tumor alignment,
    insertion_position: insertion_length.
    :param inter_mats_nor: Map of insertions in normal alignment,
    insertion_position: insertion_length.
    :param reference: Reference sequence.
    """
    inter_mats_nor.nuc, inter_mats_nor.bse = align_for_one(
        inter_mats_tum.ins_map, inter_mats_nor)
    inter_mats_tum.nuc, inter_mats_tum.bse = align_for_one(
        inter_mats_nor.ins_map, inter_mats_tum)
    ins_map = merge_insertion_mappings(
        [inter_mats_tum.ins_map, inter_mats_nor.ins_map]
    )
    ins_map = OrderedDict(sorted(ins_map.items()))
    inter_mats_tum.ins_map = ins_map
    inter_mats_nor.ins_map = ins_map
    extend_alignment_matrices(inter_mats_tum, inter_mats_nor, reference)


def align_ref(
        ref_seq: List[Text],
        ins_map: Dict[int, int],
        max_len: int,
) -> List[Text]:
    """Given an insertion mapping, align the reference genome (add gaps where
    tumor/normal has gaps)

    :param ref_seq: Reference sequence as a list of characters
    :param ins_map: Insertion position: insertion length dictionary
    :param max_len: Maximum length of the reference sequence
    :return: Reference sequence with gaps
    """
    added_no = 0
    new_seq = ref_seq
    for position in sorted(ins_map.keys()):
        if position > len(new_seq):
            continue
        ins_start_pos = position + 1 + added_no
        new_seq = new_seq[:ins_start_pos]
        new_seq.extend(['-'] * ins_map[position])
        new_seq.extend(ref_seq[ins_start_pos - added_no:])
        added_no += ins_map[position]
    return new_seq[0: max_len]


def align_for_one(
        other_ins_map: Dict[int, int],
        inter_matrices: IntermediaryMatrices
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-align one of the alignments by adding gaps on positions the other has
    insertions.

    :param other_ins_map: Insertion positions of the other alignment.
    :param inter_matrices: IntermediateMatrices object
    """
    different_insertion_positions = get_different_insertions(
        other_ins_map, inter_matrices.ins_map
    )
    added_no = 0
    self_insertion_pos = sorted(inter_matrices.ins_map.keys())
    nuc_cols, bse_cols = inter_matrices.get_transpose()

    for ins_pos in sorted(different_insertion_positions.keys()):
        ins_diff = different_insertion_positions[ins_pos]
        added_no += compute_added_no(inter_matrices.ins_map,
                                     self_insertion_pos, ins_pos)
        ins_start_pos = ins_pos + 1 + added_no
        nuc_cols = add_default_cols(
            nuc_cols,
            ins_start_pos,
            ins_diff,
            default_col=['-'] * len(nuc_cols[0])
        )
        bse_cols = add_default_cols(
            bse_cols,
            ins_start_pos,
            ins_diff,
            default_col=[0] * len(nuc_cols[0])
        )
        added_no += ins_diff

    nuc_rows = np.transpose(nuc_cols)
    for i, row in enumerate(nuc_rows):
        nuc_rows[i, :] = unify_read(row)

    return nuc_rows, np.transpose(bse_cols)


def adjust_position(inter_matst, inter_matsn):
    new_mutation_pos = inter_matsn.mutation_pos
    for pos, length in inter_matsn.ins_map.items():
        if inter_matsn.mutation_pos > pos:
            new_mutation_pos += length

    inter_matsn.mutation_pos = new_mutation_pos
    inter_matst.mutation_pos = new_mutation_pos
    return inter_matst, inter_matsn
