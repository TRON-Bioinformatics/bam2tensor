import logging
import random
from typing import Dict, List, Text, Tuple

import numpy as np
import pybedtools as pybed

logger = logging.getLogger(__name__)
random.seed(157643)


class IntermediaryMatrices:

    def __init__(
            self,
            ins_map: Dict[int, int],
            mutation_pos: int,
            mutation_length: int,
            region: pybed.cbedtools.Interval = None,
    ):
        """Initialize the object.

        :param ins_map: A dictionary of insertion position:insertion length.
        :param mutation_pos: The position of the mutation w.r.t the start of
        the matrix.
        :param mutation_length: The length of the mutation.
        :param region: The genomic region of interest
        """
        self.nuc = []
        self.bse = []
        self.map = []
        self.str = []
        self.xtt = []
        self.nmt = []
        self.ast = []
        self.mdt = []
        self.xst = []
        self.flg = []
        self.ins_map = ins_map
        self.mutation_pos = mutation_pos
        self.mutation_length = abs(int(mutation_length))
        self.region = region

    def append_read(
            self,
            nuc_lst: List[Text],
            baseq_lst: List[float],
            mapq: float,
            strand: bool,
            xt_tag: Text,
            nm_tag: float,
            as_tag: float,
            md_tag: Text,
            xs_tag: float,
            flag: int
    ):
        """Given the necessary information about a read, append it to relevant
        matrices/vectors.

        :param nuc_lst: List of nucleotides in the sequence
        :param baseq_lst: List of base qualities per nucleotide
        :param mapq: Mapping quality of the read. Integer.
        :param strand: Strand information of the read. True/False
        (forward/reverse).
        :param xt_tag: XT tag associated with the read.
        One of U(unique), M(matched), ?(?)
        :param nm_tag: NM tag associated with the read. (edit distance)
        :param as_tag: AS tag associated with the read. (alignment score)
        :param md_tag: MD tag associated with the read. (string for mismatching
        positions)
        :param nm_tag: XS tag associated with the read. (suboptimal alignment
        score)
        :param flag: SAM flag of the read. Integer
        """
        nuc_lst = unify_read(nuc_lst)
        self.nuc.append(nuc_lst)
        self.bse.append(baseq_lst)
        self.map.append(mapq)
        self.str.append(strand)
        self.xtt.append(xt_tag)
        self.nmt.append(nm_tag)
        self.ast.append(as_tag)
        self.mdt.append(md_tag)
        self.xst.append(xs_tag)
        self.flg.append(int(flag))

    def complete_rows(self, max_read_length: int):
        """Complete the rows so that each matrix has the same number of
        nucleotides & base qualities.

        :param max_read_length: The length of the read with highest num. chars.
        """
        if len(self.nuc) > 0:
            max_read_length_ = len(max(self.nuc, key=len))
            if max_read_length_ > max_read_length:
                max_read_length = max_read_length_
        for nuc_row, bse_row in zip(self.nuc, self.bse):
            diff = max_read_length - len(nuc_row)
            if diff > 0:
                nuc_row.extend(['*'] * diff)
                bse_row.extend([0] * diff)

    def get_transpose(self) -> Tuple[object, object]:
        """Transpose the nucleotide and base quality lists.

        :return: Transposed nucleotide and base quality lists.
        """
        return (
            np.transpose(self.nuc).tolist(),
            np.transpose(self.bse).tolist()
        )

    def is_empty(self) -> bool:
        """Check if the matrices are empty.

        :return: True if empty, False otherwise.
        """
        return len(self.nuc) == 0

    def shape_of_one(self):
        """Get the shape of one of the matrices

        :return: The shape of one of the matrices
        """
        return np.array(self.nuc).shape

    def fill_with_dummy(
            self,
            length: int
    ):
        """Fill the matrices with dummy values.

        :param length: Length of the number of dummy reads.
        """
        self.nuc = [['*'] * length] * 5
        self.bse = [[0] * length] * 5
        self.map = [0] * 5
        self.str = [False] * 5
        self.xtt = [''] * 5
        self.nmt = [0] * 5
        self.ast = [0] * 5
        self.mdt = [''] * 5
        self.xst = [0] * 5
        self.flg = [0] * 5

    def get_mutation_range(
            self,
            window: int,
            mut_type: str
    ):
        """Get the range of mutation within the window.

        :param window: Window size.
        :param mut_type: Type of mutation, one of point, del, ins.
        :return: A range showing where the mutation is.
        """
        mutation_length = self.mutation_length
        if self.mutation_length == 0:
            mutation_length = 1
        start = self.mutation_pos
        end = min(self.mutation_pos + mutation_length, window * 2)
        if mut_type == 'ins':
            start += 1
            end = min(self.mutation_pos + mutation_length + 1, window * 2)
        if start == end:
            end += 1
        elif start > end:
            raise Exception(
                'Mutation range is not correct: Mutation is not within the window'
            )
        return range(start, end)

    def scale_matrices_horizontal(
            self,
            reference_seq: List[Text],
            matrix_width: int,
            window: int,
            change_ref: bool = True,
    ) -> List[Text]:
        """Crop columns from or add dummy columns to the matrix to scale its
        width.

        Side effects: Changes the width of self.nuc and self.bse members.

        :param reference_seq: The reference sequence.
        :param matrix_width: The expected width of the output matrices.
        :param window: The length of the window around the mutation site.
        :param change_ref: Whether or not to change the reference sequence.
        :return: Reference sequenced, trimmed or extended with gaps.
        """
        # TODO: clean up
        # step 1: check if there is an offset (in case mutation start < window)
        offset = self.mutation_pos - window
        if offset < 0:
            padding_pattern = (0, 0), (offset * -1, 0)
            self.nuc = np.pad(self.nuc, padding_pattern, mode='constant',
                              constant_values='*')
            self.bse = np.pad(self.bse, padding_pattern, mode='constant',
                              constant_values=0)
            self.mutation_pos = self.mutation_pos + padding_pattern[1][0]
            if change_ref:
                reference_seq = np.pad(np.array(reference_seq, dtype=object),
                                       padding_pattern[1],
                                       mode='constant',
                                       constant_values='*').tolist()

        # step 2: check if there is an offset (to place the mutation in the
        # middle)
        offset = self.mutation_pos - window
        r, c = self.shape_of_one()
        width = window * 2 + 1 + offset
        if offset > 0 or c > width:
            self.nuc = self.nuc[:, offset:width]
            self.bse = self.bse[:, offset:width]
            if change_ref:
                reference_seq = reference_seq[offset:width]
        self.mutation_pos = self.mutation_pos - offset

        # step 3: if the matrix is narrower than expected matrix width, make it
        # wider by adding dummy values
        r, c = self.shape_of_one()
        if c < matrix_width:
            padding_pattern = self.calculate_padding_pattern(matrix_width, c)
            self.nuc = np.pad(self.nuc, padding_pattern, mode='constant',
                              constant_values='*')
            self.bse = np.pad(self.bse, padding_pattern, mode='constant',
                              constant_values=0)
            self.mutation_pos = self.mutation_pos + padding_pattern[1][0]

            if change_ref:
                padding_pattern = self.calculate_padding_pattern(matrix_width,
                                                                 len(
                                                                     reference_seq))
                reference_seq = np.pad(np.array(reference_seq, dtype=object),
                                       padding_pattern[1],
                                       mode='constant',
                                       constant_values='*').tolist()
        elif c > matrix_width:
            s, e = self.calculate_start_end(matrix_width, c)
            self.nuc = self.nuc[:, s:e]
            self.bse = self.bse[:, s:e]
            self.mutation_pos = self.mutation_pos - s
            if change_ref:
                reference_seq = reference_seq[s:e]

        r, c = self.shape_of_one()
        empty_array = np.array(['*'] * c)
        indices = np.where(np.any(self.nuc != empty_array, axis=1))[0]
        if len(indices) < r:
            mats = IntermediaryMatrices.sample(self, indices)
            self.nuc = mats.nuc
            self.bse = mats.bse
            self.map = mats.map
            self.str = mats.str
            self.xtt = mats.xtt
            self.nmt = mats.nmt
            self.ast = mats.ast
            self.mdt = mats.mdt
            self.xst = mats.xst
            self.flg = mats.flg

        return reference_seq

    @staticmethod
    def calculate_padding_pattern(
            matrix_width: int,
            c: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calculate how the matrix should be padded when it's expanded
        horizontally.

        :param matrix_width: Width of the matrix after it will be expanded.
        :param c: Current width of the matrix.
        :return: A tuple of tuples showing padding pattern.
        """
        pad_width = matrix_width - c
        pad_width1, pad_width2 = (
            (int(pad_width / 2), int(pad_width / 2))
            if pad_width % 2 == 0
            else (int(pad_width / 2), int(pad_width / 2) + 1)
        )
        return (0, 0), (pad_width1, pad_width2)

    @staticmethod
    def calculate_start_end(
            matrix_width: int,
            c: int
    ) -> Tuple[int, int]:
        """Calculate the start and end position for when we want to truncate a
         matrix.

        :param matrix_width: Width of the matrix after it will be truncated.
        :param c: Current width of the matrix.
        :return: A tuple including start and end positions.
        """
        diff = c - matrix_width
        s, e = (
            (int(diff / 2), c - int(diff / 2))
            if diff % 2 == 0
            else (int(diff / 2) + 1, c - int(diff / 2))
        )
        return s, e

    @classmethod
    def sample(
            cls,
            inter_mats,
            indices: np.array
    ):
        """Sample matrices such that some of the rows(reads) are removed.

        Side effect: Converts list of lists into numpy arrays.

        :type inter_mats: IntermediateMatrices.
        :param inter_mats: IntermediateMatrices object to sample from.
        :param indices: Indices of rows which will be kept.
        :return: New IntermediateMatrices object with sampled reads.
        """
        obj = cls(inter_mats.ins_map, inter_mats.mutation_pos,
                  inter_mats.mutation_length)
        obj.nuc = inter_mats.nuc[indices, :]
        obj.bse = inter_mats.bse[indices, :]
        obj.map = np.array(inter_mats.map)[indices]
        obj.str = np.array(inter_mats.str)[indices]
        obj.xtt = np.array(inter_mats.xtt)[indices]
        obj.nmt = np.array(inter_mats.nmt)[indices]
        obj.ast = np.array(inter_mats.ast)[indices]
        obj.mdt = np.array(inter_mats.mdt)[indices]
        obj.xst = np.array(inter_mats.xst)[indices]
        obj.flg = np.array(inter_mats.flg)[indices]
        return obj

    @classmethod
    def downsample(
            cls,
            inter_mats,
            downsample_by: float
    ):
        """Downsample matrices such that some of the rows(reads) are removed.

        Side effect: Converts list of lists into numpy arrays.

        :type inter_mats: IntermediateMatrices.
        :param inter_mats: IntermediateMatrices object to sample from.
        :param downsample_by: Downsampling ratio [0.-1.]
        :return: New IntermediateMatrices object with downsampled reads.
        """
        if downsample_by == 1.0:
            return inter_mats
        r, c = inter_mats.shape_of_one()
        indices = sorted(random.sample(range(r), int(r * downsample_by)))
        return IntermediaryMatrices.sample(inter_mats, indices)

    def __repr__(self) -> Text:
        """Representation of the IntermediateMatrices object.

        :return: String summarizing the matrices & vectors.
        """
        return 'Intermediary matrices object at region {}:{}-{}'.format(
            self.region.chrom, self.region.start, self.region.end
        )


def unify_read(nuc_lst) -> List[Text]:
    """When a read has gap characters ('-') between empty characters ('*')
    convert them to '*'.

    :param nuc_lst: List of nucleotides.
    :return: List of nucleotides where no '-' is led or trailed by '*'.
    """
    is_first = True
    for i in range(1, len(nuc_lst) - 1):
        if (nuc_lst[i - 1] == '*' or nuc_lst[i + 1] == '*') \
                and nuc_lst[i] == '-':
            nuc_lst[i] = '*'
            if is_first:
                j = i - 1
                while j >= 0:
                    if nuc_lst[j] == '-':
                        nuc_lst[j] = '*'
                    else:
                        break
                    j -= 1
            is_first = False
    return nuc_lst


def generate_impure_mat(
        mat_t: IntermediaryMatrices,
        mat_n: IntermediaryMatrices,
        purity: float
):
    """Generate an impure tensor at the given purity in order to upsample data

    :param mat_t: Input IntermediateMatrices object for tumor
    :param mat_n: Input IntermediateMatrices object for normal
    :param purity: Tuple containing the ratio for tumor and normal.
    :return: A tensor that is impure.
    """
    if int(purity) == 1:
        return mat_t

    tumor_ratio, normal_ratio = purity, 1 - purity
    h_t, _ = mat_t.shape_of_one()
    h_n, _ = mat_n.shape_of_one()
    no_in_tumor = int(h_t * tumor_ratio)
    no_in_normal = int(h_n * normal_ratio)

    tumor_indices = sorted(random.sample(range(h_t), no_in_tumor))
    normal_indices = sorted(random.sample(range(h_n), no_in_normal))

    mat_t = IntermediaryMatrices.sample(mat_t, tumor_indices)
    mat_n = IntermediaryMatrices.sample(mat_n, normal_indices)

    mat_t.nuc = np.concatenate((mat_t.nuc, mat_n.nuc), axis=0)
    mat_t.bse = np.concatenate((mat_t.bse, mat_n.bse), axis=0)
    mat_t.map = np.concatenate((mat_t.map, mat_n.map), axis=0)
    mat_t.str = np.concatenate((mat_t.str, mat_n.str), axis=0)
    mat_t.xtt = np.concatenate((mat_t.xtt, mat_n.xtt), axis=0)
    mat_t.nmt = np.concatenate((mat_t.nmt, mat_n.nmt), axis=0)
    mat_t.ast = np.concatenate((mat_t.ast, mat_n.ast), axis=0)
    mat_t.mdt = np.concatenate((mat_t.mdt, mat_n.mdt), axis=0)
    mat_t.xst = np.concatenate((mat_t.xst, mat_n.xst), axis=0)
    mat_t.flg = np.concatenate((mat_t.flg, mat_n.flg), axis=0)

    return mat_t
