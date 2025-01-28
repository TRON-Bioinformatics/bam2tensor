import logging

logger = logging.getLogger(__name__)

import numpy as np
import pybedtools as pybed
import os
import time
import torch


def fill_freq_for_sample(
        tensor,
        al,
        i,
        max_baseq,
        max_mq,
        matrix_height,
        read_length,
):
    rt, ct = al.shape_of_one()
    tensor[i, :, :] = encode_for_frequency(al.nuc)
    tensor[i + 1, :, :] = encode_qualities(al.nuc, al.bse / max_baseq)
    tensor[i + 2, :, :] = encode_qualities(al.nuc, np.array(al.map).reshape(rt,
                                                                            1) / max_mq)
    neg_str, neg_cov, pos_str, pos_cov = separate_by_strand(al, matrix_height)
    tensor[i + 3, :, :] = neg_cov
    tensor[i + 4, :, :] = encode_for_frequency(neg_str)
    tensor[i + 5, :, :] = pos_cov
    tensor[i + 6, :, :] = encode_for_frequency(pos_str)
    al.nmt = np.reshape(al.nmt, (len(al.nmt), 1))
    tensor[i + 7, :, :] = encode_qualities(al.nuc, np.array(al.nmt))
    al.xtt = np.reshape(al.xtt, (len(al.xtt), 1))
    tensor[i + 8, :, :] = encode_qualities(al.nuc, np.array(al.xtt) == 'U')
    nuc = get_confident_reads(al.nuc, read_length)
    if len(nuc) > 0:
        tensor[i + 9, :, :] = encode_for_frequency(nuc)
        tensor[i + 10, :, :] = compute_coverage(nuc.shape[0], nuc,
                                                matrix_height)


def encode_for_frequency(nuc_mat):
    """Given a nucleotide matrix, calculate the frequencies at each position.

    :param nuc_mat: The nucleotide matrix.
    :return: Frequency matrix from nucleotide matrix, in shape 5, nuc_mat.shape[1]
    """
    r, c = nuc_mat.shape
    encoded_mat = np.zeros((5, c), dtype='float32')
    if r == 0:
        return encoded_mat
    encoded_mat[0, :] = np.count_nonzero(nuc_mat == '-', axis=0)
    encoded_mat[1, :] = np.count_nonzero(nuc_mat == 'A', axis=0)
    encoded_mat[2, :] = np.count_nonzero(nuc_mat == 'C', axis=0)
    encoded_mat[3, :] = np.count_nonzero(nuc_mat == 'G', axis=0)
    encoded_mat[4, :] = np.count_nonzero(nuc_mat == 'T', axis=0)
    normalizer = r - np.count_nonzero(nuc_mat == '*', axis=0) + 0.000001
    return encoded_mat / normalizer


def encode_qualities(nuc_mat, norm_mat):
    """Encode the base and mapping qualities based for each position with nuc. frequencies.

    :param nuc_mat: The nucleotide matrix.
    :param norm_mat: Normalized quality matrix
    :return: Frequency matrix normalized by the qualities.
    """
    r, c = nuc_mat.shape
    encoded_mat = np.zeros((5, c), dtype='float32')
    # TODO: should this not be c, and not r:
    norm_r = r - np.count_nonzero(nuc_mat == '*', axis=0) + 0.000001

    encoded_mat[1, :] = np.sum(np.multiply(nuc_mat == 'A', norm_mat), axis=0)
    encoded_mat[2, :] = np.sum(np.multiply(nuc_mat == 'C', norm_mat), axis=0)
    encoded_mat[3, :] = np.sum(np.multiply(nuc_mat == 'G', norm_mat), axis=0)
    encoded_mat[4, :] = np.sum(np.multiply(nuc_mat == 'T', norm_mat), axis=0)
    return encoded_mat / norm_r


def encode_ref_seq(reference_seq, col_no):
    """Given a reference sequence, encode it in a 5 by col_no matrix

    :param col_no: Number of columns in the output matrix
    :return: Encoded 5 by col_no reference sequence matrix
    """
    mat = np.zeros((5, col_no), dtype='float32')
    mat[0, np.array(reference_seq) == '-'] = 1
    mat[1, np.array(reference_seq) == 'A'] = 1
    mat[2, np.array(reference_seq) == 'C'] = 1
    mat[3, np.array(reference_seq) == 'G'] = 1
    mat[4, np.array(reference_seq) == 'T'] = 1
    return mat


def separate_by_strand(al, matrix_height):
    neg_str = al.nuc[np.array(al.str) == True, :]
    r, c = neg_str.shape
    neg_cov = compute_coverage(r, neg_str, matrix_height)
    pos_str = al.nuc[np.array(al.str) == False, :]
    r, c = pos_str.shape
    pos_cov = compute_coverage(r, pos_str, matrix_height)
    return neg_str, neg_cov, pos_str, pos_cov


def compute_coverage(r, nuc, matrix_height):
    coverage = (r - np.count_nonzero(nuc == '*', axis=0)) / matrix_height
    coverage[coverage > 1] = 1.
    return coverage


def get_confident_reads(nuc_arr, actual_read_length):
    """ Get reads where the variant position is around the mid section, scarce in INDELs

    :param nuc_arr: Numpy matrix where each row is a read
    :param read_length:
    :return:
    """
    confident_seqs = []
    r, cols = nuc_arr.shape
    halfway = int(cols / 2)
    not_accepted = int(actual_read_length / 10)
    max_gaps = halfway - not_accepted
    for row in nuc_arr:
        gaps_lead = int(np.count_nonzero(row[0:halfway] == '*'))
        gaps_trail = int(np.count_nonzero(row[-halfway:] == '*'))
        if gaps_lead <= max_gaps and gaps_trail <= max_gaps:
            confident_seqs.append(row)

    return np.array(confident_seqs)


def save_tensor(
        out,
        tensor,
        region: pybed.cbedtools.Interval,
        replicate,
        purity=1.,
        downsample_ratio=1.,
        contamination=0.,
        is_3D=True
):
    """Save the tensor to a file.

    :param out: Directory to save the output file.
    :param tensor: Tensor to be saved.
    """
    outfile_template = os.path.join(
        out, 'purity-{}-downsample-{}-contamination-{}', '{}-{}-{}-{}-{}-{}.pt'
    )

    chrom, cand_pos, type, diff = region[0], int(region[3]), region[4], abs(
        int(region[5]))
    if type == 'del':
        cand_pos -= 1
    outfile = outfile_template.format(
        purity,
        downsample_ratio,
        contamination,
        chrom,
        cand_pos,
        type,
        diff,
        replicate,
        time.time()
    )
    if is_3D:
        channels, height, width = tensor.shape
        tensor = get_3D_tensor(tensor, channels, height, width)
    torch.save(tensor, outfile)


def get_3D_tensor(arr, channels, h, w):
    if channels == 20:
        x1 = torch.zeros((9, 3, h, w), dtype=torch.float)
        x1[0, 0, :, :] = torch.Tensor(arr[0, :, :])
        x1[1, 0, :, :] = torch.Tensor(arr[1, :, :])
        x1[:, 1, :, :] = torch.Tensor(arr[2:11, :, :])
        x1[:, 2, :, :] = torch.Tensor(arr[11:20, :, :])
        return x1
    if channels == 24:
        x1 = torch.zeros((11, 3, h, w), dtype=torch.float)
        x1[0, 0, :, :] = torch.Tensor(arr[0, :, :])
        x1[1, 0, :, :] = torch.Tensor(arr[1, :, :])
        x1[:, 1, :, :] = torch.Tensor(arr[2:13, :, :])
        x1[:, 2, :, :] = torch.Tensor(arr[13:24, :, :])
        return x1
