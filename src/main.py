import multiprocessing
import pybedtools as pybed
import traceback
from typing import Text, Tuple

import sys


sys.path.append('./bam2tensor')

from aligner import compute_alignment
from tensor import Tensor
from tensor_helper import save_tensor


import logging

logger = logging.getLogger(__name__)


def main(
        tumor_bam: Text,
        normal_bam: Text,
        replicate_pair: Text,
        reference: Text,
        window: int,
        out_path: Text,
        purity: float,
        contamination: float,
        downsample_ratio: float,
        max_coverage: int,
        read_length: int,
        max_mapq: int,
        max_baseq: int,
        region: pybed.cbedtools.Interval,
):
    """Compute the tensor and save to the disk.

    :param tumor_bam: Path to the tumor BAM file.
    :param normal_bam: Path to the tumor BAM file.
    :param reference: Path to the reference genome.
    :param window: Length of the window around a candidate
    :param out_path: Path to the output directory
    :param purity: Synthetic tumor purity coefficient. [0., 1.]
    :param contamination:  Synthetic normal contamination coefficient. [0., 1.]
    :param downsample_ratio: Synthetic downsampling coefficient. [0., 1.]
    :param max_coverage: Approximate upper bound for coverage (without outliers)
    :param read_length: Average/expected length of the aligned reads.
    :param max_mapq: Maximum mapping quality value in the alignment file.
    :param max_baseq: Maximum base quality value in the alignment file.
    :param region: Interval object for the genomic region of interest
    """
    al = compute_alignment(tumor_bam, normal_bam, reference, region, window)
    if is_alignment_successful(al):
        tensor = generate_and_save_tensors(
            al=al,
            region=region,
            replicate=replicate_pair,
            window=window,
            out_path=out_path,
            purity=purity,
            contamination=contamination,
            downsample_ratio=downsample_ratio,
            max_coverage=max_coverage,
            read_length=read_length,
            max_mapq=max_mapq,
            max_baseq=max_baseq,
        )
        return tensor


def generate_and_save_tensors(
        al: Tuple,
        region: pybed.cbedtools.Interval,
        replicate: Text,
        window: int,
        out_path: Text,
        purity: float,
        contamination: float,
        downsample_ratio: float,
        max_coverage: int,
        read_length: int,
        max_mapq: int,
        max_baseq: int
):
    """Generate the final tensors and save as .pt objects on disk.

    :param al: Tuple containing aligned tumor, normal and reference matrices.
    :param region: Interval object for the genomic region of interest
    :param window: Length of the window around a candidate
    :param out_path: Path to the output directory
    :param purity: Synthetic tumor purity coefficient. [0., 1.]
    :param contamination:  Synthetic normal contamination coefficient. [0., 1.]
    :param downsample_ratio: Synthetic downsampling coefficient. [0., 1.]
    :param max_coverage: Approximate upper bound for coverage (without outliers)
    :param read_length: Average/expected length of the aligned reads.
    :param max_mapq: Maximum mapping quality value in the alignment file.
    :param max_baseq: Maximum base quality value in the alignment file.
    """
    try:
        tensor = Tensor(
            (al, region),
            window=window,
            purity=purity,
            contamination=contamination,
            downsample_ratio=downsample_ratio,
            max_coverage=max_coverage,
            read_length=read_length,
            max_mapq=max_mapq,
            max_baseq=max_baseq,
        )
        if tensor.tensor is not None:
            save_tensor(
                out_path,
                tensor.tensor,
                region,
                replicate,
                purity,
                downsample_ratio,
                contamination
            )
            return tensor
    except:
        traceback.print_exc()
        logger.info('Couldn\'t generate tensor for region: {}'.format(region))


def is_alignment_successful(al: Tuple) -> bool:
    """ Check if the re-alignment with insertions successfully completed.

    Reasons for failure are usually related to absence of reads, either both in
    tumor/normal, or only normal. If the reads are absent in the tumor, an
    alignment is still returned in case it shows a large deletion

    :param al: Tuple containing IntermediaryMatrices and reference sequence.
    For matched: Tuple[IntermediaryMatrices, IntermediaryMatrices, Text]
    For unmatched: Tuple[IntermediaryMatrices,  Text]
    :return: Boolean value on whether alignment was successful or failed.
    """
    return al and not (al[0].is_empty() and (len(al) == 2 or al[1].is_empty()))


def execute_async(method, map_args, num_threads):
    """Asynchronously execute a method using map_args.

    :param method: Method to execute
    :param map_args: Arguments to the function (as an iterable???)
    :param num_threads: Number of threads
    :return: The output of the function
    """
    pool = multiprocessing.Pool(num_threads)
    try:
        pool.map_async(method, map_args).get()
    except Exception as inst:
        traceback.print_exc()
        logger.error(inst)
    finally:
        pool.close()
