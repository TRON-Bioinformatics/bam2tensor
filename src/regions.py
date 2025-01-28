import logging
import pandas as pd
import pybedtools as pybed
from typing import List, Text, Tuple

logger = logging.getLogger(__name__)


def get_candidates(candidate_path: Text, purity: float = 1.,
                   downsample: float = 1.):
    """Get the list of candidates with relevant information;
    chrom, pos, ref, and alt.

    :param candidate_path: Path to the files containing candidate variants.
    :param is_single: Whether the bams are matched or not. (False if matched)
    :return: A list of candidates.
    """
    df = pd.read_csv(candidate_path, sep='\t')

    # only when training: augment the examples so that the distributions
    # are similar to each other. purity and downsample mixtures are not to
    # be used during validation/test stages - so here we can look at the labels
    if purity != 1. or downsample != 1.:
        df_synt = df.copy()
        df_synt.loc[df_synt['FILTER'].isna(), 'FILTER'] = 'unknown'
        labels = df_synt['FILTER'].unique()
        if ('somatic' not in labels) and ('consensus' not in labels):
            raise Exception(
                'Candidate file not formatted correctly for generating '
                'purity/downsampling mixes. It should contain a column named '
                '"FILTER" which includes "somatic" or "consensus" labels'
                'Please beware that purity/downsampling augmentation is only '
                'appropriate for training data as the somatic class is '
                'augmented intentionally.'
            )
        vaf_field = 'primary_af'
        depth_field = 'primary_dp'
        ac_field = 'primary_ac'

        normal_vaf_field = 'normal_af'
        normal_ac_field = 'normal_ac'

        df_synt[vaf_field] = df_synt[vaf_field] * purity
        df_synt[depth_field] = df_synt[depth_field] * downsample
        df_synt = df_synt[df_synt[depth_field] >= 7]

        label_groups = []
        for gr in df_synt.groupby('FILTER'):
            dfg = gr[1].copy()
            if gr[0] in ['somatic', 'consensus']:
                dfg = dfg[dfg[ac_field] > 3]
                dfg = dfg[dfg[vaf_field] > 0.01]
                dfg = dfg[
                    (
                            (dfg[vaf_field] < 0.1) &
                            (dfg[depth_field] > 100)
                    ) |
                    (
                            (dfg[vaf_field] >= 0.1) &
                            (dfg[vaf_field] < 0.2) &
                            (dfg[depth_field] >= 35)
                    ) |
                    (
                            (dfg[vaf_field] >= 0.2) &
                            (dfg[vaf_field] < 0.7) &
                            (dfg[depth_field] >= 10)
                    ) |
                    (
                            (dfg[vaf_field] >= 0.7) &
                            (dfg[vaf_field] <= 1.)
                    )
                    ]
                label_groups.append(dfg)
            elif gr[0] in ['unknown']:
                dfg = dfg[~(
                        (dfg[vaf_field] > 0.3) &
                        (dfg[ac_field] > 10) &
                        (dfg[normal_vaf_field] < 0.01) &
                        (dfg[normal_ac_field] < 2)
                )]
                label_groups.append(dfg)
            elif gr[0] in ['snp', 'no mutation']:
                label_groups.append(dfg)

        df = pd.concat(label_groups).drop_duplicates().reset_index(drop=True)

    cands = []
    for _, r in df.iterrows():
        try:
            cands.append((r['CHROM'], r['POS'], r['REF'], r['ALT']))
        except:
            cands.append((r['#CHROM'], r['POS'], r['REF'], r['ALT']))
    logger.info('Number of candidates: {}'.format(len(cands)))
    return cands


def compute_regions(
        candidates: List[Tuple], window: int
) -> pybed.BedTool:
    """Compute the the windows around the candidates

    :param candidates: A list of candidates containing tuples.
    For matched tumor-normal: chromosome, position, reference , alternative.
    For only tumor (for SVs): chromosome, position, , , size, orientation,
    is_first_breakpoint, structural variant type.
    :param window: The length of the window to be explored around a candidate.
    :param is_single: Whether it's matched tumor-normal or only tumor.
    :return: Windows around all the candidates.
    """
    windows = []
    for cand in candidates:
        chrom, start, end, pos, muttype, diff = get_metadata_paired(
            cand, window
        )
        windows.append(
            '{}\t{}\t{}\t{}\t{}\t{}'.format(
                chrom, start, end, pos, muttype, diff
            )
        )

    windows_bed = pybed.BedTool('\n'.join(windows), from_string=True)

    return windows_bed


def get_metadata_paired(cand: Tuple, window: int) -> Tuple:
    """ Get metadata for candidates in matched tumor-normal sequences.

    :param cand: Candidate tuple with chrom, pos, ref , alt.
    :param window: Window around the candidate that will be included in tensors
    :return: chrom, start, end, pos, muttype, diff
    """
    pos = cand[1]
    if cand[2] != 'N':
        diff = len(cand[3]) - len(cand[2])
    else:
        diff = 0
    muttype = 'point'
    if diff < 0:
        muttype = 'del'
        pos = pos + 1
    elif diff > 0:
        muttype = 'ins'
    start = pos - window
    end = pos + window
    if start < 0:
        start = 1
    return cand[0], start, end, pos, muttype, diff
