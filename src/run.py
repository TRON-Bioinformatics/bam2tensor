import logging
import warnings
from functools import partial

import fire
import os
from typing import List, Text

from main import main, execute_async

from regions import get_candidates, compute_regions

FORMAT = '%(levelname)s %(asctime)-15s %(name)-20s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class Params:
    """Generate tensors as input to neural network models"""

    def __init__(
            self,
            out: Text,
            tumor_bam: Text,
            normal_bam: Text,
            replicate_pair: Text,
            reference: Text,
            candidates_path: Text,
            sample: Text,
            dataset: Text,
            labels_path: Text = None,
            purity: float = 1.,
            contamination: float = 0.,
            downsample_ratio: float = 1.,
            read_length: int = 50,
            window: int = 150,
            max_coverage: int = 500,
            max_mapq: int = 60,
            max_baseq: int = 82,
            threads: int = 1,
    ):
        """ Generate input tensors to the neural network.

        :param out: Output directory.
        :param tumor_bam: Path to the tumor BAM file.
        :param normal_bam: Path to the normal BAM file.
        :param reference: Path to the reference genome.
        :param candidates_path: Path to the candidates file.
        :param labels_path: Path to the labels file.
        :param sample: Name of the sample.
        :param dataset: The dataset the sample should be in. One of
        train/valid/call.
        :param purity: Simulated tumor purity ratio. In range [0.1, 1.0],
        1 means the tumor sample contains only tumor reads.
        :param contamination: Simulated normal contamination ratio. In range
        [0.0, 0.99]. 0 means the normal sample contains only normal reads.
        :param downsample_ratio: Simulated downsampling ratio. In range
        [0.1, 1.0]. 1 means no downsampling
        :param read_length: The read length (from sequencing)
        :param window: The length of the window around the candidate variant.
        :param max_coverage: Maximum coverage value (excluding outliers).
        Do not change unless you are retraining the network.
        :param max_mapq: Maximum mapping quality value.
        :param max_baseq: Maximum base quality value.
        :param threads: Number of threads. Too many threads may lead the
        execution to halt.
        """
        self.sample = sample
        self._set_dataset(dataset)
        self.tumor_bam = self._get_file(tumor_bam)
        self.normal_bam = self._get_file(normal_bam)
        self.replicate_pair = replicate_pair
        self.reference = self._get_file(reference)
        self.candidates_path = self._get_file(candidates_path)
        if labels_path is not None:
            self.labels_path = self._get_file(labels_path)

        self._set_up_folder_structure(
            out, sample, window, dataset, candidates_path, labels_path
        )
        self.purity = self._get_value(purity, 'purity', 0.1, 1.0)
        self.contamination = self._get_value(
            contamination, 'contamination', 0.0, 0.99
        )
        self.downsample_ratio = self._get_value(
            downsample_ratio, 'downsample_ratio', 0.1, 1.0
        )
        self.read_length = self._get_value(read_length, 'read_length', 5, 5000)
        self.window = self._get_value(window, 'window', 5, 5000)
        self.max_coverage = self._get_value(
            max_coverage, 'max_coverage', 11, 10000
        )
        self.max_mapq = self._get_value(max_mapq, 'max_mapq', 10, 255)
        self.max_baseq = self._get_value(max_baseq, 'max_baseq', 10, 255)
        self.threads = self._get_value(threads, 'threads', 1, 32)
        self.generate_tensors()

    def generate_tensors(self):
        logger.info(self.__repr__())
        subfolder = os.path.join(
            self.out_path, 'purity-{}-downsample-{}-contamination-{}'.format(
                self.purity, self.downsample_ratio, self.contamination
            )
        )
        os.makedirs(subfolder, exist_ok=True)
        candidates = get_candidates(
            self.candidates_path,
            self.purity,
            self.downsample_ratio
        )
        regions = compute_regions(candidates, self.window)

        func = partial(
            main,
            self.tumor_bam,
            self.normal_bam,
            self.replicate_pair,
            self.reference,
            self.window,
            self.out_path,
            self.purity,
            self.contamination,
            self.downsample_ratio,
            self.max_coverage,
            self.read_length,
            self.max_mapq,
            self.max_baseq,
        )
        execute_async(func, regions, self.threads)

    def _set_up_folder_structure(
            self, out, sample, window, dataset, candidates, labels
    ):
        home = os.path.join(out, dataset, sample)
        self.out_path = os.path.join(home, 'freq{}'.format(str(window)))
        os.makedirs(self.out_path, exist_ok=True)
        candidates_path_sym = os.path.join(home, 'candidates.tsv')
        try:
            os.symlink(candidates, candidates_path_sym)
        except FileExistsError:
            candidates_path_real = os.path.realpath(candidates_path_sym)
            if candidates_path_real != os.path.realpath(candidates):
                warnings.warn(
                    'Candidate file in the current directory is not the same '
                    'as the given one. Please verify that this is intentional.'
                )
        if labels is not None:
            labels_path_sym = os.path.join(home, 'labels.tsv')
            try:
                os.symlink(labels, labels_path_sym)
            except FileExistsError:
                labels_path_real = os.path.realpath(labels_path_sym)
                if labels_path_real != os.path.realpath(labels):
                    warnings.warn(
                        'Labels file in the current directory is not the same '
                        'as the given one. Please verify that this is '
                        'intentional.'
                    )
        elif dataset == 'call':
            return
        else:
            raise Exception(
                'Labels path not given in training/validation dataset'
            )

    def _get_file(self, f):
        if not os.path.exists(f):
            raise Exception('Path to the file does not exist: {}'.format(f))
        return f

    def _get_value(self, v, name, range_start, range_end):
        if v < range_start or v > range_end:
            raise Exception(
                '{} given for {} value. It should have been in range '
                '[{}, {}]'.format(v, name, range_start, range_end)
            )
        return v

    def _set_dataset(self, dataset):
        if dataset not in ['train', 'valid', 'call']:
            raise Exception(
                'Dataset type not supported. Should be one of train/valid/call'
            )
        self.dataset = dataset

    def __repr__(self):
        return 'Inputs to bam2tensor:\n{}'.format(
            '\n'.join([
                'tumor_bam            ' + self.tumor_bam,
                'normal_bam           ' + self.normal_bam,
                'reference            ' + self.reference,
                'sample               ' + self.sample,
                'window               ' + str(self.window),
                'output_path          ' + self.out_path,
                'candidates_path      ' + self.candidates_path,
                'purity               ' + str(self.purity),
                'downsample_ratio     ' + str(self.downsample_ratio),
                'coverage_upper_bound ' + str(self.max_coverage),
                'read_length          ' + str(self.read_length),
                'max_mapq             ' + str(self.max_mapq),
                'max_baseq            ' + str(self.max_baseq),
                'threads              ' + str(self.threads),
            ])
        )


if __name__ == '__main__':
    fire.Fire(Params)
