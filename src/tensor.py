import sys
from typing import Text, Tuple

sys.path.append('./bam2tensor')
from intermediary_matrices import IntermediaryMatrices, generate_impure_mat
from tensor_helper import *

logger = logging.getLogger(__name__)

MIN_COV = 5


class Tensor:
    """
    Class for encapsulating a tensor. Takes the re-aligned matrices as input
    in the constructor and shapes them so that all of them are the same size.
    """

    def __init__(
            self,
            annotated_alignment: Tuple[Tuple, Tuple],
            window: int,
            max_coverage: int,
            read_length: int,
            max_mapq: int,
            max_baseq: int,
            purity: float = 1.,
            contamination: float = 0.,
            downsample_ratio: float = 1.,
    ):
        """Constructor for the tensor class.

        :param annotated_alignment: A tuple containing the intermediary
        matrices and the genomic region of interest.
        :param window: The window around the candidate that tensor should
        include.
        :param out_home: The path under which tensor will be saved
        :param max_coverage: Maximum expected coverage value without outliers.
        :param read_length: Expected average length of the aligned reads.
        :param max_mapq: Maximum mapping quality value in the alignment file.
        :param max_baseq: Maximum base quality value in the alignment file.
        :param purity: Synthetic tumor purity coefficient. [0., 1.]
        :param contamination: Synthetic normal contamination coefficient.
        [0., 1.]
        :param downsample_ratio: Synthetic downsampling coefficient. [0., 1.]
        :param from_hdf5: Whether the alignments are coming from an HDF5 file.
        """
        al, self.region = annotated_alignment
        self.matrix_width = window * 2 + 1
        self.matrix_height = max_coverage
        self.read_length = read_length
        self.window = window
        self.type = self.region[4]
        alt, aln, ref_seq = al

        alt = generate_impure_mat(alt, aln, purity)
        if contamination > 0.:
            aln = generate_impure_mat(aln, alt, 1. - contamination)
        alt = IntermediaryMatrices.downsample(alt, downsample_ratio)
        aln = IntermediaryMatrices.downsample(aln, downsample_ratio)
        if len(alt.nuc) < MIN_COV or len(aln.nuc) < MIN_COV:
            raise Exception(
                'Coverage lower than expected value. Will return nothing.'
            )

        self.coverage = self.matrix_height * 2

        self.reference_seq = ref_seq
        self.tensor = self.compute_alignment_tensor(
            alt, aln, max_mapq, max_baseq
        )

    def compute_alignment_tensor(self, alt, aln, max_mapq, max_baseq):
        """Take a matrix tuple, compress it and convert to a numpy tensor.

        :param alt: IntermediateMatrices object for tumor
        :param aln: IntermediateMatrices object for normal
        :param max_mapq: Maximum value the mapping quality can have
        :param max_baseq: Maximum value the base quality can have
        :return: A tensor that includes all the matrices
        """

        self.reference_seq = alt.scale_matrices_horizontal(
            self.reference_seq, self.matrix_width, self.window
        )
        aln.scale_matrices_horizontal(
            self.reference_seq, self.matrix_width, self.window,
            change_ref=False
        )

        return self.get_tensor_frequency(alt, aln, max_mapq, max_baseq)

    def get_tensor_frequency(self, alt, aln, max_mq, max_baseq):
        """Compute the tensor of frequencies similar to NeuSomatic

        Missing (compared to NeuSomatic):
            - Clipping information for reads supporting different bases
            - Paired-end information
            - Alignment score may not be calculated the same.

        :param alt: IntermediateMatrices object for tumor
        :param aln: IntermediateMatrices object for normal
        :param max_mq: Maximum value the mapping quality can have
        :param max_baseq: Maximum value the base quality can have
        :return: A tensor that includes all the matrices sized
        [channel, matrix_width, matrix_height]
        """
        channels = 24  # number of channels
        rt, ct = alt.shape_of_one()
        tensor = np.zeros((channels, 5, ct), dtype='float32')

        tensor[0, :, :] = encode_ref_seq(self.reference_seq, ct)  # reference
        tensor[1, :, alt.get_mutation_range(self.window, self.type)] = 1
        fill_freq_for_sample(
            tensor,
            alt,
            2,
            max_baseq,
            max_mq,
            self.matrix_height,
            self.read_length
        )
        fill_freq_for_sample(
            tensor,
            aln,
            13,
            max_baseq,
            max_mq,
            self.matrix_height,
            self.read_length
        )
        return tensor
