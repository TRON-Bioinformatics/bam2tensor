TUMOR_BAM = 'tests/input/tumor.bam'
NORMAL_BAM = 'tests/input/normal.bam'
REFERENCE = 'tests/input/fasta.fa'
TUMOR_BAM_REAL = 'tests/big_input/tumor.bam'
NORMAL_BAM_REAL = 'tests/big_input/hg19_SORTED.fa'
REFERENCE_REAL = ''

CANDIDATES = 'tests/input/candidates.tsv'

AL_NOR1 = 'tests/expected_output/{}/alignment_nor1.txt'.format('compute_alignment')
AL_NOR2 = 'tests/expected_output/{}/alignment_nor2.txt'.format('compute_alignment')
AL_TUM1 = 'tests/expected_output/{}/alignment_tum1.txt'.format('compute_alignment')
AL_TUM2 = 'tests/expected_output/{}/alignment_tum2.txt'.format('compute_alignment')
BQ_NOR1 = 'tests/expected_output/{}/basequal_nor1.txt'.format('compute_alignment')
BQ_NOR2 = 'tests/expected_output/{}/basequal_nor2.txt'.format('compute_alignment')
BQ_TUM1 = 'tests/expected_output/{}/basequal_tum1.txt'.format('compute_alignment')
BQ_TUM2 = 'tests/expected_output/{}/basequal_tum2.txt'.format('compute_alignment')
MQ_NOR1 = 'tests/expected_output/{}/mapping_nor1.txt'.format('compute_alignment')
MQ_NOR2 = 'tests/expected_output/{}/mapping_nor2.txt'.format('compute_alignment')
MQ_TUM1 = 'tests/expected_output/{}/mapping_tum1.txt'.format('compute_alignment')
MQ_TUM2 = 'tests/expected_output/{}/mapping_tum2.txt'.format('compute_alignment')

AL_NOR1_BAM = 'tests/expected_output/{}/alignment_nor1.txt'.format('compute_alignment_for_bam')
AL_NOR2_BAM = 'tests/expected_output/{}/alignment_nor2.txt'.format('compute_alignment_for_bam')
AL_TUM1_BAM = 'tests/expected_output/{}/alignment_tum1.txt'.format('compute_alignment_for_bam')
AL_TUM2_BAM = 'tests/expected_output/{}/alignment_tum2.txt'.format('compute_alignment_for_bam')
BQ_NOR1_BAM = 'tests/expected_output/{}/basequal_nor1.txt'.format('compute_alignment_for_bam')
BQ_NOR2_BAM = 'tests/expected_output/{}/basequal_nor2.txt'.format('compute_alignment_for_bam')
BQ_TUM1_BAM = 'tests/expected_output/{}/basequal_tum1.txt'.format('compute_alignment_for_bam')
BQ_TUM2_BAM = 'tests/expected_output/{}/basequal_tum2.txt'.format('compute_alignment_for_bam')
MQ_NOR1_BAM = 'tests/expected_output/{}/mapping_nor1.txt'.format('compute_alignment_for_bam')
MQ_NOR2_BAM = 'tests/expected_output/{}/mapping_nor2.txt'.format('compute_alignment_for_bam')
MQ_TUM1_BAM = 'tests/expected_output/{}/mapping_tum1.txt'.format('compute_alignment_for_bam')
MQ_TUM2_BAM = 'tests/expected_output/{}/mapping_tum2.txt'.format('compute_alignment_for_bam')
INS_MAP_TUM1 = {3: 1, 5: 3, 12: 1, 14: 3}
INS_MAP_NOR1 = {5: 1, 13: 1}
INS_MAP_TUM2 = {0: 3, 7: 1, 9: 3}
INS_MAP_NOR2 = {0: 1, 8: 1}
