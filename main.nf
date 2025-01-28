#!/usr/bin/env nextflow

// DSL 2
nextflow.enable.dsl = 2

params.bam2tensor = false

params.help = false
params.input_files = false
params.window = false
params.reference = false
params.publish_dir = false
params.max_coverage = false
params.read_length = false
params.max_mapq = false
params.max_baseq = false


def helpMessage() {
    log.info"""
Usage:
    bam2tensor.nf --input_files input_files --reference reference.fasta --window 150

Input:
    * input_files: the path to a tab-separated values file containing in each row the sample name, sample type (tumor or normal) and path to the BAM file
    Sample type will be added to the BAM header @SN sample name
    The input file does not have header!
    Example input file (tab separated):
    sample1 dataset(train/valid/call) tma_bam1 nor_bam1 candidates1 labels1
    sample2 dataset(train/valid/call) tma_bam2 nor_bam2 candidates2 labels2
    ...
	* window: window around the candidate variant. Integer, e.g. 15, 50...
    * publish_dir: Output directory of the tensors
    * reference: path to the FASTA genome reference (indexes expected *.fai, *.dict)

Output:
    * Tensors
	"""
}

if (params.help) {
    helpMessage()
    exit 0
}

if (!params.reference) {
    exit 1, "--reference is required"
}
if (!params.window) {
    exit 1, "--window is required"
}
if (!params.input_files) {
  exit 1, "--input_files is required!"
}
if (!params.max_coverage) {
  exit 1, "--max_coverage is required!"
}
if (!params.read_length) {
  exit 1, "--read_length is required!"
}
if (!params.max_mapq) {
  exit 1, "--max_mapq is required!"
}
if (!params.max_baseq) {
  exit 1, "--max_baseq is required!"
}


process bam2tensor {
    cpus "${params.cpus}"
    memory "${params.memory}"
    publishDir "${params.publish_dir}", mode:"move"
    tag "${name}"

    conda (params.enable_conda ? 'environment.yml' : null)


    input:
	  tuple val(name), val(dataset), val(replicates), file(tma), file(tma_ind), file(nor), file(nor_ind), val(candidates)

    """

    python src/run.py \
    --out ${params.publish_dir} \
    --tumor_bam ${tma} \
    --normal_bam ${nor} \
    --replicate_pair ${replicates} \
    --reference ${params.reference} \
    --sample ${name} \
    --dataset ${dataset} \
    --window ${params.window} \
    --candidates_path ${candidates} \
    --purity 1.0 \
    --contamination 0.0 \
    --downsample_ratio 1.0 \
    --max_coverage ${params.max_coverage} \
    --read_length ${params.read_length} \
    --max_mapq ${params.max_mapq} \
    --max_baseq ${params.max_baseq} \
    --threads ${task.cpus}
    """
}

workflow {
    Channel.fromPath(params.input_files) \
        | splitCsv(header: ['name', 'dataset', 'replicates', 'tma', 'tmi', 'nor', 'noi', 'candidates'], sep: "\t") \
        | map{ row -> tuple(
            row.name,
            row.dataset,
            row.replicates,
            file(row.tma),
            file(row.tmi),
            file(row.nor),
            file(row.noi),
            file(row.candidates),
            )
        }  \
        | bam2tensor
}
