# bam2tensor


Toolbox to convert BAM files into tensors

## Installation

Download this repository, go to the directory it resides and run:

```
git clone https://github.com/TRON-Bioinformatics/bam2tensor.git
cd bam2tensor
pip install -e .
```

## Requirements

* Python 3.9+
* Packages listed under environment.yml
* The required libraries can be found under `setup.cfg` and are automatically installed when you install this package as shown above.

## Running the nextflow pipeline

### Usage:
```
    nextflow run tron-bioinformatics/bam2tensor
    -profile conda \
    --input_files input_files \
    --publish_dir out_dir \
    --reference genome_ref.fa \
    --window 150 \
    --max_coverage 500 \
    --read_length 50 \
    --max_mapq 60 \
    --max_baseq 82
```

### Input:

* input_files: the path to a tab-separated values file containing in each row the sample name and a BAM file

    The input file does not have a header!

    Example input file:

    name1	tumor_bam1  tumor_bai1  normal_bam1 normal_bai1 candidates1.tsv

    name2	tumor_bam2  tumor_bai2  normal_bam2 normal_bai2 candidates2.tsv

* reference: the reference genome

* window: length of the window to be included around the variant

* max_coverage: Maximum coverage value to normalize coverage matrices

* read_length: The length of majority of the reads in BAM

* max_mapq: Maximum mapping quality to normalize mapping quality matrices, values indicating unknown mapping quality is ignored

* max_baseq: Maximum base quality to normalize base quality matrices, values indicating unknown base quality is ignored


### Optional input:

* publish_dir: the folder where to publish output

* memory: the ammount of memory used by each job (default: 15g)

* cpus: the number of CPUs used by each job (default: 8)


### Output:

* Tensors under the output folder


