
all : clean test check

clean:
	rm -rf output
	rm -f .nextflow.log*
	rm -rf .nextflow*


test:
	nextflow main.nf --help
