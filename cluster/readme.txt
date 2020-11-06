Generating peptide and protein similarity matrices to evaluate under cluster-based settings

Input: peptide fasta file & protein fasta file

Step1: Get aligment scores from fasta files separately from https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library 

usefule commands :
cd  ./smith-waterman-src/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD
python pyssw.py -p xxx.fasta xxx.fasta > xxx_output.txt  (get raw alignment scores)

Step2:
Use "sequence_similarity.py" to parse output and normalize alignment scores to get similarity matrices of peptides and proteins, respectively.


Step3:
Use functions in "cluster.py" to split peptides and proteins into clusters under different settings for cross validations