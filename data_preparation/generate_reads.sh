
#!/bin/bash

# Define the base directory where you want to search for files
WORKING_DIR=$(pwd)
BASE_DIR="/home/databases/refseq_bac/refseq/bacteria"
num_frags = 1000
length = 150
size_dist="/home/projects/metagnm_asm/pestis_PRJEB46734/insize/pestis.insize"
fragsim = "/home/ctools/gargammel/src/fragSim"
art_illumina = "/home/ctools/gargammel/art_src_MountRainier_Linux/art_illumina"
leeHom = "/home/ctools/leeHom-1.2.15/src/leeHom"
seqkit = "/home/ctools/seqkit-2.2.0/seqkit"
fastqMatches = "$WORKING_DIR/pipeline_scripts/FastqMatches.py"
fastq_ORF = "$WORKING_DIR/pipeline_scripts/FastqORF.py"
create_frames = "$WORKING_DIR/pipeline_scripts/create_frames.py"
count = 0
if [ ! -d $WORKING_DIR/temp ]; then
	mkdir -p $WORKING_DIR/temp
fi

temp_dir = $WORKING_DIR/temp
# Loop through all the directories in the base directory
for dir in $BASE_DIR/*/; do

    # Enter the directory
    cd "$dir"
    # Look for the fn.gz file
    file=$(find . -name "*.fn.gz" -type f)
    gff_file = $(find . -name "*genomic.gff" -type f)
    cds_seq = $(find . -name "*protein.fna" -type f)
    # Check if the file exists
    if [ -f "$file" ]; then
    	ID = $(basename "$file")
        # Unzip the file
        gunzip "$file"
        # Declare the unzipped file as input
        input=$(basename "$file" .gz)
	# index input
	samtools index $input
        # Run your command with the input file
        $fragsim -n $num_frags -l $length $input > $temp_dir/frag_out.fa
	$art_illumina -i $temp_dir/frag_out.fa --len $length -o $temp_dir/ --insRate 0 --insRate2 0 -dr 0 -dr2 0 --minQ 38 --seqSys HS25 --rcount 1 --paired --amplicon --noALN --quiet 
        $leeHom --ancientdna -t 4 -fq1 $temp_dir/1.fq -fq2 $temp_dir/2.fq -fqo $temp_dir/merge_read_out
	gzip -d $temp_dir/merged_read_out.fq.gz
	$seqkit rmdup $temp_dir/merge_read_out.fq > temp/rm_dups.fq 
	python $fastqMatches $temp_dir/rm_dups.fq $gff_file
	mv out.fq $temp_dir
	python $fastq_ORF $temp_dir/out.fq $cds_seq
	python $create_frames output.csv $temp_dir/merged_reads.fq.gz "$WORKING_DIR/results/" $ID
	rm fq_interval_matches.csv
	rm ref_gff.bed
 	rm ref_gff.csv
	rm -r temp/*
	rm fastq.bed					
    else
    
        # Print an error message if the file does not exist
        echo "File not found in directory: $dir"
        
    fi
	count=$((count+1))
done

