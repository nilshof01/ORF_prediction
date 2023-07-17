
#!/bin/bash

# Define the base directory where you want to search for files
WORKING_DIR=$(PWD)
BASE_DIR="/home/databases/refseq_bac/refseq/bacteria"
num_frags=6000
length=33
min_length=18
size_dist="/home/projects/metagnm_asm/pestis_PRJEB46734/insize/pestis.insize"
fragsim="/home/ctools/gargammel/src/fragSim"
art_illumina="/home/ctools/gargammel/art_src_MountRainier_Linux/art_illumina"
leeHom="/home/ctools/leeHom-1.2.15/src/leeHom"
seqkit="/home/ctools/seqkit-2.2.0/seqkit"
fastqMatches="$WORKING_DIR/src/data_preparation/pipeline_scripts/FastqMatches.py"
fastq_ORF="$WORKING_DIR/src/data_preparation/pipeline_scripts/FastqORF.py"
create_frames="$WORKING_DIR/src/data_preparation/pipeline_scripts/create_frames_real.py"
trim_frames="$WORKING_DIR/src/data_preparation/pipeline_scripts/trim_frames.py"
max_dirs=35000 # max 36757


if [ ! -d $WORKING_DIR/temp ]; then
	mkdir -p $WORKING_DIR/temp
fi
count=0
temp_dir=$WORKING_DIR/temp
echo $temp_dir
# Loop through all the directories in the base directory
for dir in $BASE_DIR/*/; do


    
    if [ $count -ge $max_dirs ]; then
        break
    fi
    # Enter the directory
    cd "$dir"
    # Look for the fn.gz file
    echo "$dir"

    input=$(find . -type f -name "*genomic.fna" | grep -v "cds*")
    ID=$(basename "$input" .fna)
    echo $ID
    if ls /home/people/s220672/ReadsMatchProtein/results/6000frags/ | grep $ID; then
      found=true
    else
      found=false
    fi
    echo "Found substring: $found"
    file=$(find . -type f -name "*genomic.fna.gz" | grep -v "cds*")
    echo $file
    gff_file=$(find . -type f -name "*genomic.gff.gz")
    echo $gff_file
    cds_seq=$(find . -type f -name "*protein.faa.gz")
    echo $cds_seq
    input_dirty=$dir$file
    input=$(realpath "$input_dirty")
    echo $input

    #gff_dirty=$dir$gff_file
  # # gff_true=$(realpath "$gff_dirty")
   # cds_dirty=$dir$cds_seq
    #cds_true=$(realpath "$cds_dirty")
    
    
    # Check if the file exists and output does not exist
    if [ -f "$file" ] && [ -f "$gff_file" ] && [ -f "$cds_seq" ] && found=false; then
    	
        # Unzip the file
        
#        gunzip "$file"
        # Declare the unzipped file as input
        #input=$file
        #input=$(basename "$file" fna.gz)
	# index input
         gunzip $input

         gunzip $cds_seq
         cds_seq=$(find . -type f -name "*protein.faa")
         input=$(find . -type f -name "*genomic.fna" | grep -v "cds*")
         ID=$(basename "$input" .fna)
         echo $ID
         samtools faidx $input
        # Run your command with the input file
        
    	$fragsim -n ${num_frags} -l ${length} ${input} > $temp_dir/frag_out.fa
	$art_illumina -i ${temp_dir}/frag_out.fa --len ${length} -o ${temp_dir}/ --insRate 0 --insRate2 0 -dr 0 -dr2 0 --minQ 38 --seqSys HS25 --rcount 1 --paired --amplicon --noALN --quiet 
    	$leeHom --ancientdna -t 50 -fq1 "${temp_dir}/1.fq" -fq2 "$temp_dir/2.fq" -fqo $temp_dir/merge_read_out

	$seqkit rmdup $temp_dir/merge_read_out.fq.gz > ${WORKING_DIR}/temp/rm_dups.fq 
	python $fastqMatches ${temp_dir}/rm_dups.fq $gff_file
	mv out.fq ${temp_dir}
	python $fastq_ORF ${temp_dir}/out.fq $cds_seq

	python $create_frames $WORKING_DIR/output_df.csv $temp_dir/rm_dups.fq ${WORKING_DIR}/results/ $ID
	python $trim_frames ${WORKING_DIR}/results/$ID.fq $length $min_length $ID $WORKING_DIR/results/
   	gzip $input
   	gzip $cds_seq
	rm -r ${WORKING_DIR}/results/*.fq
	rm fq_interval_matches.csv
	rm ref_gff.bed
 	rm ref_gff.csv
	rm -r ${WORKING_DIR}/temp/*
  
	rm fastq.bed					
	count=$((count+1))
    else
        # Print an error message if the file does not exist
        echo "File not found in directory: $dir"
        
    fi
    cd ..

done



