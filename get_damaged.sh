#!/bin/bash 


WORKING_DIR="/home/projects/metagnm_asm/nils"
all_csvs="/net/domus/home/people/s220672/ReadsMatchProtein/results/6000frags_63nt"
size_dist="/home/projects/metagnm_asm/pestis_PRJEB46734/insize/pestis63.insize"
dam_high="/home/projects/metagnm_asm/pestis_PRJEB46734/sim/damage/dhigh"
dam_mid="/home/projects/metagnm_asm/pestis_PRJEB46734/sim/damage/dmid"

deamsim="/home/ctools/gargammel/src/deamSim"
random_trim="/home/projects/metagnm_asm/nils/scripts/random_trim.py"
get_deam_csv="/home/projects/metagnm_asm/nils/scripts/rev_comp.py"

if [ ! -d "${WORKING_DIR}/tmps" ]; then 
        mkdir "${WORKING_DIR}/tmps";
fi

if [ ! -d "${WORKING_DIR}/dresults" ]; then 
        mkdir "${WORKING_DIR}/dresults";
fi

input=$1
#for input in ${all_csvs}/*csv; do
	name="$(basename "${input}" .csv)"
	fasta_file="${WORKING_DIR}/tmps/${name}.fasta"

	line_number=0
	while IFS=, read -r _ sequence _ header; do
		if (( (line_number) % 6 == 0 )); then
			echo ">${header}" >> "${fasta_file}"
			echo "${sequence}" >> "${fasta_file}"
		fi
		((line_number++))
	done < "${input}"

	python $random_trim ${fasta_file} "${WORKING_DIR}/tmps/${name}_trim.fa"

	$deamsim -matfile ${dam_high} "${WORKING_DIR}/tmps/${name}_trim.fa" > "${WORKING_DIR}/tmps/${name}_dhigh.fa"
	$deamsim -matfile ${dam_mid} "${WORKING_DIR}/tmps/${name}_trim.fa" > "${WORKING_DIR}/tmps/${name}_dmid.fa"

	python $get_deam_csv "${WORKING_DIR}/tmps/${name}_dhigh.fa" "${WORKING_DIR}/tmps/${name}_dhigh.csv"
	python $get_deam_csv "${WORKING_DIR}/tmps/${name}_dmid.fa" "${WORKING_DIR}/tmps/${name}_dmid.csv"

	for file in tmps/*{_dhigh,_dmid}.csv; do
		# Process each file here
		dname="$(basename "${file}" .csv)"
		out="$WORKING_DIR/dresults/${dname}_fi.csv"
		
		# Read lines from the first CSV file
		exec 5< "$input"
		exec 6< "$file"

		# Read and compare rows
		while IFS=',' read -r -u 5 -a row1 && IFS=',' read -r -u 6 -a row2; do
			# Compare rows
			echo "${row1[0]},${row2[0]},${row1[2]},${row1[3]}" >> "$out"
		done
		
		# Close the file descriptors
		exec 5<&-
		exec 6<&-
	done

	rm -rf "${WORKING_DIR}/tmps"

#done
