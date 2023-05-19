import pandas as pd
import logomaker
from Bio.Seq import Seq
import os

def show_logo(path, show_aa = False, tp = True, reduce_to):
    path = os.path.normpath(path)
    reads = pd.read_csv(path, header = None)

    reads = reads.iloc[:, 0:4]
    reads.columns = ["empty", "sequence", "truepositive", "id"]
    reads['sequence'] = reads["sequence"].str.slice(stop=30)
    limit = reduce_to * 6
    reads[:limit, :]
    reads = reads.iloc[:limit, :]
    if tp == True:
        result = reads.loc[reads['truepositive'] == 1, 'sequence']
    else:
        result = reads.loc[reads["truepositive"] != 1, "sequence"]
    if show_aa == True:
        aa_seq = result.apply(lambda x: str(Seq(x).translate()))
        sequences = aa_seq
        molecules = "ACDEFGHIKLMNPQRSTVWY"
    else:
        sequences = result
        molecules = "ATGC"

        #local_report = local_report.sort_values("cloneFraction").groupby("Experiment", as_index = False).head(3) # filters out three highest values of each group


    chosen_sequence = sequences.iloc[0].__len__()
    compDict = {aa: chosen_sequence*[0] for aa in molecules}

    for seq in sequences:
        for aa_position in range(len(seq)):
            molecules = seq[aa_position]
            if molecules == '*':
                pass
            else:
                compDict[molecules][aa_position] += 1
    aa_distribution = pd.DataFrame.from_dict(compDict)
        #aa_distribution = aa_distribution.divide(aa_distribution.shape[1])
    aa_distribution = aa_distribution.divide(aa_distribution.sum(axis = 1), axis = 0)
    logo_plot = logomaker.Logo(aa_distribution,
                               shade_below=.5,
                               fade_below=.5,
                               font_name='Arial Rounded MT Bold',
                               )
    logo_plot.style_xticks(anchor=0,
                           spacing=1,
                           rotation=0)
