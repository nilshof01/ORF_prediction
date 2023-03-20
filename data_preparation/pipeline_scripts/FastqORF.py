#!/usr/bin/env python
# coding: utf-8


import sys
import os
import Bio
from Bio import SeqIO
from Bio import Seq
import re
import pandas as pd
import time
from collections import Counter

from itertools import zip_longest, islice

import numpy as np
from pydivsufsort import divsufsort, kasai


print("Run <pip install pydivsufsort> before", file=sys.stdout)

# INPUT
fq = str(sys.argv[1])
aa_cds = str(sys.argv[2])

#fq = "/net/node07/home/projects/metagnm_asm/1_gnm_man/snake_asm/trimmed/10e7/f40/f40_nodam_2000samples.fq"
#aa_cds = "/net/node07/home/projects/metagnm_asm/1_gnm_man/snake_sim/RMP/data/GCF_000155995.1_ASM15599v1_translated_cds.faa"

# bwt code start
"""bwt.py
Author: Kemal Eren
Efficient string search using the Burrows-Wheeler transform.
Lookup with find() is fast - O(len(query)) - but first the
Burrows-Wheeler data structures must be computed. They can be
precomputed via make_all() and provided to find() with the 'bwt_data'
argument.
The only slow part of make_all() is computing the suffix array. If
desired, the suffix array may instead be computed with a more
efficient method elsewhere, then provided to find() or make_all() via
the 'sa' argument.
"""

EOS = "\0"


def find(query, reference, mismatches=0, bwt_data=None, sa=None):
    """Find all matches of the string 'query' in the string
    'reference', with at most 'mismatch' mismatches.
    Examples:
    ---------
    >>> find('abc', 'abcabcabc')
    [0, 3, 6]
    >>> find('gef', 'abcabcabc')
    []
    >>> find('abc', 'abcabd', mismatches=1)
    [0, 3]
    >>> find('abdd', 'abcabd', mismatches=1)
    []
    """
    assert len(query) > 0

    if bwt_data is None:
        bwt_data = make_all(reference, sa=sa)
    alphabet, bwt, occ, count, sa = bwt_data
    assert len(alphabet) > 0

    if not set(query) <= alphabet:
        return []

    length = len(bwt)
    results = []

    # a stack of partial matches
    class Partial(object):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    partials = [Partial(query=query, begin=0, end=len(bwt) - 1,
                        mismatches=mismatches)]

    while len(partials) > 0:
        p = partials.pop()
        query = p.query[:-1]
        last = p.query[-1]
        letters = [last] if p.mismatches == 0 else alphabet
        for letter in letters:
            begin, end = update_range(p.begin, p.end, letter, occ, count, length)
            if begin <= end:
                if len(query) == 0:
                    results.extend(sa[begin : end + 1])
                else:
                    mm = p.mismatches
                    if letter != last:
                        mm = max(0, p.mismatches - 1)
                    partials.append(Partial(query=query, begin=begin,
                                            end=end, mismatches=mm))
    return sorted(set(results))


def make_sa(s):
    r"""Returns the suffix array of 's'
    Examples:
    ---------
    >>> make_sa('banana\0')
    [6, 5, 3, 1, 0, 4, 2]
    """  
    sa = divsufsort(s)
    sa_clean = []
    for i in sa:
        if i != " ":
            sa_clean.append(i)
            
            
    len_sa = len(sa_clean)
    len_s = len(s)
    if len_sa != len_s:
        print(len_sa, len_s)
    
    return sa_clean


def make_bwt(s, sa=None):
    r"""Computes the Burrows-Wheeler transform from a suffix array.
    Examples:
    ---------
    >>> make_bwt('banana\0')
    'annb\x00aa'
    """
    if sa is None:
        sa = make_sa(s)
    return "".join(s[idx - 1] for idx in sa)


def make_occ(bwt, letters=None):
    r"""Returns occurrence information for letters in the string
    'bwt'. occ[letter][i] = the number of occurrences of 'letter' in
    bwt[0 : i + 1].
    Examples:
    ---------
    >>> make_occ('annb\x00aa')['\x00']
    [0, 0, 0, 0, 1, 1, 1]
    >>> make_occ('annb\x00aa')['a']
    [1, 1, 1, 1, 1, 2, 3]
    >>> make_occ('annb\x00aa')['b']
    [0, 0, 0, 1, 1, 1, 1]
    >>> make_occ('annb\x00aa')['n']
    [0, 1, 2, 2, 2, 2, 2]
    """
    if letters is None:
        letters = set(bwt)
    result = {letter : [0] for letter in letters}
    result[bwt[0]] = [1]
    for letter in bwt[1:]:
        for k, v in result.items():
            v.append(v[-1] + (k == letter))
    return result


def make_count(s, alphabet=None):
    """Returns count information for the letters in the string 's'.
    count[letter] contains the number of symbols in 's' that are
    lexographically smaller than 'letter'.
    Examples:
    ---------
    >>> make_count('sassy') == {'a': 0, 'y': 4, 's': 1}
    True
    """
    if alphabet is None:
        alphabet = set(s)
    c = Counter(s)
    total = 0
    result = {}
    for letter in sorted(alphabet):
        result[letter] = total
        total += c[letter]
    return result


def update_range(begin, end, letter, occ, count, length):
    """update (begin, end) given a new letter"""
    newbegin = count[letter] + occ[letter][begin - 1] + 1
    newend = count[letter] + occ[letter][end]
    return newbegin, newend


def make_all(reference, sa=None, eos=EOS):
    """Returns the data structures needed to perform BWT searches"""
    alphabet = set(reference)
    assert eos not in alphabet
    count = make_count(reference, alphabet)

    reference = "".join([reference, eos])
    if sa is None:
        sa = make_sa(reference)
    bwt = make_bwt(reference, sa)
    occ = make_occ(bwt, alphabet | set([eos]))

    for k, v in occ.items():
        v.extend([v[-1], 0]) # for when pointers go off the edges

    return alphabet, bwt, occ, count, sa

# bwt code end


# Testing the suffix array package
#third = divsufsort("GYDYPDIQRAILA")
#print(third)
#new = []
#for i in third:
#    if i != " ":
#        new.append(i)
#print(new)



# function to extract all 6 reading frames
def extract_orfs(seq):
    rc_seq = seq.reverse_complement()
    frames = []
    for seqs in [seq, rc_seq]:
        for s in range(3):
            orf = seqs[s::]
            cutoff = len(orf) - len(orf)%3
            orf = orf[:cutoff]
            aa = orf.translate()
            frames.append((orf, aa))
    return tuple(frames)


# function to concatenate all protein sequences to a single string
def aa_to_string(aa_cds_path):
    aa_dic = SeqIO.index(aa_cds_path, "fasta")
    aa_ls = []
    for item in aa_dic:
        aa_ls.append(str(aa_dic[item].seq))
    aa_str = ":".join(map(str,aa_ls))
    return aa_str


# function to yield the result-list
def get_result(fq, aa_string, bwt_data, aa_dic):
    
    result_dic = {}
    
    with open(fq) as handle:
        for record in SeqIO.parse(handle, "fastq"):
            
            # Get all 6 reading frames per read
            orfs = extract_orfs(record.seq)

            # Initializing Matrix for each read: Matching ORF is indicated by 0 or 1
            orf_nums = [0,0,0,0,0,0] 
            # Iterating over reading frame indices and find matches in concatenated Amino Acid string from reference
            for i in range(0,6):
                aa_read = orfs[i][1]

                # Find matches via BWT algorithm (imported library)
                matches = find(aa_read, aa_string, mismatches=0, bwt_data=bwt_data)
                
                # append matches to results and modify match-matrix
                if len(matches) != 0:
                    orf_nums[i] += 1
                    ident = str(record.id) + ":" + str(i + 1)
                    result_dic[ident] = [record.id, str(aa_read), orf_nums]    
                
    return result_dic


# Initializing the reference Amino Acid sequences and concatenating all sequences to one string, separated by colon (Enables assignment of matchs to protein later if needed)
aa_dic = SeqIO.index(aa_cds, "fasta")
print("Creating the reference dictionary containing all AA-Seqs: Done", file=sys.stdout)
aa_string = aa_to_string(aa_cds)
print("Concatenating all AA-Seqs to a single string: Done", file=sys.stdout)

# Timer
t0 = time.time()
bwt_data = make_all(aa_string)
t1 = time.time()
total = t1 - t0
print("Time for generating suffix array of reference: ", total, file=sys.stdout)

print("Start matching process", file=sys.stdout)

t2 = time.time()
result = get_result(fq, aa_string, bwt_data, aa_dic)
t3 = time.time()

print("Time for matching: ", t3 - t2, file=sys.stdout)
print("Matching: Done", file=sys.stdout)

# Creating a dataframe with all results
df = pd.DataFrame.from_dict(result, orient = "index", columns = ["Read ID", "Read AA-Seq", "Match Matrix"])
df['Identifier'] = df.index
df = df.reset_index(drop = True)

df_g = df.sort_values(["Read ID"])
print("Number of matches: ", len(df_g), file=sys.stdout)

# Number of duplicates
ls = df["Read ID"].values.tolist()
x = np.array(ls)
unique = np.unique(x)
print("Number of unique matches (one ORF per read only): ", len(unique), file=sys.stdout)


df_g.to_csv('output_df.csv', sep='\t', header=None, index=False)


