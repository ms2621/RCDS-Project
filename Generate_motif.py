'''
Generate possible isoforms for each motif
'''

def generate_bre():
    comb_bre = []  # different isoforms of one motif

    for m in ['G', 'C']:
        seq1 = ''
        seq1 += m
        for n in ['G', 'C']:
            seq2 = seq1
            seq2 += n
            for o in ['G', 'A']:
                seq3 = seq2
                seq3 += (o + 'CGCC')
                comb_bre.append(seq3)
    return comb_bre


def generate_dpe():
    comb_dpe = []  # different isoforms of one motif

    for m in ['A', 'G']:
        seq1 = ''
        seq1 += (m + 'G')
        for n in ['A', 'T']:
            seq2 = seq1
            seq2 += n
            for o in ['C', 'T']:
                seq3 = seq2
                seq3 += o
                for p in ['G', 'A', 'C']:
                    seq4 = seq3
                    seq4 += p
                    comb_dpe.append(seq4)
    return comb_dpe


def generate_inr():
    comb_inr = []  # different isoforms of one motif

    for m in ['C', 'T']:
        seq1 = ''
        seq1 += m
        for n in ['C', 'T']:
            seq2 = seq1
            seq2 += (n + 'A')
            for o in ['A', 'C', 'G', 'T']:
                seq3 = seq2
                seq3 += o
                for p in ['T', 'A']:
                    seq4 = seq3
                    seq4 += p
                    for q in ['C', 'T']:
                        seq5 = seq4
                        seq5 += q
                        for r in ['C', 'T']:
                            seq6 = seq5
                            seq6 += r
                            comb_inr.append(seq6)
    return comb_inr


def count_motif(seq_letters, seq_numbers=[]):
    # TATA
    start_index = 0
    count = 0
    string = 'TATAAA'  # modify the motif here
    str_len = len(string)  # length of the motif
    while seq_letters.find(string, start_index) != -1:
        count += 1
        start_index = seq_letters.find(string, start_index) + str_len
    seq_numbers.append(count)

    # BRE
    count_all_bre = []
    bre_all = generate_bre()

    for c in bre_all:
        string = c  # modify the motif here
        str_len = len(string)  # length of the motif
        
        start_index = 0
        count = 0
        while seq_letters.find(string, start_index) != -1:
            count += 1
            start_index = seq_letters.find(string, start_index) + str_len
        seq_numbers.append(count)

    # DPE
    count_all_dpe = []
    dpe_all = generate_dpe()

    for c in dpe_all:
        string = c  # modify the motif here
        str_len = len(string)  # length of the motif
        
        start_index = 0
        count = 0
        while seq_letters.find(string, start_index) != -1:
            count += 1
            start_index = seq_letters.find(string, start_index) + str_len
        seq_numbers.append(count)

    # Inr
    count_all_inr = []
    inr_all = generate_inr()

    for c in inr_all:
        string = c  # modify the motif here
        str_len = len(string)  # length of the motif
        
        start_index = 0
        count = 0
        while seq_letters.find(string, start_index) != -1:
            count += 1
            start_index = seq_letters.find(string, start_index) + str_len
        seq_numbers.append(count)
    
    return seq_numbers


# testing for above
# foldername = 'Data_YuDengLab'
# datafile = 'Data_model_construction_YuDengLab'

# filename = ''+str(foldername)+'/'+str(datafile)+'.csv'

# with open(filename, 'r') as f:
#     file = f.readlines()
#     j = 1
#     for h in file:
#         line = h.strip().split(',')
#         test_seq = line[1]
#         count_motif(test_seq, [])
#         print(test_seq)
#         j += 1
#         if j > 10:
#             break