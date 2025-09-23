pairs = []
total = len(pairs)

with open("bilingual_pairs\\fra_copy.txt", 'r') as f:
    pairs = f.readlines()

for pair_ind in range(total):
    pairs[pair_ind] = pairs[pair_ind].strip().split('\t')
    pairs[pair_ind] = pairs[pair_ind][:-1]
    print(f'{pair_ind} / {total} \t | \t {pair_ind/total * 100:.2f}%')

with open("bilingual_pairs/fra edit.txt", 'w') as f:
    for pair in pairs:
        temp = (pair.split('\t'))[:-1]
        f.write(f'{temp[0]}\t{temp[len(temp)-1]}\n')