import os

directory='/home/vietduong/Desktop/metoo_project/'

data = open(directory+'test_sents.txt','r').readlines()
data = [sent.strip() for sent in data]

os.chdir(directory+'results')

count = 0

for sent in data:
    if count < 100:
        output_file = ('0'*(4-len(str(count))))+str(count)
        print(output_file)
        print('trips-web "{}" > {}.json'.format(sent,output_file))
        os.system('trips-web "{}" > {}.json'.format(sent,output_file))
        count = count+1
