import csv

txt_file = r"/home/vietduong/Desktop/metoo_project/metoo_tweets.txt"
csv_file = r"/home/vietduong/Desktop/metoo_project/metoo_tweets.csv"

in_txt = csv.reader(open(txt_file,"rb"),delimiter = "\t")
out_csv = csv.writer(open(csv_file,"wb"))
out_csv.writerows(in_txt)
