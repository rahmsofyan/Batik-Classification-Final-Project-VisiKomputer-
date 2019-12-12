import os 

dir_test = './testing/'
col_batik = [i for i in os.listdir(dir_test)]
for i in col_batik:
	print(i)