from sys import argv

#usage: avg.py outfile infile1 infile2 ...

outfile, infiles = open(argv[1],"w"), [open(arg,"r") for arg in argv[2:]]

# hell yea! i f***ing love python :'D
for lines in zip(*infiles):
    for columns in zip(*[l.split() for l in lines]):
        print("%e" % ( sum([float(c) for c in columns])/len(columns) ) ,file=outfile, end="\t")
    print("", file=outfile)

for f in [outfile]+infiles:
    f.close()
