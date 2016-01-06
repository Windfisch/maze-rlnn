from sys import argv

#usage: avg.py infile shift > outfile

infile = open(argv[1],"r")
shift = int(argv[2])

log = [0]*shift

for line in infile:
    a,b = line.split()
    a=int(float(a))
    b=float(b)
    log += [b]
    print("%i\t%e\t%e\t%e" % (a,(b-log[-shift])/shift,b,log[-shift]))

infile.close()
