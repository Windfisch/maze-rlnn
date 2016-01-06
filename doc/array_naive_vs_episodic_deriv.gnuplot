set xrange [0: 1000]

plot 'derivs/old.log' with lines,\
     'derivs/new.log' with lines
pause -1
