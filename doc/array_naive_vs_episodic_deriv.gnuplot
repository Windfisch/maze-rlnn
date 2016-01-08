set xrange [0: 1000]

plot 'derivs/old.log' with lines,\
     'derivs/new.log' with lines,\
     'derivs/nn.log' using ($1, $2*20) with lines,\
     'derivs/nn2.log' using ($1, $2*20) with lines
pause -1
