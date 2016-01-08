set xrange [0: 1000]

plot 'logs/old.log' with lines,\
     'logs/new.log' with lines,\
     'logs/nn.log' using ($1,$2*20) with lines,\
     'logs/nn2.log' using ($1,$2*20) with lines
pause -1

