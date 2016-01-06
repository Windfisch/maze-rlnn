set xrange [0: 1000]

plot 'logs/old.log' with lines,\
     'logs/new.log' with lines
pause -1

