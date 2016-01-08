set xrange [0: 1000]

plot 'logs/old.log' with lines title 'QArray, old textbook strategy',\
     'logs/new.log' with lines title 'QArray, new reversed strategy',\
     'logs/nn2.log' using ($1, $2*20) with lines title 'QNN',\
     'logs/nn.log' using ($1, $2*20) with lines title 'QNN, friendlyness=1'
pause -1

