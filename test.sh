#!/bin/bash

name=$1
num=$2
shift 2

mkdir -p derivs
mkdir -p logs
mkdir -p logtmp

for ((i=0;i<$num;i++)); do
	if [ ! -e logtmp/$name.$i.log ]; then
		python sol.py $@ --file logtmp/$name.$i.log
	fi
done

python avg.py logs/$name.log logtmp/$name.*.log
python shift.py logs/$name.log 50 > derivs/$name.log

echo "plot 'logs/$name.log' with lines" > $name.gnuplot
echo "pause -1" >> $name.gnuplot
echo "plot 'derivs/$name.log' with lines" > $name.d.gnuplot
echo "pause -1" >> $name.d.gnuplot
