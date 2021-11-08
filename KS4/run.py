import sys
import subprocess

# goal: at given h11, do everything between startand end with num_ex examples
# so the number of runs is (start-end)/num_ex
start, end, num_ex = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

# pass to generate.sage is h11 starter numex
for h11 in [10, 20, 30, 40, 50]:
	for curstart in range(start,end,num_ex):
		bashCommand = "sage generate.sage " + str(h11) + " " + str(curstart) + " " + str(num_ex)
		print bashCommand

		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		output, error = process.communicate()