from dga_classifier import lstm

with open('cnn/mix0.data') as f:
	X = [line.strip() for line in f.readlines()]
with open('cnn/mix0.label') as f2:
	y = [int(line.strip()) for line in f2.readlines()]


print lstm.test(X,y)['confusion_matrix']

