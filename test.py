from dga_classifier import lstm

with open('malware_dns_53_uniq') as f:
	X = [line.strip() for line in f.readlines()]
	y = [1] * len(X)


print lstm.test(X,y)['confusion_matrix']


with open('1m_domains.txt') as f:
	X = [line.strip() for line in f.readlines()]
	y = [0] * len(X)


print lstm.test(X,y)['confusion_matrix']

