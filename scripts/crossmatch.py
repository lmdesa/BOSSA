import sys
sys.path.append('..')
from src.postprocessing import CompactBinaryPopulation

def main(canonical_distributions):
	sampler = CompactBinaryPopulation(canonical_distributions=canonical_distributions)
	sampler.crossmatch_sample()

if __name__ == '__main__':
	canonical_distributions = ''
	print('Was the ZAMS sample generated with uncorrelated parameter distributions? (Y/N)')
	while type(canonical_distributions) is not bool:
		canonical_distributions = str(input()).upper()
		if canonical_distributions == 'Y':
			canonical_distributions = True
		elif canonical_distributions == 'N':
			canonical_distributions = False
		else:
			print('Please reply with Y or N.')
			pass

	main(canonical_distributions)