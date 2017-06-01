import argparse


def a(list):
	for x in list:
		if(x == 1.0):
			print "aaaaaaa"

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument("trainning_file", help="arquivo de treinamento")
	arg_parser.add_argument("test_file", help="arquivo de teste")
	arg_parser.add_argument('--sections', nargs='+', type=float, default=[], help='List of sections to use.')
	arg_parser = arg_parser.parse_args()
	

	a(arg_parser.sections)