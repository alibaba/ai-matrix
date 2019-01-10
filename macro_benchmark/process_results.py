import sys
import csv
import xlsxwriter
import argparse

def create_excel(results_dir):
	models = ["cnn", "deepinterest", "maskrcnn", "nmt", "ssd", "dssd", "ncf", "dien"]

	wb = xlsxwriter.Workbook(results_dir + "/results_train.xlsx")
	for model in models:
		ws = wb.add_worksheet(model)
		with open(results_dir + "/results_" + model + "_train.csv") as csvfile:
			table = csv.reader(csvfile)
			i = 0
			for row in table:
				ws.write_row(i, 0, row)
				i += 1
	wb.close()

	wb = xlsxwriter.Workbook(results_dir + "/results_infer.xlsx")
	for model in models:
		ws = wb.add_worksheet(model)
		with open(results_dir + "/results_" + model + "_infer.csv") as csvfile:
			table = csv.reader(csvfile)
			i = 0
			for row in table:
				ws.write_row(i, 0, row)
				i += 1
	wb.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--results_dir", default="results", help="location of results" )
	args = parser.parse_args()
	create_excel(args.results_dir)
