import numpy as np
import os
import argparse

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def process_infer():
    table0 = []
    table0.append(['System performance, unit is images/second'])
    table0.append(['System performance measures the performance of the entire system, including both CPU work and GPU work. ' +
                   'The benchmark is timed right before and after the script is ran.'])
    table0.append(['batch 1'])

    table1 = []
    table1.append(['Approximate accelerator performance, unit is images/second'])
    table1.append(['Approximate accelerator performance aims at measuring only the performance of the accelerator. ' +
                   'The benchmark does its best to exclude CPU work from performance measurement and approximates the accelerator performance.'])
    table1.append(['batch 1'])

    row_table0 = []
    row_table1 = []
    fname = "./results/result_infer.txt"
    with open(fname, "r") as ins:
        for line in ins:
            if (len(line.split()) > 2):
                if ( line.split()[0]== 'Approximate' and line.split()[1]== 'accelerator' and line.split()[2]== 'performance' and is_number(line.split()[-1])):
                    val = line.split()[-1]
                    val = float(val)
                    row_table1.append(val)
                if ( line.split()[0]== 'System' and line.split()[1]== 'performance' and is_number(line.split()[-1])):
                    val = line.split()[-1]
                    val = float(val)
                    row_table0.append(val)
    table0.append(row_table0)
    table1.append(row_table1)

    # write results to file
    fname = './results/results_infer.csv'
    with open(fname, "w") as outfile:
        for line in table0:
            for entry in line:
                outfile.write(str(entry)+",")
            outfile.write("\n")

        outfile.write("\n\n\n")

        for line in table1:
            for entry in line:
                outfile.write(str(entry)+",")
            outfile.write("\n")

def process_train():
    table0 = []
    table0.append(['System performance, unit is images/second'])
    table0.append(['System performance measures the performance of the entire system, including both CPU work and GPU work. ' +
                   'The benchmark is timed right before and after the script is ran.'])
    table0.append(['batch 2'])

    table1 = []
    table1.append(['Approximate accelerator performance, unit is images/second'])
    table1.append(['Approximate accelerator performance aims at measuring only the performance of the accelerator. ' +
                   'The benchmark does its best to exclude CPU work from performance measurement and approximates the accelerator performance.'])
    table1.append(['batch 2'])

    row_table0 = []
    row_table1 = []
    fname = "./results/result_train.txt"
    with open(fname, "r") as ins:
        for line in ins:
            if (len(line.split()) > 2):
                if ( line.split()[0]== 'Approximate' and line.split()[1]== 'accelerator' and line.split()[2]== 'performance' and is_number(line.split()[-1])):
                    val = line.split()[-1]
                    val = float(val)
                    row_table1.append(val)
                if ( line.split()[0]== 'System' and line.split()[1]== 'performance' and is_number(line.split()[-1])):
                    val = line.split()[-1]
                    val = float(val)
                    row_table0.append(val)
    table0.append(row_table0)
    table1.append(row_table1)
    
    # write results to file
    fname = './results/results_train.csv'
    with open(fname, "w") as outfile:
        for line in table0:
            for entry in line:
                outfile.write(str(entry)+",")
            outfile.write("\n")

        outfile.write("\n\n\n")

        for line in table1:
            for entry in line:
                outfile.write(str(entry)+",")
            outfile.write("\n")
                   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="process train data" )
    parser.add_argument("--infer", action='store_true', help="process infer data" )

    args = parser.parse_args()
    if args.train is True:
        process_train()
    elif args.infer is True:
        process_infer()
    else:
        print("Must set train or infer when processing data!")
        exit()

if __name__ == '__main__':
    main()
