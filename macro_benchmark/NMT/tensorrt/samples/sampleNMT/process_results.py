import numpy as np
import os
import argparse

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def process_infer_trt(pc):
    table0 = []
    table0.append(['System performance, unit is sentences/second'])
    table0.append(['System performance measures the performance of the entire system, including both CPU work and GPU work. ' +
    	   'The benchmark is timed right before and after the script is ran.'])
    table0.append(['model','batch 64','batch 128','batch 256'])

    table1 = []
    table1.append(['Approximate accelerator performance, unit is sentences/second'])
    table1.append(['Approximate accelerator performance aims at measuring only the performance of the accelerator. ' +
    	   'The benchmark does its best to exclude CPU work from performance measurement and approximates the accelerator performance.'])
    table1.append(['model','batch 64','batch 128','batch 256'])

    models = ['nmt']
    bs = ['64', '128', '256']
    #pc = ['fp32', 'fp16']

    folder_path = "./results_infer_" + pc + "/"
    for md in models:
        row_table0 = [md]
        row_table1 = [md]
        for b in bs:
            fname = folder_path + "result_" + md + '_' + b + '_' + pc + '.txt'
            with open(fname, "r") as ins:
                for line in ins:
                    if not line.strip():
                        continue
                    if (line.split(' ')[-1]=='samples/sec\n' and  is_number(line.split(' ')[-2])):
                        #print('=======', line)
                        val = line.split()[-2]
                        val = float(val)
                        row_table1.append(val)
                        flag1 = True
                    if ( line.split()[0]== 'System' and line.split()[1]== 'performance' and is_number(line.split()[-1])):
                        val = line.split()[-1]
                        val = float(val)
                        row_table0.append(val)
                        flag0 = True
            if flag0 == False:
                row_table0.append(0.0000001)
            if flag1 == False:
                row_table1.append(0.0000001)
        table0.append(row_table0)
        table1.append(row_table1)

    # write results to file
    fname = folder_path + 'results_' + pc + '.csv'
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
    parser.add_argument("--infer_trt_precision", type=str, default='fp32', help="process infer trt data" )

    args = parser.parse_args()
    process_infer_trt( args.infer_trt_precision)


if __name__ == '__main__':
    main()
