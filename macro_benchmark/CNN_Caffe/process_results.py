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
    table0.append(['System performance, unit is images/second'])
    table0.append(['System performance measures the performance of the entire system, including both CPU work and GPU work. ' +
                   'The benchmark is timed right before and after the script is ran.'])
    table0.append(['model','batch 16','batch 32','batch 64'])

    table1 = []
    table1.append(['Approximate accelerator performance, unit is images/second'])
    table1.append(['Approximate accelerator performance aims at measuring only the performance of the accelerator. ' +
                   'The benchmark does its best to exclude CPU work from performance measurement and approximates the accelerator performance.'])
    table1.append(['model','batch 16','batch 32','batch 64'])

    models = ['googlenet_bvlc','resnet50','resnet152', 'densenet121','squeezenetv1.1']
    bs = ['16', '32', '64']
    #pc = ['fp32', 'fp16']
   
    folder_path = "./results_infer_trt_" + pc +"/" 
    for md in models:
        row_table0 = [md]
        row_table1 = [md]
        for b in bs:
            fname = folder_path + md + '_' + b + '_' + pc + '.txt'
            with open(fname, "r") as ins:
                for line in ins:
                    if not line.strip():
                        continue             
                    if (line.split(' ')[1]== 'Average'):
                        if (is_number(line.split()[6])):
                            val = line.split(' ')[6]
                            val = float(val)
                            val = "{0:.2f}".format(1/val*1000*float(b))
                        else:
                            print("Missing result, check if test is finished!")
                row_table1.append(val)
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
    #TODO
    #parser.add_argument("--compare", help="compare card type" )
    #parser.add_argument("--train", action='store_true', help="process train data" )
    #parser.add_argument("--infer", action='store_true', help="process infer data" )
    parser.add_argument("--infer_trt", action='store_true', help="process infer trt data" )
    parser.add_argument("--infer_trt_precision", type=str, default='fp32', help="process infer trt data" )

    args = parser.parse_args()
	    
    process_infer_trt( args.infer_trt_precision)

if __name__ == '__main__':
    main()
