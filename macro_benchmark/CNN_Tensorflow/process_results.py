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
    table0.append(['model','batch 16','batch 32','batch 64'])

    table1 = []
    table1.append(['Approximate accelerator performance, unit is images/second'])
    table1.append(['Approximate accelerator performance aims at measuring only the performance of the accelerator. ' +
                   'The benchmark does its best to exclude CPU work from performance measurement and approximates the accelerator performance.'])
    table1.append(['model','batch 16','batch 32','batch 64'])

    models = ['googlenet','resnet50','resnet152', 'densenet121', 'synNet']
    bs = ['16', '32', '64']
    
    for md in models:
        row_table0 = [md]
        row_table1 = [md]
        for b in bs:
            fname = "./results_infer/result_" + md + '_' + b + '.txt'
            with open(fname, "r") as ins:
                arr = np.array([])
                flag = False
                for line in ins:
                    if not line.strip():
                        continue             
                    if (line.split(' ')[0]== '' and len(line.split())==4 and is_number(line.split(' ')[-1])):
                        val = line.split()[-1]
                        val = float(val)
                        arr = np.append(arr, val)
                        flag = True
                if flag == False:
                    arr = np.append(arr, 0.0000001)
                arr = arr[1:] #remove the 1st perf number which is still in warm up period
                row_table1.append(np.mean(arr))
                if (is_number(line.split()[-1])):
                    row_table0.append(line.split()[-1])
                else:
                    print("Missing result, check if test is finished!")
        table0.append(row_table0)
        table1.append(row_table1)

    # write results to file
    fname = './results_infer/results.csv'
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
    table0.append(['model','batch 16','batch 32','batch 64'])

    table1 = []
    table1.append(['Approximate accelerator performance, unit is images/second'])
    table1.append(['Approximate accelerator performance aims at measuring only the performance of the accelerator. ' +
                   'The benchmark does its best to exclude CPU work from performance measurement and approximates the accelerator performance.'])
    table1.append(['model','batch 16','batch 32','batch 64'])

    models = ['googlenet','resnet50','resnet152', 'densenet121', 'synNet']
    bs = ['16', '32', '64']
    
    for md in models:
        row_table0 = [md]
        row_table1 = [md]
        for b in bs:
            fname = "./results_train/result_" + md + '_' + b + '.txt'
            with open(fname, "r") as ins:
                flag = False
                for line in ins:
                    if not line.strip():
                        continue             
                    if ( line.split()[0]== 'Images/sec:' and  is_number(line.split()[1] )):
                        val = line.split()[1]
                        val = float(val)
                        flag = True
                if flag == False:
                    val = 0.0000001
                row_table1.append(val)
                if (is_number(line.split()[-1])):
                    row_table0.append(line.split()[-1])
                else:
                    print("Missing result, check if test is finished!")
        table0.append(row_table0)
        table1.append(row_table1)
    
    # write results to file
    fname = './results_train/results.csv'
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
    parser.add_argument("--compare", help="compare card type" )
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

    if args.compare is not None:
        if args.compare == 'P4':
            df_card = pd.read_csv('./results_p4/results.csv')
            df['fp32 speedup'] = df['fp32'] / df_card['fp32'] 
        if args.compare == 'V100':
            df_card = pd.read_csv('./results_v100/results.csv')
            df['fp32 speedup'] = df['fp32'] / df_card['fp32'] 
    
    #print(df)
    if args.compare is not None:
        print('Total speedup on all the models of fp32: ', df['fp32 speedup'].mean())

if __name__ == '__main__':
    main()
