import numpy as np
import pandas as pd
import os
import argparse

def update( num_dt ):
    df = pd.DataFrame(columns = ['model','batch','fp32','fp32 std','fp16','fp16 std','int8','int8 std'])
    row = 0
    models = ['alexnet','googlenet','vgg16','resnet50','resnet152', 'densenet121']
    bs = ['16', '32', '64']
    if num_dt == 2:
        dt = ['fp32','fp16']
    else:
        dt = ['fp32','fp16','int8']
    
    for md in models:
        for i in range(len(bs)):
            speed_arr = []
            for j in range(len(dt)):
                fname = "./results/" + md + '_' + bs[i] + '_' + dt[j] + '.txt'
                #print('Open file: ', fname)
                with open(fname, "r") as ins:
                    arr = np.array([])
                    for line in ins:
                        if (line.split(' ')[0] == 'Average'):
                            val = line.split(' ')[-2]
                            val = float(val)
                            arr = np.append(arr, val)
                            #print(arr)
                    speed = float(bs[i]) * 1000 / arr
                    speed_arr.append(np.mean(speed))               
                    speed_arr.append(np.std(speed))               
               
            if num_dt == 2:
                df.loc[row] =  [md, bs[i]] + speed_arr + [0,0]
            else:
                df.loc[row] =  [md, bs[i]] + speed_arr

            row += 1
    
    return df

                    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_dt", help="number of data types", required=True, type=int)
    parser.add_argument("--compare", help="compare card type" )

    args = parser.parse_args()
    df = update(args.num_dt)
    df.to_csv('results.csv', encoding ='utf-8')
    if args.compare is not None:
        if args.compare == 'P4':
            df_p4 = pd.read_csv('./results_p4/results.csv')
            df['fp32 speedup'] = df['fp32'] / df_p4['fp32'] 
            df['fp16 speedup'] = df['fp16'] / df_p4['fp16'] 
            df['int8 speedup'] = df['int8'] / df_p4['int8']
        if args.compare == 'V100':
            df_p4 = pd.read_csv('./results_v100/results.csv')
            df['fp32 speedup'] = df['fp32'] / df_p4['fp32'] 
            df['fp16 speedup'] = df['fp16'] / df_p4['fp16'] 
    
    print(df)
    print('Total speedup on all the models of fp32 ', df['fp32 speedup'].mean())
    print('Total speedup on all the models of fp16 ', df['fp16 speedup'].mean())
    print('Total speedup on all the models of int8 ', df['int8 speedup'].mean())

if __name__ == '__main__':
    main()
