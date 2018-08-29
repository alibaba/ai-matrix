import numpy as np
import pandas as pd
import os
import argparse

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def update( ):
    df = pd.DataFrame(columns = ['model','batch','fp32','fp32 std'])
    row = 0
    models = ['alexnet','googlenet','vgg16','resnet50','resnet152', 'densenet121', 'synNet']
    bs = ['16', '32', '64']
    
    for md in models:
        for b in bs:
            speed_arr = []
            fname = "./results/result_" + md + '_' + b + '.txt'
                #print('Open file: ', fname)
            with open(fname, "r") as ins:
                    arr = np.array([])
                    for line in ins:
                        if ( line.split(' ')[0]== '' and  is_number(line.split(' ')[-1] )):
                            val = line.split(' ')[-1]
                            val = float(val)
                            arr = np.append(arr, val)
                    arr = arr[1:] #remove the 1st perf number which is still in warm up period
                    speed_arr.append(np.mean(arr))               
                    speed_arr.append(np.std(arr))
                    df.loc[row] = [md, b] + speed_arr
               
            row += 1
    return df

                   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", help="compare card type" )

    args = parser.parse_args()
    df = update()
    df.to_csv('results.csv', encoding ='utf-8')
    if args.compare is not None:
        if args.compare == 'P4':
            df_card = pd.read_csv('./results_p4/results.csv')
            df['fp32 speedup'] = df['fp32'] / df_card['fp32'] 
        if args.compare == 'V100':
            df_card = pd.read_csv('./results_v100/results.csv')
            df['fp32 speedup'] = df['fp32'] / df_card['fp32'] 
    
    print(df)
    if args.compare is not None:
        print('Total speedup on all the models of fp32: ', df['fp32 speedup'].mean())

if __name__ == '__main__':
    main()
