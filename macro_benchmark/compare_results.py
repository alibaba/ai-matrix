#!/usr/bin/python

import numpy as np
import os
import argparse
import shutil
import process_results

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def compare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", help="location of target results, target is normalized to ref" )
    parser.add_argument("--ref_dir", help="location of reference results" )
    args = parser.parse_args()

    out_dir = args.target_dir + "_vs_" + args.ref_dir
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    ops = ["train", "infer"]
    models = ["cnn", "deepinterest", "dien", "dssd", "maskrcnn", "ncf", "nmt", "ssd_resnet34", "ssd_resnet18"]
    for op in ops:
        for model in models:
            fname_target = args.target_dir + "/results_" + model + "_" + op + ".csv"
            fname_ref = args.ref_dir + "/results_" + model + "_" + op + ".csv"
            fname_out = out_dir + "/results_" + model + "_" + op + ".csv"
            table_target = []
            table_ref = []

            with open(fname_target, "r") as f_target:
                for line in f_target:
                    table_target.append(line)

            with open(fname_ref, "r") as f_ref:
                for line in f_ref:
                    table_ref.append(line)

            if len(table_target) != len(table_ref):
                print("Target results has different length than reference results!")
                print("Target file: ", fname_target)
                print("Reference file: ", fname_ref)
                exit()

            with open(fname_out, "w") as f_out:
                for line_target, line_ref in zip(table_target, table_ref):
                    items_target = line_target.split(",")
                    if len(items_target) < 2:
                        f_out.write(items_target[0])
                        continue

                    if is_number(items_target[-2]):
                        items_ref = line_ref.split(",")
                        for item_target, item_ref in zip(items_target, items_ref):
                            if is_number(item_target):
                                f_out.write(str(float(item_target)/float(item_ref)))
                                f_out.write(",")
                            else:
                                f_out.write(item_target)
                                if item_target != "\n":
                                    f_out.write(",")
                    else:
                        f_out.write(line_target)

    process_results.create_excel(out_dir)

if __name__ == '__main__':
    compare()
