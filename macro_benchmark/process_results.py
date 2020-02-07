import sys
import csv
import xlsxwriter
import argparse
def createMd(args):
    results_dir = args.results_dir
    if args.trt_fp32:
        pc='fp32'
    elif args.trt_fp16:
        pc='fp16'
    elif args.trt_int8:
        pc='int8'
    else:
        print("wrong precision type")
        exit()


    models = ["cnn_tf", "cnn_caffe", "ssd_caffe"]
    dc = {}
    # add table for cnn tensorflow
    with open(results_dir + "/results_cnn_tf/results_"+pc+".csv") as csvfile:
        dc["googlenet"]=[]
        dc["resnet50"]=[]
        dc["resnet152"]=[]
        dc["densenet121"]=[]
        table = csv.reader(csvfile)
        flag = False
        for row in table:
            if(not row):
                 continue
            if row[0] == "Approximate accelerator performance": 
               flag = True
               continue
            if row[0] == "googlenet" and flag:
                dc["googlenet"].append(row[1])
                dc["googlenet"].append(row[2])
                dc["googlenet"].append(row[3])
            if row[0] == "resnet50" and flag:
                dc["resnet50"].append(row[1])
                dc["resnet50"].append(row[2])
                dc["resnet50"].append(row[3])
            if row[0] == "resnet152" and flag:
                dc["resnet152"].append(row[1])
                dc["resnet152"].append(row[2])
                dc["resnet152"].append(row[3])
            if row[0] == "densenet121" and flag:
                dc["densenet121"].append(row[1])
                dc["densenet121"].append(row[2])
                dc["densenet121"].append(row[3])
    with open(results_dir + "/results_"+pc+".md", "w") as md:
        txt = "#CNN-Tensorflow  \nThe test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  \n  \n" 
        txt=txt+"| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  \n"
        txt=txt+"|-----------|-----------|---------|---------|---------|  \n"
        txt=txt+"| Googlenet | "+pc+     "|"+dc["googlenet"][0]+"|"+dc["googlenet"][1]+"|"+dc["googlenet"][2]+"|  \n"
        txt=txt+"| Resnet50 | "+pc+      "|"+dc["resnet50"][0]+"|"+dc["resnet50"][1]+"|"+dc["resnet50"][2]+"|  \n"
        txt=txt+"| Resnet152 | "+pc+     "|"+dc["resnet152"][0]+"|"+dc["resnet152"][1]+"|"+dc["resnet152"][2]+"|  \n"
        txt=txt+"| Densenet121 | "+pc+   "|"+dc["densenet121"][0]+"|"+dc["densenet121"][1]+"|"+dc["densenet121"][2]+"|  \n  \n"
        txt=txt+"*BS: Batch Size*  \n"
        txt=txt+"*Unit: Img/sec*  \n  \n"
        md.write(txt)
        md.close() 

    # add table for cnn caffe
    with open(results_dir + "/results_cnn_caffe/results_"+pc+".csv") as csvfile:
        dc["googlenet"]=[]
        dc["resnet50"]=[]
        dc["resnet152"]=[]
        dc["densenet121"]=[]
        dc["squeezenetv1.1"]=[]
        table = csv.reader(csvfile)
        flag = False
        for row in table:
            if(not row):
                 continue
            if row[0] == "Approximate accelerator performance": 
               flag = True
               continue
            if row[0] == "googlenet_bvlc" and flag:
                dc["googlenet"].append(row[1])
                dc["googlenet"].append(row[2])
                dc["googlenet"].append(row[3])
            if row[0] == "resnet50" and flag:
                dc["resnet50"].append(row[1])
                dc["resnet50"].append(row[2])
                dc["resnet50"].append(row[3])
            if row[0] == "resnet152" and flag:
                dc["resnet152"].append(row[1])
                dc["resnet152"].append(row[2])
                dc["resnet152"].append(row[3])
            if row[0] == "densenet121" and flag:
                dc["densenet121"].append(row[1])
                dc["densenet121"].append(row[2])
                dc["densenet121"].append(row[3])
    
            if row[0] == "squeezenetv1.1" and flag:
                dc["squeezenetv1.1"].append(row[1])
                dc["squeezenetv1.1"].append(row[2])
                dc["squeezenetv1.1"].append(row[3])
    with open(results_dir + "/results_"+pc+".md", "a") as md:
        txt="#CNN-Caffe  \nThe test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  \n  \n"
        txt=txt+"| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  \n"
        txt=txt+"|-----------|-----------|---------|---------|---------|  \n"
        txt=txt+"| Googlenet | "+pc+    "|"+dc["googlenet"][0]+"|"+dc["googlenet"][1]+"|"+dc["googlenet"][2]+"|  \n"
        txt=txt+"| Resnet50 | "+pc+     "|"+dc["resnet50"][0]+"|"+dc["resnet50"][1]+"|"+dc["resnet50"][2]+"|  \n"
        txt=txt+"| Resnet152 | "+pc+    "|"+dc["resnet152"][0]+"|"+dc["resnet152"][1]+"|"+dc["resnet152"][2]+"|  \n"
        txt=txt+"| Densenet121 | "+pc+  "|"+dc["densenet121"][0]+"|"+dc["densenet121"][1]+"|"+dc["densenet121"][2]+"|  \n"
        txt=txt+"| Squeezenetv1.1 | "+pc+"|"+dc["squeezenetv1.1"][0]+"|"+dc["squeezenetv1.1"][1]+"|"+dc["squeezenetv1.1"][2]+"|  \n  \n"
        txt=txt+"*BS: Batch Size*  \n"
        txt=txt+"*Unit: Img/sec*  \n  \n"
        md.write(txt)         
        md.close()
    
    # add table for ssd caffe
    with open(results_dir + "/results_ssd_caffe/results_"+pc+".csv") as csvfile:
        dc["ssd-vgg16"]=[]
        table = csv.reader(csvfile)
        flag = False
        for row in table:
            if(not row):
                 continue
            if row[0] == "Approximate accelerator performance": 
               flag = True
               continue
            if row[0] == "ssd-vgg16" and flag:
                dc["ssd-vgg16"].append(row[1])
                dc["ssd-vgg16"].append(row[2])
                dc["ssd-vgg16"].append(row[3])
    with open(results_dir + "/results_"+pc+".md", "a") as md:
        txt="#SSD-Caffe  \nThe test below is run on Nvidia Tensor RT with trained weights. The results is for throughput with synthetic inputs.  \n  \n"    
        txt=txt+"| Models    | Precision | BS = 16 | BS = 32 | BS = 64 |  \n"
        txt=txt+"|-----------|-----------|---------|---------|---------|  \n"
        txt=txt+"| SSD-VGG16 | "+pc+    "|"+dc["ssd-vgg16"][0]+"|"+dc["ssd-vgg16"][1]+"|"+dc["ssd-vgg16"][2]+"|  \n  \n"
        txt=txt+"*BS: Batch Size*  \n"
        txt=txt+"*Unit: Img/sec*  \n  \n"
        md.write(txt)         
        md.close()


def create_excel(args):
    results_dir = args.results_dir
    if args.trt_fp32:
        models = ["cnn_tf", "cnn_caffe", "ssd_caffe"]
        wb = xlsxwriter.Workbook(results_dir + "/results_infer.xlsx")
        for model in models:
            ws = wb.add_worksheet(model)
            with open(results_dir + "/results_" + model + "/results_fp32.csv") as csvfile:
                table = csv.reader(csvfile)
                i = 0
                for row in table:
                    ws.write_row(i, 0, row)
                    i += 1
        wb.close()
	
    else:
        models = ["cnn", "deepinterest", "maskrcnn", "nmt", "ssd_resnet34", "ssd_resnet18", "dssd", "ncf", "dien", "bert", "faster_rcnn", "wide_deep", "cpn", "seglink", "crnn"]
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
            if model == "ssd_resnet34":
                continue
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
    parser.add_argument("--trt_fp32", action="store_true", help="process trt results" )
    parser.add_argument("--trt_fp16", action="store_true", help="process trt results" )
    parser.add_argument("--trt_int8", action="store_true", help="process trt results" )
    args = parser.parse_args()
    if args.trt_fp32 or args.trt_fp16 or args.trt_int8:
#        create_excel(args)
        createMd(args)
    else:
        create_excel(args)
