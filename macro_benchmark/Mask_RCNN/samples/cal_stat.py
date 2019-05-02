
def main():  
    bs = ['4', '8', '16']
    ptime = []
    for batch in bs:
        fname = "./results/result_" + batch +'.txt'
        with open(fname, "r") as ins:
            for line in ins:
                pass
            last = line
            print (last)
            if(last.split(' ')[0] != 'Total'):
                ctime = 100000000000
            else:
                ctime = float(last.split(' ')[4])
            ptime.append(ctime)
            print(ptime)
    P4_time = [63.7, 119.6, 100000000000] # P4 is OOM on batch=16
    V100_time = [40.2, 73.6, 136.7]
 
    print ('Speedup vs P4,   batch size    ')
    print ('%0.2f,           %0.0f' % (P4_time[0]/ptime[0], 4))
    print ('%0.2f,           %0.0f' % (P4_time[1]/ptime[1], 8))
    print ('%0.2f,           %0.0f' % (P4_time[2]/ptime[2], 16))
    
    print ('Speedup vs V100, batch size    ')
    print ('%0.2f,           %0.0f' % (V100_time[0]/ptime[0], 4))
    print ('%0.2f,           %0.0f' % (V100_time[1]/ptime[1], 8))
    print ('%0.2f,           %0.0f' % (V100_time[2]/ptime[2], 16))

if __name__ == '__main__':
    main()
