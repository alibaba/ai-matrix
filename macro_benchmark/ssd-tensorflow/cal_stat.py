
def main():  
    fname = './results/result.txt'
    with open(fname, "r") as ins:
        for line in ins:
           pass
        last = line
        print(last.split(' '))
        ctime = float(last.split(' ')[6])
    # check github results folder for timing below 
    P4_time = 7.41
    V100_time = 5.83
 
    print ('Speedup vs P4 ')
    print ('%0.2f' % (P4_time/ctime))
    
    print ('Speedup vs V100')
    print ('%0.2f' % (V100_time/ctime))

if __name__ == '__main__':
    main()
