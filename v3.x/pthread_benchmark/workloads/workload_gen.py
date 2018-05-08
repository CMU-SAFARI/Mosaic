import random

bench_list = ["SRAD", "CONS", "BLK", "LIB", "SCAN", "HS", "BFS2", "MUM", "LUD", "LUH", "RED", "FWT", "NN", "SAD", "QTC", "SCP", "NW", "3DS", "LPS", "TRD", "JPEG", "HISTO", "GUPS", "MM", "CFD", "FFT", "SPMV", "SC", "BP", "RAY"]

workload_counts = 60
num_app = 7

workload_name = "sevenapp_random.wkld"

f = open(workload_name,"w")

for i in range(workload_counts):
    count = 0
    for i in range(num_app):
        f.write(bench_list[random.randint(0,len(bench_list)-1)])
        count = count+1
        if(count != num_app):
            f.write("-")
        else:
            f.write("\n")

f.close()

