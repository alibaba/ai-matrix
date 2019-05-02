#### 1. To test peak performance, please run with command below. card_type = V100 or P100 or T4. The number of repeated runs = 10 in this case.  
./test_allgemm.sh *card_type*
  
#### 2. To run pressure test, please run with command below. card_type = V100 or P100 or T4. The number of repeated runs = 2000 in this case.  
./test_allgemm_pressure.sh *card_type* 
