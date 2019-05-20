## Format of the Results Folder
The folder name should follow the following format:

{VENDOR_NAME}\_{ACCELERATOR_NAME}\_{NUMBER_OF_ACCELERATORS}\_{TEST_DATE}

{NUMBER_OF_ACCELERATORS} is optional for single-card trainging.

{TEST_DATE} is in the format of MonthDateYear.

An example is as follows:

Nvidia_V100_122018, Nvidia V100 single card tested in December, 2018


## System Documentation
A system information file is required in each test results folder to document the testing system. The file should contain the following information:

1. Accelerator<br />

2. CPU<br />

3. Operating system<br />

4. AI Frameworks (Such as Tensorflow and Caffe)<br />

5. Other information
