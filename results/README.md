# Results
The AI Matrix results are uploaded to this folder result_a_b_c where a_b_c means AIMatrix version a_b_c

## Format of the Results Folder
The folder name should follow the following format:

{VENDOR_NAME}\_{ACCELERATOR_NAME}\_{NUMBER_OF_ACCELERATORS}\_{TEST_DATE}

{NUMBER_OF_ACCELERATORS} is optional for single-card trainging.

{TEST_DATE} is in the format of MonthDateYear.

An example is as follows:

NVIDIA_V100_12072018, NVIDIA V100 single card tested on December 7, 2018

NVIDIA_V100_8_12072018, NVIDIA V100 8 card tested on December 7, 2018

## System Documentation
A README.md file is required in each test results folder to document the testing system. The README file should contain the following information:

1. CPU<br />
	a. Vendor<br />
	b. CPU type<br />
	c. Number of cores<br />
	d. Total number of threads<br />
	e. Frequency<br />
	f. Memory size<br />

2. Accelerator<br />
	a. Vendor<br />
	b. Accelerator type<br />
	c. Number of accelerators used for training<br />
	d. Frequency<br />
	e. Memory size<br />
	f. Other information<br />

3. Operating system<br />
	a. OS name<br />
	b. OS version<br />
	c. Library information (Such as python version)<br />
	d. Other information<br />

4. Accelerator specific libraries and driver (Such as cuda for NVIDIA GPUs)<br />
	a. Library name<br />
		i. Version<br />
	b. Driver version<br />
	c. Other information<br />

5. AI Frameworks (Such as Tensorflow and Caffe)<br />
	a. Framework name<br />
		i. Version<br />
	b. Other information<br />

6. Other information
