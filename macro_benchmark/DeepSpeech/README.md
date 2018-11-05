# DeepSpeech

Here we use a pre-trained English model as the inference model.  
Data: The model can only take audio file .wav format, with 16k freq and 16bit compression.  
Run the command for inference. It will translate two audio clips.
The DeepSpeech code is already downloaded(from https://github.com/mozilla/DeepSpeech)
Step 1:
```
pip3 install deepspeech-gpu
```
Step 2:
```
cd macro_benchmark/DeepSpeech
./0_download.sh
./1_run.sh
```
CUDA 8 needed. CUDA 9 is not supported yet.  
