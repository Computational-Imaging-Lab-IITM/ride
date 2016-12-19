# Compressive Image Recovery Using Recurrent Generative Model

Code corresponding to the paper : (https://arxiv.org/abs/1612.04229)  
Forked from the original code for RIDE, which can be found [here](https://github.com/lucastheis/ride/)

## Results

## Requirements

Same as original RIDE version (https://github.com/lucastheis/ride/)

## Usage

1. Missing Pixel Interpolation

  - Run `python experiments/map_interpolate_stack.py` for default parameters. Following options can be changed :
  ```
  -m/--model  <Path to the model. Trained 1 layer and 2 layer models available in models/>
  -d/--data   <Path to the test images in mat format. Images chosen from BSDS dataset in the paper available in data/>
  -h/--holes  <Fraction of pixels removed from the image at random. Default is 70%>
  -m/--momentum <Momentum set for gradient ascent in image reconstruction>
  -l/--lr     <Learning Rate set for gradient ascent in image reconstruction>
  -N/--niter  <Number of iterations for gradient ascent>
  -p/--path   <Path to save the resulting images>
  -q/--mode   <Mode to run Caffe in>
  -D/-device  <Device ID for GPU>
  -s/--size   <Size of test images>
  -f/--flip   <Flag to carry out direction flipping as mentioned in paper>
  -e/--ent_max <For thresholding posterior entropy as mentioned in paper>
  -r/-resume   <For resuming the gradient ascent from previous npy file at certain iteration>
  -I/--index   <To select which test image to work on from the mat file>
  ```

  - The test image will be divided into four parts and each gradient ascent will run on each part simultaneously. To stitch the four reconstructed parts use `python experiments/stitch_stack.py`. Specify index of the test image using `-I` and iteration to choose for the reconstructed npy file using `-i` option 

2. Single Pixel Camera Reconstruction

  - Create compressive sensing matrix using `python experiments/create_Phi.py`

  - Run `python experiments/map_single_pixel_stack.py` for default parameters. Following options can be changed :
  ```
  -m/--model  <Path to the model. Trained 1 layer and 2 layer models available in models/>
  -d/--data   <Path to the test images in mat format. Images chosen from BSDS dataset in the paper available in data/>
  -n/--noise_std <For adding noise to the sensed measurements. By default no noise is added>
  -d/--momentum <Momentum set for gradient ascent in image reconstruction>
  -l/--lr     <Learning Rate set for gradient ascent in image reconstruction>
  -N/--niter  <Number of iterations for gradient ascent>
  -p/--path   <Path to save the resulting images>
  -q/--mode   <Mode to run Caffe in>
  -D/-device  <Device ID for GPU>
  -s/--size   <Size of test images>
  -f/--flip   <Flag to carry out direction flipping as mentioned in paper>
  -e/--ent_max <For thresholding posterior entropy as mentioned in paper>
  -r/-resume   <For resuming the gradient ascent from previous npy file at certain iteration>
  -K/--image_num   <To select first K images from mat file to test >
  ``` 
