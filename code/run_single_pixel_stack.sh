# #Expt 1
# mkdir map_single_pixel/expt1/
# parallel -j 8 'mkdir map_single_pixel/{}' ::: {0..4}
# parallel -j 8 'mkdir map_single_pixel/cleaned_img/' ::: {0..4}
# python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -p '/home/cplab-ws1/ride/code/map_single_pixel/' \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 1 \
# -K 5 \

#40% Measurements

#RIDE
# 0 ssim 0.928892369573 psnr 35.5972187535
# 1 ssim 0.892590745267 psnr 32.0905221408
# 2 ssim 0.947521021256 psnr 35.4415161014
# 3 ssim 0.810051547995 psnr 28.2987823965
# 4 ssim 0.944755839272 psnr 36.8837614667

#TVAL3
# 0 ssim 0.8978 psnr 31.721
# 1 ssim 0.7969 psnr 28.0924
# 2 ssim 0.8420 psnr 28.4038
# 3 ssim 0.7776 psnr 28.0183
# 4 ssim 0.9016 psnr 32.5708

#30% Measurements

#RIDE
# 0 ssim 0.898337580958 psnr 33.8692097402
# 1 ssim 0.835253188772 psnr 30.0329660115
# 2 ssim 0.919853547058 psnr 33.0299790348
# 3 ssim 0.723675708993 psnr 26.2391711738
# 4 ssim 0.919323815247 psnr 34.7045616767

#TVAL3
# 0 ssim 0.8334 psnr 29.5418
# 1 ssim 0.7268 psnr 26.6480
# 2 ssim 0.7895 psnr 26.2003
# 3 ssim 0.7171 psnr 26.8033
# 4 ssim 0.8498 psnr 30.2081

#25% Measurements

#RIDE
# 0 ssim 0.871687819835 psnr 32.7199530778
# 1 ssim 0.791630365619 psnr 28.8520276363
# 2 ssim 0.897591816639 psnr 31.7091833807
# 3 ssim 0.701796557519 psnr 26.0556427832
# 4 ssim 0.900179204455 psnr 33.3790827733


#TVAL3
# 0 ssim 0.8021 psnr 28.4020
# 1 ssim 0.6810 psnr 25.8610
# 2 ssim 0.7452 psnr 25.0833
# 3 ssim 0.6751 psnr 26.0894
# 4 ssim 0.8202 psnr 29.1073

#10% Measurements

#RIDE
# 0 ssim 0.686104939507 psnr 25.7714407716
# 1 ssim 0.477140536869 psnr 23.6743085066
# 2 ssim 0.551151710253 psnr 20.6484562748
# 3 ssim 0.39062567391 psnr 20.4675738573
# 4 ssim 0.657434672911 psnr 25.8876130577

#TVAL3
# 0 ssim 0.6379 psnr 23.1190
# 1 ssim 0.4798 psnr 23.0110
# 2 ssim 0.5459 psnr 21.1264
# 3 ssim 0.5013 psnr 23.4392
# 4 ssim 0.6806 psnr 25.4636



#DAMP
# 0 psnr 33.986437 ssim 0.90800830 
# 0 psnr 31.586437 ssim 0.86980725 
# 0 psnr 29.693178 ssim 0.83581810 
# 0 psnr 21.892422 ssim 0.62456140 

# 1 psnr 27.333594 ssim 0.73989330 
# 1 psnr 26.083967 ssim 0.67352125 
# 1 psnr 25.436566 ssim 0.63298110 
# 1 psnr 22.322556 ssim 0.40255540 

# 2 psnr 36.039108 ssim 0.96133630 
# 2 psnr 31.665062 ssim 0.92296725 
# 2 psnr 29.089633 ssim 0.88330810 
# 2 psnr 21.138884 ssim 0.59540240 

# 3 psnr 26.898153 ssim 0.66092930 
# 3 psnr 25.837631 ssim 0.61449325 
# 3 psnr 25.265706 ssim 0.58395910 
# 3 psnr 22.524018 ssim 0.42487740 

# 4 psnr 38.451082 ssim 0.97040130 
# 4 psnr 30.174160 ssim 0.85827425 
# 4 psnr 28.063893 ssim 0.79575810 
# 4 psnr 24.410501 ssim 0.653400 

# #Expt 2
# mkdir map_single_pixel/expt1/
# parallel -j 8 'mkdir map_single_pixel/{}' ::: {0..4}
# parallel -j 8 'mkdir map_single_pixel/cleaned_img/' ::: {0..4}
# python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -p '/home/cplab-ws1/ride/code/map_single_pixel/' \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 1 \
# -K 5 \
# -y 0.3 | tee map_single_pixel/0p3_log.txt 


#Expt 3
# parallel -j 8 'mkdir map_single_pixel/{}' ::: {0..4}
# parallel -j 8 'mkdir map_single_pixel/cleaned_img/' ::: {0..4}
# python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -p '/home/cplab-ws1/ride/code/map_single_pixel/' \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 1 \
# -K 5 \
# -y 0.25 | tee map_single_pixel/0p25_log.txt 

# #Expt 4
# parallel -j 8 'mkdir map_single_pixel/{}' ::: {0..4}
# parallel -j 8 'mkdir map_single_pixel/cleaned_img/' ::: {0..4}
# python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -p '/home/cplab-ws1/ride/code/map_single_pixel/' \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 1 \
# -K 5 \
# -y 0.10 | tee map_single_pixel/0p10_log.txt 

# #Expt 5
# parallel -j 8 'mkdir map_single_pixel/{}' ::: {0..4}
# parallel -j 8 'mkdir map_single_pixel/cleaned_img/' ::: {0..4}
# python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -p '/home/cplab-ws1/ride/code/map_single_pixel/' \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 1 \
# -K 5 \
# -y 0.15 | tee map_single_pixel/0p15_log.txt 

 
#RIDE
 
# 0  & 35.59 & 0.928
# 1  & 32.09 & 0.892
# 2  & 35.44 & 0.947
# 3  & 28.29 & 0.810
# 4  & 36.88 & 0.944
 
# 0  & 33.86 & 0.898
# 1  & 30.03 & 0.835
# 2  & 33.02 & 0.919
# 3  & 26.23 & 0.723
# 4  & 34.70 & 0.919
 
# 0  & 32.71 & 0.871
# 1  & 28.85 & 0.791
# 2  & 31.70 & 0.897
# 3  & 26.05 & 0.701
# 4  & 33.37 & 0.900
 
# 0  & 25.77 & 0.686
# 1  & 23.67 & 0.477
# 2  & 20.64 & 0.551
# 3  & 20.46 & 0.390
# 4  & 25.88 & 0.657



#TVAL3
#TVAL3
# 0  & 31.72& 0.897
# 1  & 28.09 & 0.796
# 2  & 28.40 & 0.842
# 3  & 28.01 & 0.777
# 4  & 32.57 & 0.901
#TVAL3
# 0  & 29.54 & 0.833
# 1  & 26.64 & 0.726
# 2  & 26.20 & 0.789
# 3  & 26.80 & 0.717
# 4  & 30.20 & 0.849
#TVAL3
# 0  & 28.40 & 0.802
# 1  & 25.86 & 0.681
# 2  & 25.08 & 0.745
# 3  & 26.08 & 0.675
# 4  & 29.10 & 0.820

# #Expt 5
# parallel -j 8 'mkdir map_single_pixel/{}' ::: {0..4}
# parallel -j 8 'mkdir map_single_pixel/cleaned_img/' ::: {0..4}
# python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 3.5 \
# -d data/BSDS_Cropped/img_data.mat \
# -p '/home/cplab-ws1/ride/code/map_single_pixel/' \
# -m 0.9 \
# -r 350 \
# -e 3.5 \
# -f 1 \
# -K 5 \
# -y 0.15 | tee map_single_pixel/0p15_log.txt 

# #Expt 6
# parallel -j 8 'mkdir map_single_pixel/{}' ::: {3}
# parallel -j 8 'mkdir map_single_pixel/cleaned_img/' ::: {0..4}
# python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l  \
# -d data/BSDS_Cropped/img_data.mat \
# -p '/home/cplab-ws1/ride/code/map_single_pixel/' \
# -m 0.9 \
# -r 350 \
# -e 3.5 \
# -f 1 \
# -K 5 \
# -y 2.5 | tee map_single_pixel/0p15_log.txt 

# #Expt 7 No entropy
# parallel -j 8 'mkdir map_single_pixel/{}' ::: 0
# parallel -j 8 'mkdir map_single_pixel/cleaned_img/' ::: 0
# python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -p '/home/cplab-ws1/ride/code/map_single_pixel/' \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 0 \
# -K 1 \
# -y 2.5 | tee map_single_pixel/exp7_log.txt 


#Expt 8 With  entropy 3.5
# parallel -j 8 'mkdir map_single_pixel/{}' ::: 0
# parallel -j 8 'mkdir map_single_pixel/cleaned_img/' ::: 0
# python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -p '/home/cplab-ws1/ride/code/map_single_pixel/' \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 0 \
# -K 1 \
# -y 2.5 | tee map_single_pixel/exp7_log.txt 

# #Expt 9 With  entropy 2.5
# parallel -j 8 'mkdir map_single_pixel/{}' ::: 0
# parallel -j 8 'mkdir map_single_pixel/cleaned_img/' ::: 0
# python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -p '/home/cplab-ws1/ride/code/map_single_pixel/' \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 0 \
# -K 1 \
# -y 2.5 | tee map_single_pixel/exp9_log.txt 

# #Expt 10 With noise
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: {1..3}  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/' ::: {1..3} ::: 0.4  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/{3}' ::: {1..3} ::: 0.4 ::: {0..4} 
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/cleaned_img/' ::: {1..3} ::: 0.4
# parallel -j 8 'python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 0 \
# -K 5 \
# -n {1} \
# -y {2} | tee map_single_pixel/exp10_log.txt' ::: {1..3} ::: 0.4

# #Expt 11 House color
# mkdir map_single_pixel/0.15/  
# parallel 'mkdir map_single_pixel/0.15/{}/'  ::: {0..11} 
# mkdir map_single_pixel/0.15/cleaned_img/ 
# python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 128 \
# -l 5.0 \
# -d data/bird_rgb.mat \
# -m 0.9 \
# -r 0 \
# -e 100 \
# -f 1 \
# -K 12 \
# -n -1 \
# -N 610 \
# -y 0.15 | tee map_single_pixel/exp_bird_p5_log.txt

# #Expt 12 With noiseand projection
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 1  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/' ::: 1 ::: 0.4  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/{3}' ::: 1 ::: 0.4 ::: 0 
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/cleaned_img/' ::: 1 ::: 0.4
# parallel -j 8 'python experiments/map_single_pixel_noisy.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 0 \
# -K 1 \
# -n {1} \
# -y {2} | tee map_single_pixel/exp12_log.txt' ::: 1 ::: 0.4

# #Expt 13 With noise No entropy and correct lamda and projection
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 1  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/' ::: 1 ::: 0.4  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/{3}' ::: 1 ::: 0.4 ::: 0 
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/cleaned_img/' ::: 1 ::: 0.4
# parallel -j 8 'python experiments/map_single_pixel_noisy.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 2.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 4.0 \
# -f 0 \
# -K 1 \
# -n {1} \
# -y {2} | tee map_single_pixel/exp12_log.txt' ::: 1 ::: 0.4

# #Expt 14 With noise entropy 4 and lamda = 1 and projection
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 1  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/' ::: 1 ::: 0.4  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/{3}' ::: 1 ::: 0.4 ::: {0..1} 
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/cleaned_img/' ::: 1 ::: 0.4
# parallel -j 8 'python experiments/map_single_pixel_noisy.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 2.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 4.0 \
# -f 0 \
# -K 2 \
# -n {1} \
# -y {2} | tee map_single_pixel/exp12_log.txt' ::: 1 ::: 0.4

# #Expt 15 With noise entropy 3.5 and lamda = 2.0, and projection
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 1  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/' ::: 1 ::: 0.4  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/{3}' ::: 1 ::: 0.4 ::: 0 
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/cleaned_img/' ::: 1 ::: 0.4
# parallel -j 8 'python experiments/map_single_pixel_noisy.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 0 \
# -K 1 \
# -n {1} \
# -y {2} | tee map_single_pixel/exp12_log.txt' ::: 1 ::: 0.4


# #Expt 16 With noise entropy 3.5,lamda = 0.5,lower learning rate, and projection
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 1  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/' ::: 1 ::: 0.4  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/{3}' ::: 1 ::: 0.4 ::: 0 
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/cleaned_img/' ::: 1 ::: 0.4
# parallel -j 8 'python experiments/map_single_pixel_noisy.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 2.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 0 \
# -K 1 \
# -n {1} \
# -y {2} | tee map_single_pixel/exp12_log.txt' ::: 1 ::: 0.4

# #Expt 17 With noise entropy 3.5,lamda = 2.0,higher learning rate, and projection
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 1  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/' ::: 1 ::: 0.4  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/{3}' ::: 1 ::: 0.4 ::: 0 
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/cleaned_img/' ::: 1 ::: 0.4
# parallel -j 8 'python experiments/map_single_pixel_noisy.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 10.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 0 \
# -K 1 \
# -n {1} \
# -y {2} | tee map_single_pixel/exp12_log.txt' ::: 1 ::: 0.4

# #Expt 18 With noise entropy 3.5,lamda = 1.0 and projection
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 1  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/' ::: 1 ::: 0.4  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/{3}' ::: 1 ::: 0.4 ::: 0 
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/cleaned_img/' ::: 1 ::: 0.4
# parallel -j 8 'python experiments/map_single_pixel_noisy.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 0 \
# -K 1 \
# -n {1} \
# -y {2} | tee map_single_pixel/exp12_log.txt' ::: 1 ::: 0.4

# #Expt 19 With no entropy lamda = 1.0
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 1  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/' ::: 1 ::: 0.4  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/{3}' ::: 1 ::: 0.4 ::: 0 
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/cleaned_img/' ::: 1 ::: 0.4
# parallel -j 8 'python experiments/map_single_pixel_noisy.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 1000 \
# -f 0 \
# -K 1 \
# -n {1} \
# -y {2}' ::: 1 ::: 0.4

# #Expt 20 With no entropy lamda = 1.0 
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 1  	
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/' ::: 1 ::: 0.4  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/{3}' ::: 1 ::: 0.4 ::: 0 
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/cleaned_img/' ::: 1 ::: 0.4
# parallel -j 8 'python experiments/map_single_pixel_noisy.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 100 \
# -f 0 \
# -K 1 \
# -n {1} \
# -y {2}' ::: 1 ::: 0.4

# #Expt 21 With no entropy lamda = 1.0 clipping projection 
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 1  	
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/' ::: 1 ::: 0.4  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/{3}' ::: 1 ::: 0.4 ::: 0 
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/cleaned_img/' ::: 1 ::: 0.4
# parallel -j 8 'python experiments/map_single_pixel_noisy.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 100 \
# -f 0 \
# -K 1 \
# -n {1} \
# -y {2}' ::: 1 ::: 0.4
		
# #Expt 22 With entropy 100 lamda = 0.0 clipping projection 
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 1  	
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/' ::: 1 ::: 0.4  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/{3}' ::: 1 ::: 0.4 ::: 0 
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/cleaned_img/' ::: 1 ::: 0.4
# parallel -j 8 'python experiments/map_single_pixel_noisy.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 2.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 100 \
# -f 0 \
# -K 1 \
# -n {1} \
# -y {2}' ::: 1 ::: 0.4

# #Expt 23 With entropy 100 lamda = 0.0 clipping projection sigma/2
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 1  	
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/' ::: 1 ::: 0.4  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/{3}' ::: 1 ::: 0.4 ::: 0 
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/cleaned_img/' ::: 1 ::: 0.4
# parallel -j 8 'python experiments/map_single_pixel_noisy.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 2.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 100 \
# -f 0 \
# -K 1 \
# -n {1} \
# -y {2}' ::: 1 ::: 0.4

# #Expt 24 With entropy 100 lamda = 0.0 clipping projection sigma/2
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 1  	
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/' ::: 1 ::: 0.4  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/{3}' ::: 1 ::: 0.4 ::: {0..4} 
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}/cleaned_img/' ::: 1 ::: 0.4
# parallel -j 8 'python experiments/map_single_pixel_noisy.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 160 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 0 \
# -K 5 \
# -n {1} \
# -N 500 \
# -y {2}' ::: 1 ::: 0.4

# #Expt 25 bird color 0.1
# mkdir map_single_pixel/0.1/  
# parallel 'mkdir map_single_pixel/0.10/{}/'  ::: {0..11} 
# mkdir map_single_pixel/0.1/cleaned_img/ 
# python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 128 \
# -l 5.0 \
# -d data/bird_rgb.mat \
# -m 0.9 \
# -r 0 \
# -e 100 \
# -f 1 \
# -K 12 \
# -n -1 \
# -N 610 \
# -y 0.1 | tee map_single_pixel/exp_bird_p1_log.txt

# #Expt 25 bird color 0.1
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 0.1  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}' ::: 0.1 ::: {0..11} 
# parallel -j 8 'mkdir map_single_pixel/{1}/cleaned_img/' ::: 0.1
# python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 128 \
# -l 5.0 \
# -d data/bird_rgb.mat \
# -m 0.9 \
# -r 0 \
# -e 100 \
# -f 1 \
# -K 12 \
# -n -1 \
# -N 610 \
# -y 0.1


# #Expt 26 bird color 7 margin
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 0.1  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}' ::: 0.1 ::: {0..11} 
# parallel -j 8 'mkdir map_single_pixel/{1}/cleaned_img/' ::: 0.1
# python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 128 \
# -l 7.0 \
# -d data/bird_rgb.mat \
# -m 0.9 \
# -r 0 \
# -e 100 \
# -f 1 \
# -K 12 \
# -n -1 \
# -N 2010 \
# -y 0.1

# #Expt 27 bird color 7 margin
# parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 0.05  
# parallel -j 8 'mkdir map_single_pixel/{1}/{2}' ::: 0.05 ::: {0..11} 
# parallel -j 8 'mkdir map_single_pixel/{1}/cleaned_img/' ::: 0.05
# python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 128 \
# -l 7.0 \
# -d data/bird_rgb.mat \
# -m 0.9 \
# -r 0 \
# -e 100 \
# -f 1 \
# -K 12 \
# -n -1 \
# -N 2010 \
# -y 0.05

#Expt 27 bird color 7 margin 7%
parallel -j 8 'mkdir map_single_pixel/{1}/' ::: 0.07  
parallel -j 8 'mkdir map_single_pixel/{1}/{2}' ::: 0.07 ::: {0..11} 
parallel -j 8 'mkdir map_single_pixel/{1}/cleaned_img/' ::: 0.07
python experiments/map_single_pixel_stack.py results/12/12/rim.10082016.221222.xpck \
-H 0.7 \
-i 1 \
-s 128 \
-l 7.0 \
-d data/bird_rgb.mat \
-m 0.9 \
-r 0 \
-e 100 \
-f 1 \
-K 12 \
-n -1 \
-N 2110 \
-y 0.07