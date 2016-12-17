# python experiments/map_interpolate_stack.py results/13/13/rim.11082016.163429.xpck -H 0.7 -i 1 -s 256 -l 0.5 -d data/house.png -m 0.9 -r 0 -I -1

#Expt 1 
# parallel -j 8 'mkdir map_interpolate/{}' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/cleaned_img/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/0/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/1/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/2/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/3/' ::: 1

# parallel -j 8 \
# 'python experiments/map_interpolate_stack.py results/13/13/rim.11082016.163429.xpck \
# -H 0.7 \
# -i 1 \
# -s 256 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 500 \
# -I {}' ::: 1

# #Expt 2 Starting with random image
# parallel -j 8 'mkdir map_interpolate/{}' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/cleaned_img/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/0/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/1/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/2/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/3/' ::: 1

# parallel -j 8 \
# 'python experiments/map_interpolate_stack.py results/13/13/rim.11082016.163429.xpck \
# -H 0.7 \
# -i 1 \
# -s 256 \
# -l 1.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 200 \
# -I {}' ::: 1

# #Expt 3 Max entropy 4.0
# parallel -j 8 'mkdir map_interpolate/{}' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/cleaned_img/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/0/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/1/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/2/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/3/' ::: 1

# parallel -j 8 \
# 'python experiments/map_interpolate_stack.py results/13/13/rim.11082016.163429.xpck \
# -H 0.7 \
# -i 1 \
# -s 256 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 4.0 \
# -I {}' ::: 1

# #Expt 4 Max entropy 3.5 and random directions
# parallel -j 8 'mkdir map_interpolate/{}' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/cleaned_img/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/0/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/1/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/2/' ::: 1
# parallel -j 8 'mkdir map_interpolate/{}/3/' ::: 1

# parallel -j 8 \
# 'python experiments/map_interpolate_stack.py results/13/13/rim.11082016.163429.xpck \
# -H 0.7 \
# -i 1 \
# -s 256 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 1 \
# -I {}' ::: 1

# #Expt 5 Same for next image
# parallel -j 8 'mkdir map_interpolate/{}' ::: 2
# parallel -j 8 'mkdir map_interpolate/{}/cleaned_img/' ::: 2
# parallel -j 8 'mkdir map_interpolate/{}/0/' ::: 2
# parallel -j 8 'mkdir map_interpolate/{}/1/' ::: 2
# parallel -j 8 'mkdir map_interpolate/{}/2/' ::: 2
# parallel -j 8 'mkdir map_interpolate/{}/3/' ::: 2

# parallel -j 8 \
# 'python experiments/map_interpolate_stack.py results/13/13/rim.11082016.163429.xpck \
# -H 0.7 \
# -i 1 \
# -s 256 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 1 \
# -I {}' ::: 2

# #Expt 6 Same for next image
# parallel -j 8 'mkdir map_interpolate/{}' ::: 3
# parallel -j 8 'mkdir map_interpolate/{}/cleaned_img/' ::: 3
# parallel -j 8 'mkdir map_interpolate/{}/0/' ::: 3
# parallel -j 8 'mkdir map_interpolate/{}/1/' ::: 3
# parallel -j 8 'mkdir map_interpolate/{}/2/' ::: 3
# parallel -j 8 'mkdir map_interpolate/{}/3/' ::: 3

# parallel -j 8 \
# 'python experiments/map_interpolate_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 256 \
# -l 3.5 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 3.0 \
# -f 1 \
# -I {}' ::: 3

# #Expt 7 Same for next image
# parallel -j 8 'mkdir map_interpolate/{}' ::: 9
# parallel -j 8 'mkdir map_interpolate/{}/cleaned_img/' ::: 9
# parallel -j 8 'mkdir map_interpolate/{}/0/' ::: 9
# parallel -j 8 'mkdir map_interpolate/{}/1/' ::: 9
# parallel -j 8 'mkdir map_interpolate/{}/2/' ::: 9
# parallel -j 8 'mkdir map_interpolate/{}/3/' ::: 9

# parallel -j 8 \
# 'python experiments/map_interpolate_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 256 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 1 \
# -I {}' ::: 9

# #Expt 8 Same for next image
# parallel -j 8 'mkdir map_interpolate/{}' ::: 0
# parallel -j 8 'mkdir map_interpolate/{}/cleaned_img/' ::: 0
# parallel -j 8 'mkdir map_interpolate/{}/0/' ::: 0
# parallel -j 8 'mkdir map_interpolate/{}/1/' ::: 0
# parallel -j 8 'mkdir map_interpolate/{}/2/' ::: 0
# parallel -j 8 'mkdir map_interpolate/{}/3/' ::: 0

# parallel -j 8 \
# 'python experiments/map_interpolate_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 256 \
# -l 5.0 \
# -d data/BSDS_Cropped/img_data.mat \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 1 \
# -I {}' ::: 7

# #Expt 9 House Color
# parallel -j 8 'mkdir map_interpolate/{}' ::: {0..2}
# parallel -j 8 'mkdir map_interpolate/{}/cleaned_img/' ::: {0..2}
# parallel -j 8 'mkdir map_interpolate/{}/0/' ::: {0..2}
# parallel -j 8 'mkdir map_interpolate/{}/1/' ::: {0..2}
# parallel -j 8 'mkdir map_interpolate/{}/2/' ::: {0..2}
# parallel -j 8 'mkdir map_interpolate/{}/3/' ::: {0..2}

# parallel -j 8 \
# 'python experiments/map_interpolate_stack.py results/12/12/rim.10082016.221222.xpck \
# -H 0.7 \
# -i 1 \
# -s 256 \
# -l 5.0 \
# -d data/house_color.mat \
# -m 0.9 \
# -r 0 \
# -e 3.5 \
# -f 1 \
# -I {}' ::: {0..2}

