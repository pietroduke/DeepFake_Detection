import splitfolders

splitfolders.ratio('path/to/faces/dataset/before/splited',
                   output="path/to/faces/dataset/after/splited",
                   ratio=(0.8, 0.2)) #tỉ lệ chia cho train:val = 0.8:0.2
