import splitfolders

splitfolders.ratio('dataset_faces/CelebV2Faces/train_val',
                   output="dataset_faces/DatasetCELEB_train",
                   ratio=(0.8, 0.2)) #tỉ lệ chia cho train:val = 0.8:0.2