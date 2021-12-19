import cv2
import glob

#cứ 8 giây thì lấy 1 frame
fps = 8

#tổng video có trong folder dataset
folders = glob.glob('path/to/videos/dataset')
videonames_list = []
for folder in folders:
    for f in glob.glob(folder+'/*.mp4'):
      videonames_list.append(f)
      print(f)
print('There are {} videos in Folder'.format(len(videonames_list)))

#sử dụng vòng lặp để lấy frame
count = 0
for i in range(0,len(videonames_list)):
  video_data = videonames_list[i]
  video = cv2.VideoCapture(video_data)
  success = True
  while success:
    success,image = video.read()
    name = 'path/to/frames/dataset/'+str(count)+'.jpg'
    if success == True:
      if count % fps == 0:
        cv2.imwrite(name,image)
        print('Frame {} Extracted Successfully'.format(count))
      count+=1
    else:
      i = i+1
    i = i+1
  print('\n\n\nVideo {} Extracted Successfully\n\n\n'.format(video_data))
