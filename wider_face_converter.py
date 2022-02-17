# Read file
length = 0

with open('C:/Users/Admin/Desktop/mAP/detection_wider_mtcnn_3.txt', 'r') as f:
# with open('C:/Users/Admin/Desktop/mAP/ground_wider_3.txt', 'r') as f:

    lines = f.readlines()

    # Get index of images
    image_id = []
    for index, line in enumerate(lines):
        length += 1
        if 'jpg' in line:
            image_id.append(index)

f.close()

print('Dataset has {} images'.format(len(image_id)))

# Processing
print('Processing...')
for id in image_id:

    # Get new file name
    file_name = lines[id].rstrip(".jpg\n")
    file_name = file_name.split('/')
    file_name = file_name[3] + '.txt'

    # Create new file and write data
    # with open('mAP_Interview_dnn/input/ground-truth/' + file_name, 'w') as f:
    with open('mAP_Interview_mtcnn/input/detection-results/' + file_name, 'w') as f:

        id = id + 1
        while True:
            id += 1
            if (id == length) or ('jpg' in lines[id]):
                break
            line = lines[id].split()
            # new_line = 'face ' + str(line[0]) + ' ' + str(line[1]) + ' ' \
            #            + str(int(line[0]) + int(line[2])) + ' ' \
            #            + str(int(line[1]) + int(line[3])) + '\n'
            new_line = 'face 0.5 ' + str(line[0]) + ' ' + str(line[1]) + ' ' + str(line[2]) + ' ' + str(line[3]) + ' ' + '\n'
            f.write(new_line)

    f.close()

print('Done!')