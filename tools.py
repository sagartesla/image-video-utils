import numpy as np
import os, cv2


def read_raw_planar(y_path,shape):
    ch = shape[0]
    h = shape[1]
    w = shape[2]

    f = open(y_path, mode='rb')
    data = bytearray(f.read())

    img_serial_data = np.array(data)
    img_serial_data = img_serial_data.astype(np.uint8)
    frame_no = 0
    frame_len = ch * h * w
    total_frames = len(data) / frame_len
    print ('Total Frames: {}'.format(total_frames))
    while True:
        print(frame_no,frame_no*frame_len ,(frame_no+1)*frame_len)
        frame = img_serial_data[frame_no*frame_len : (frame_no+1)*frame_len]
        op = np.reshape(frame, newshape=shape)
        op = np.transpose(op,[1,2,0]).copy()
        cv2.putText(op, 'frame no: {}'.format(frame_no), (5, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 0), 2)
        cv2.namedWindow("read_raw_planar", 0)
        cv2.setWindowProperty("read_raw_inter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('read_raw_planar',op[:,:,[2,1,0]])
        frame_no = (frame_no + 1)%total_frames
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    exit(-1)


def read_raw_inter(y_path,shape):
    ch = shape[0]
    h = shape[1]
    w = shape[2]

    f = open(y_path, mode='rb')
    data = bytearray(f.read())

    img_serial_data = np.array(data)
    img_serial_data = img_serial_data.astype(np.uint8)
    frame_no = 0
    frame_len = ch * h * w
    total_frames = len(data) / frame_len
    print ('Total Frames: {}'.format(total_frames))

    while True:
        print (frame_no,frame_no*frame_len ,(frame_no+1)*frame_len)
        frame = img_serial_data[frame_no*frame_len : (frame_no+1)*frame_len]
        op = np.reshape(frame, newshape=(h,w,ch))
        cv2.putText(op,'frame no: {}'.format(frame_no),(5,55),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.5,(0,255,0),2)
        cv2.namedWindow("read_raw_inter", 0)
        cv2.setWindowProperty("read_raw_inter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('read_raw_inter',op[:,:,[2,1,0]])
        frame_no = (frame_no + 1)%total_frames
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    exit(-1)


def batch_raw_to_file_from_multiple_folder(folder_path):
    flist = os.listdir(folder_path)
    frame_number = 1
    for fl in flist:
        img_list = os.listdir(folder_path+fl)
        print ('Folder: {} Number of Images: {}'.format(fl,len(img_list)))
        for imgn in img_list:
            if not imgn.endswith('jpg'):
                continue
            image_filepath = folder_path + fl + '/' + imgn
            print(image_filepath)
            # BGR to RGB Conversion
            img = cv2.imread(image_filepath)[:,:,[2,1,0]]
            # img = cv2.resize(img, (48,48))
            cv2.imshow('batch_raw_to_file_from_multiple_folder',img)
            cv2.waitKey(1)
            height, width, ch = img.shape
            img = np.transpose(img, [2, 0, 1]) # For Interleaved to Planar Output
            img_q = np.reshape(img, (height * width * ch,)).astype(np.uint8)
            f = open('/tmp/raw.rgbp', mode='ab')
            f.write(img_q.tobytes())
            f.close()
            frame_number = frame_number + 1
    exit(-2)


def batch_raw_to_file_signed(folder_path):
    img_list = os.listdir(folder_path)
    frame_number = 0
    for imgn in img_list:
        if '.jpg' not in imgn:
            continue
        image_filepath = folder_path + imgn
        print(image_filepath)

        # BGR to RGB Conversion
        img = cv2.imread(image_filepath)[:,:,[2,1,0]]
        cv2.imshow('batch_raw_to_file_signed',img)
        cv2.waitKey(1)

        img = (img - 127.5) / 128.0
        height, width, ch = img.shape

        img = np.transpose(img, [2, 0, 1]) # For Interleaved to Planar Output
        img_q = np.reshape(img, (height * width * ch,)).astype(np.float32)

        output_raw_filepath = "/tmp/raw_singed.rgbp"
        f = open(output_raw_filepath, mode='ab')
        f.write(img_q.tobytes())
        f.close()
        print(frame_number)
        frame_number = frame_number + 1
    exit(-2)


def batch_raw_to_file(folder_path):
    img_list = os.listdir(folder_path)
    frame_number = 0
    for imgn in img_list:
        if '.jpg' not in imgn:
            continue
        image_filepath = folder_path + imgn
        print(image_filepath)

        # BGR to RGB Conversion
        img = cv2.imread(image_filepath)[:,:,[2,1,0]]
        img = cv2.resize(img, (48, 48))
        cv2.imshow('batch_raw_to_file',img)
        cv2.waitKey(1)

        height, width, ch = img.shape

        img = np.transpose(img, [2, 0, 1]) # For Interleaved to Planar Output
        img_q = np.reshape(img, (height * width * ch,)).astype(np.uint8)

        output_raw_filepath = "/tmp/raw.rgbp"
        f = open(output_raw_filepath, mode='ab')
        f.write(img_q.tobytes())
        f.close()
        print(frame_number)
        frame_number = frame_number + 1
    exit(-1)


def create_raw_planar(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    frame_no = 0
    while True:
        ret, img = cap.read()
        if ret == False:
            break
        img = img[:,:,[2,1,0]] #BGR to RGB
        height, width, ch = img.shape
        cv2.imshow('create_raw_planar',img[:,:,[2,1,0]])
        wk = cv2.waitKey(1)
        img = np.transpose(img, [2, 0, 1]) # For Interleaved to Planar Output
        f = open('/tmp/raw.rgbp', mode='ab')
        img_q = np.reshape(img, (height * width * ch,)).astype(np.uint8)
        f.write(img_q.tobytes())
        f.close()
        print('frame_{}'.format(frame_no))
        if frame_no == num_frames or wk & 0xff == ord('q'):
            print('RAW video created')
            cap.release()
            exit(-3)
        frame_no = frame_no + 1


def create_raw_inter(video_path ,num_frames):
    cap = cv2.VideoCapture(video_path)
    frame_no = 0
    while True:
        ret, img = cap.read()
        if ret == False:
            break
        img = img[:,:,[2,1,0]] #BGR to RGB
        height, width, ch = img.shape
        cv2.imshow('create_raw_inter',img[:,:,[2,1,0]])
        wk = cv2.waitKey(1)
        f = open('/tmp/raw.rgbi', mode='ab')
        img_q = np.reshape(img, (height * width * ch,)).astype(np.uint8)
        f.write(img_q.tobytes())
        f.close()
        print('frame_{}'.format(frame_no))
        if frame_no == num_frames or wk & 0xff == ord('q'):
            print('RAW video created')
            cap.release()
            exit(-3)
        frame_no = frame_no + 1



video_file_path = 'test.mp4'
raw_video_file_path = 'test_raw.rgbp'
input_images_folder_path = '/u/test/'

create_raw_planar(video_file_path, -1)
read_raw_planar(raw_video_file_path, (3,720,1280))

# create_raw_inter(video_file_path, -1)
# read_raw_inter(raw_video_file_path,(3,720,1280))

# batch_raw_to_file_from_multiple_folder(input_images_folder_path)
# batch_raw_to_file_signed(input_images_folder_path)
# batch_raw_to_file(input_images_folder_path)

