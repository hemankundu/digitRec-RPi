import cv2

def save(gray):
    file_name = input('[input.png]: ')
    if file_name == '':
        file_name = 'input.png'
    elif '.' not in file_name:
        file_name += '.png'
    cv2.imwrite(file_name, gray)
    print(file_name, 'saved..')

def capture_with_display(cap):
    cv2.namedWindow('frame')
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        key = cv2.waitKey(10)
        if  key == ord('q'):
                cv2.destroyWindow('frame')
                cv2.waitKey(5)
                break
        elif key == ord('\n') or key == ord('\r'):
            cv2.destroyWindow('frame')
            cv2.waitKey(5)
            save(gray)
            break

def capture_without_display(cap):
    while True:
        in_str = input('[capture]/exit: ')
        if  in_str == '' or in_str == 'capture':
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            save(gray)
            break
        elif in_str == 'exit':
            break

def capture(cam_id, display, size):
    w, h = size
    #cam_id = int(input('camera id: '))
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("failed to open camera!")
        exit(0)
    print('Opened camera ',cam_id, ', size ', size)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    #display = True
    if display:
        capture_with_display(cap)
    else:
        capture_without_display(cap)
    cap.release()
    print('capture exiting..')
