import pyzed.sl as sl
import cv2

def main():
    # ZED 카메라 초기화
    zed = sl.Camera()

    # 카메라 설정
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 해상도 설정
    init_params.camera_fps = 30  # FPS 설정

    # 카메라 열기
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f'Failed to open ZED camera: {err}')
        exit(1)

    # 이미지 저장을 위한 Mat 객체 생성
    left_image = sl.Mat()
    right_image = sl.Mat()

    # 루프 실행하여 이미지 표시
    key = ''
    while key != 113:  # 'q'를 누를 때까지 계속
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # 좌우 이미지 취득
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)

            # OpenCV를 사용하여 이미지 표시
            cv2.imshow('Left Image', left_image.get_data())
            cv2.imshow('Right Image', right_image.get_data())

            key = cv2.waitKey(10)
        else:
            print("Failed to grab")

    # 카메라 종료
    zed.close()

if __name__ == "__main__":
    main()
