import cv2
import pyzed.sl as sl
import torch
from depth_anything_v2.dpt import DepthAnythingV2
import numpy as np
import time

def init_camera(fps=30):
    cam = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_fps = fps  # FPS 설정
    if cam.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Camera open failed")
        exit(1)
    return cam

def load_model():
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vits'  # 선택할 인코더 ('vitb', 'vitl', 'vitg' 가능)

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'/home/lcw/Desktop/pyPro/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    return model

def preprocess_image(frame, input_size=308):
    """
    입력 이미지의 크기를 조절하고 전처리하는 함수
    """
    # BGR -> RGB 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 입력 크기만큼 이미지 크기 조정
    frame_resized = cv2.resize(frame_rgb, (input_size, input_size))
    # float32 변환 및 정규화
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    # HWC -> CHW 변환
    frame_transposed = torch.from_numpy(frame_normalized).permute(2, 0, 1)
    # 배치 차원 추가
    frame_batched = frame_transposed.unsqueeze(0)
    return frame_batched

# 깊이 맵 시각화 함수
def visualize_depth_map(depth_map, ori_shape):
    depth_map_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_visual = cv2.convertScaleAbs(depth_map_visual)
    depth_map_visual = cv2.applyColorMap(depth_map_visual, cv2.COLORMAP_JET)
    depth_map_visual = cv2.resize(depth_map_visual, ori_shape[::-1])
    return depth_map_visual

def process_stream(cam, model, target_fps=30, image_size=308):
    image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    frame_duration = 1.0 / target_fps  # 프레임당 목표 시간(초)
    inference_times = []  # 인퍼런스 시간을 저장할 리스트
    toral_frame_times=[]
    first_inference_done = False  # 첫 번째 인퍼런스가 완료되었는지 확인하는 플래그

    while True:
        start_time = time.time()  # 시작 시간 기록

        if cam.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # 인퍼런스 시간 측정
            inference_start_time = time.time()
            
            cam.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            ori_shape = frame.shape[:2]

            # 이미지 전처리 (image_size를 사용)
            input_tensor = preprocess_image(frame, input_size=image_size)
            input_tensor = input_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

            # 모델 추론
            with torch.no_grad():
                depth_map = model(input_tensor)

            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time

            # 첫 번째 인퍼런스는 제외하고 나머지를 기록
            if first_inference_done:
                inference_times.append(inference_time)
            else:
                first_inference_done = True  # 첫 번째 인퍼런스가 끝났음을 기록

            # 깊이 맵 처리
            depth_map = depth_map.squeeze().cpu().numpy()
            depth_map = depth_map.reshape((1, image_size, image_size))

            depth_map_vis = visualize_depth_map(depth_map[0], ori_shape)

            # depth_map_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            # depth_map_vis = cv2.convertScaleAbs(depth_map_vis)
            # depth_map_vis = cv2.applyColorMap(depth_map_vis, cv2.COLORMAP_JET)
            # depth_map_vis = cv2.resize(depth_map_vis, ori_shape[::-1])

            # 깊이 맵 시각화 및 원본 이미지 표시
            cv2.imshow("Depth Map", depth_map_vis)
            cv2.imshow("Left Camera Image", frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 실행 시간 계산 및 딜레이 추가
        elapsed_time = time.time() - start_time
        toral_frame_times.append(elapsed_time)
        if elapsed_time < frame_duration:
            time.sleep(frame_duration - elapsed_time)

    cam.close()
    cv2.destroyAllWindows()

    # 첫 번째 인퍼런스를 제외한 평균 인퍼런스 타임 계산
    if inference_times:  # 리스트가 비어있지 않으면 평균 계산
        average_inference_time = sum(inference_times) / len(inference_times)
        average_frame_time = sum(toral_frame_times) / len(toral_frame_times)
        fps = 1 / average_frame_time

        # 평균 인퍼런스 타임과 FPS 출력
        print(f"Average inference time (excluding first inference): {average_inference_time:.4f} seconds")
        print(f"Average FPS: {fps:.2f} FPS")
    else:
        print("No inference times recorded.")

if __name__ == "__main__":
    fps = 30  # 원하는 FPS 설정
    image_size = 406  # 308, 364, 518 등으로 변경 가능 input_size must be divisible by 14
    cam = init_camera(fps=fps)
    model = load_model()
    process_stream(cam, model, target_fps=fps, image_size=image_size)