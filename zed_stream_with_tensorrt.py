import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # CUDA 초기화
import pyzed.sl as sl
import time
import torch

# ZED 카메라 초기화 함수
def init_camera(fps=30):
    cam = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_fps = fps  # FPS 설정
    if cam.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("카메라 열기 실패")
        exit(1)
    return cam

# TensorRT 엔진 로드 함수
def load_trt_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print("TensorRT 엔진 로드 실패")
        exit(1)
    return engine

# 이미지 전처리 함수
def preprocess_image(frame, input_size):
    """
    입력 이미지의 크기를 조절하고 전처리하는 함수
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_size, input_size))
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    frame_transposed = torch.from_numpy(frame_normalized).permute(2, 0, 1)
    frame_batched = frame_transposed.unsqueeze(0)
    return frame_batched.flatten()

# TensorRT 추론 함수
def trt_infer(context, bindings, d_input, h_input, d_output, h_output, stream):
    # 입력 데이터를 호스트에서 디바이스로 복사 (Pinned memory 사용)
    cuda.memcpy_htod_async(d_input, h_input, stream)
    start_time = time.time()

    # 추론 실행
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # 출력 데이터를 디바이스에서 호스트로 복사 (Pinned memory 사용)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()  # 비동기 작업 완료 대기
    end_time = time.time()

    inference_time = end_time - start_time
    return inference_time

# 깊이 맵 시각화 함수
def visualize_depth_map(depth_map, ori_shape):
    depth_map_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_visual = cv2.convertScaleAbs(depth_map_visual)
    depth_map_visual = cv2.applyColorMap(depth_map_visual, cv2.COLORMAP_JET)
    depth_map_visual = cv2.resize(depth_map_visual, ori_shape[::-1])
    return depth_map_visual

# 실시간 스트림 처리 함수
def process_stream(cam, engine, context, target_fps=30, image_size=518):
    image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    frame_duration = 1.0 / target_fps

    # 입력/출력 버퍼 설정 (Pinned memory 사용)
    h_input = cuda.pagelocked_empty(trt.volume((1, 3, image_size, image_size)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # 바인딩 및 스트림 설정
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()

    # 추론 시간 측정을 위한 변수 초기화
    inference_times = []
    total_frame_times = []
    first_inference_done = False  # 첫 번째 인퍼런스가 완료되었는지 확인하는 플래그

    while True:
        loop_start_time = time.time()

        if cam.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            ori_shape = frame.shape[:2]

            # 이미지 전처리
            input_tensor = preprocess_image(frame, image_size)
            np.copyto(h_input, input_tensor.cpu().numpy())  # 호스트 메모리에 입력 복사

            # 모델 추론 수행 및 시간 측정
            inference_time = trt_infer(context, bindings, d_input, h_input, d_output, h_output, stream)

            # 첫 번째 인퍼런스는 제외하고 나머지를 기록
            if first_inference_done:
                inference_times.append(inference_time)
            else:
                first_inference_done = True  # 첫 번째 인퍼런스가 끝났음을 기록

            # 출력 데이터 처리
            depth_map = h_output.reshape((1, image_size, image_size))

            # 깊이 맵 시각화
            depth_map_vis = visualize_depth_map(depth_map[0], ori_shape)

            # 결과 시각화
            cv2.imshow("Depth Map", depth_map_vis)
            cv2.imshow("Left Camera Image", frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 한 프레임 처리 시간 계산
        frame_time = time.time() - loop_start_time
        total_frame_times.append(frame_time)

        # 프레임 주기보다 짧으면 대기
        if frame_time < frame_duration:
            time.sleep(frame_duration - frame_time)

    cam.close()
    cv2.destroyAllWindows()

    # 첫 번째 인퍼런스를 제외한 평균 인퍼런스 타임 및 프레임 타임 계산
    if inference_times:
        average_inference_time = sum(inference_times) / len(inference_times)
        average_frame_time = sum(total_frame_times) / len(total_frame_times)
        fps = 1 / average_frame_time

        # 평균 인퍼런스 타임과 FPS 출력
        print(f"Average inference time (excluding first inference): {average_inference_time:.4f} seconds")
        print(f"Average FPS (total frame time): {fps:.2f} FPS")
    else:
        print("No inference times recorded.")

# 메인 함수
if __name__ == "__main__":
    fps = 30  # 원하는 FPS 설정
    image_size = 518  # TensorRT 모델의 입력 크기
    engine_path = 'depth_anything_v2_vits_518.trt'  # TensorRT 엔진 경로

    # ZED 카메라 및 TensorRT 엔진 초기화
    cam = init_camera(fps=fps)
    engine = load_trt_engine(engine_path)
    context = engine.create_execution_context()

    # 실시간 스트림 처리
    process_stream(cam, engine, context, target_fps=fps, image_size=image_size)