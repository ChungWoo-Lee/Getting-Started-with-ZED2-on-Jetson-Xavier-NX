import tensorrt as trt
import pyzed.sl as sl
import numpy as np
import ctypes

# TensorRT 로깅 및 플러그인 설정
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer(engine, input_image):
    with engine.create_execution_context() as context:
        # 입력 데이터를 TensorRT 엔진에 맞게 변환
        input_image = input_image.astype(np.float32)

        # 배치 크기와 추론 관련 설정 후 실행
        # 컨텍스트를 이용해 엔진에서 추론 실행

# ZED 카메라 스트리밍 및 추론 실행
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # 성능 모드로 깊이 설정
init_params.coordinate_units = sl.UNIT.METER        # 단위를 미터로 설정
zed.open(init_params)

runtime_params = sl.RuntimeParameters()

# TensorRT 엔진 로드
engine = load_engine('depth_anything_v2.engine')

while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        # ZED 카메라에서 이미지 가져오기
        image = sl.Mat()
        zed.retrieve_image(image, sl.VIEW.LEFT)
        image_np = image.get_data()

        # TensorRT 엔진으로 깊이 추정
        depth_estimation = infer(engine, image_np)

        # 추정 결과 처리 및 시각화
