import torch
import tensorrt as trt
import os
from pathlib import Path
import time
from depth_anything_v2.dpt import DepthAnythingV2


# 1. 모델 로드 함수
def load_model(encoder: str, weights_path: str):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Load the model
    model = DepthAnythingV2(**model_configs[encoder]).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the weights
    model.load_state_dict(torch.load(weights_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()  # Set the model to evaluation mode
    
    return model


# 2. ONNX 변환 함수
def export_onnx(model, encoder: str, input_size: int):
    dummy_input = torch.ones((1, 3, input_size, input_size)).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    onnx_file_path = Path(f"depth_anything_v2_{encoder}_{input_size}.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'}, 'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
    )
    
    print(f"Model exported to {onnx_file_path}")
    return onnx_file_path


# 3. TensorRT 엔진 생성 함수
def get_engine(onnx_file_path, engine_file_path, input_size: int):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # ONNX 모델을 파싱
    with open(onnx_file_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # TensorRT 빌더 설정
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB 워크스페이스 설정
    
    # TensorRT 엔진 생성
    serialized_engine = builder.build_serialized_network(network, config)
    
    # 엔진 파일로 저장
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"TensorRT engine saved to {engine_file_path}")
    return serialized_engine


if __name__ == '__main__':
    image_size = 518  # Model input size
    encoder = 'vits'  # Select encoder
    weights_path = './depth_anything_v2_vits.pth'  # Weights file location

    # 1. 모델 로드
    model = load_model(encoder, weights_path)

    # 2. ONNX 변환
    onnx_file_path = export_onnx(model, encoder, input_size=image_size)

    # 3. TensorRT 엔진 생성
    engine_file_path = f'depth_anything_v2_{encoder}_{image_size}.trt'
    engine = get_engine(onnx_file_path, engine_file_path, input_size=image_size)

    if engine:
        print("TensorRT engine is ready and saved to file.")
    else:
        print("Failed to create the TensorRT engine.")


