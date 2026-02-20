"""
Wav2Lip 推理 API - 适配 M1 Mac (MPS) + CUDA + CPU
供 Node.js 通过 subprocess 调用，输出 JSON 格式进度和结果
"""
import sys
import os
import json
import argparse
import time
import subprocess
import platform
import ssl

# 修复 macOS SSL 证书问题
if platform.system() == 'Darwin':
    try:
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
    except ImportError:
        ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import cv2
import torch

# 添加当前目录到 path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# 设置本地模型缓存目录，避免运行时从网络下载
LOCAL_CACHE_DIR = os.path.join(SCRIPT_DIR, 'face_detection', 'checkpoints')
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
# 让 PyTorch hub 从本地加载模型
os.environ['TORCH_HOME'] = os.path.join(SCRIPT_DIR, 'face_detection')
# torch hub 会在 TORCH_HOME/hub/checkpoints 里找模型
TORCH_HUB_CACHE = os.path.join(SCRIPT_DIR, 'face_detection', 'hub', 'checkpoints')
if not os.path.exists(TORCH_HUB_CACHE):
    os.makedirs(TORCH_HUB_CACHE, exist_ok=True)
    # 如果本地已有模型文件，建立软链接或复制
    for fname in os.listdir(LOCAL_CACHE_DIR):
        src = os.path.join(LOCAL_CACHE_DIR, fname)
        dst = os.path.join(TORCH_HUB_CACHE, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except:
                import shutil
                shutil.copy2(src, dst)

import audio
from models import Wav2Lip
import face_detection

# ==================== 工具函数 ====================

def emit(msg_type, data):
    """输出 JSON 消息给 Node.js"""
    print(json.dumps({"type": msg_type, **data}, ensure_ascii=False), flush=True)

def emit_progress(percent, message):
    emit("progress", {"percent": percent, "message": message})

def emit_error(message):
    emit("error", {"message": message})

def emit_result(data):
    emit("result", data)

def get_device():
    """获取最佳可用设备"""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

# ==================== 核心推理 ====================

mel_step_size = 16

def load_model(checkpoint_path, device):
    """加载 Wav2Lip 模型"""
    model = Wav2Lip()
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    s = checkpoint["state_dict"]
    new_s = {k.replace('module.', ''): v for k, v in s.items()}
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images, device, batch_size=8, nosmooth=False, pads=(0, 10, 0, 0)):
    """人脸检测 - 快速模式：只检测少量采样帧，其余帧复用/插值"""
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D,
        flip_input=False,
        device='cpu'
    )
    
    n_frames = len(images)
    
    # 快速模式：采样检测（每隔 sample_rate 帧检测一次）
    if n_frames <= 10:
        sample_indices = list(range(n_frames))
    else:
        # 每 N 帧采样一次，保证首尾都检测
        sample_rate = max(1, n_frames // 8)  # 最多检测 ~8 次
        sample_indices = list(range(0, n_frames, sample_rate))
        if sample_indices[-1] != n_frames - 1:
            sample_indices.append(n_frames - 1)
    
    emit_progress(27, f"采样 {len(sample_indices)}/{n_frames} 帧进行人脸检测...")
    
    sampled_images = [images[i] for i in sample_indices]
    
    predictions = []
    while True:
        try:
            for i in range(0, len(sampled_images), batch_size):
                predictions.extend(
                    detector.get_detections_for_batch(
                        np.array(sampled_images[i:i + batch_size])
                    )
                )
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError('图像太大，无法进行人脸检测')
            batch_size //= 2
            continue
        break
    
    del detector
    
    # 将采样检测结果扩展到所有帧
    pady1, pady2, padx1, padx2 = pads
    sampled_boxes = []
    for rect, image in zip(predictions, sampled_images):
        if rect is None:
            raise ValueError('未检测到人脸！请确保视频中包含清晰的人脸。')
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        sampled_boxes.append([x1, y1, x2, y2])
    
    # 插值到所有帧
    all_boxes = np.zeros((n_frames, 4))
    for i in range(len(sample_indices) - 1):
        idx_a = sample_indices[i]
        idx_b = sample_indices[i + 1]
        box_a = np.array(sampled_boxes[i])
        box_b = np.array(sampled_boxes[i + 1])
        for j in range(idx_a, idx_b + 1):
            t = (j - idx_a) / max(1, idx_b - idx_a)
            all_boxes[j] = box_a * (1 - t) + box_b * t
    # 确保最后一帧
    all_boxes[sample_indices[-1]] = sampled_boxes[-1]
    
    if not nosmooth:
        all_boxes = get_smoothened_boxes(all_boxes, T=5)
    
    results = [
        [image[int(y1):int(y2), int(x1):int(x2)], (int(y1), int(y2), int(x1), int(x2))]
        for image, (x1, y1, x2, y2) in zip(images, all_boxes)
    ]

    return results

def datagen(frames, mels, face_det_results, img_size=96, wav2lip_batch_size=32):
    """数据生成器"""
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    for i, m in enumerate(mels):
        idx = i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()
        face = cv2.resize(face, (img_size, img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
        img_masked = img_batch.copy()
        img_masked[:, img_size // 2:] = 0
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch, coords_batch


def run_inference(args):
    """主推理流程"""
    device = get_device()
    emit_progress(0, f"使用设备: {device}")
    
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        # 尝试默认路径
        checkpoint_path = os.path.join(SCRIPT_DIR, 'checkpoints', 'wav2lip_gan.pth')
    
    if not os.path.exists(checkpoint_path):
        emit_error("模型文件不存在，请先下载 wav2lip_gan.pth")
        return
    
    face_path = args.face
    audio_path = args.audio
    output_path = args.output
    
    if not os.path.exists(face_path):
        emit_error(f"视频文件不存在: {face_path}")
        return
    if not os.path.exists(audio_path):
        emit_error(f"音频文件不存在: {audio_path}")
        return

    # ---- 1. 读取视频帧 ----
    emit_progress(5, "读取视频帧...")
    
    is_image = face_path.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
    
    if is_image:
        full_frames = [cv2.imread(face_path)]
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(face_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (
                    frame.shape[1] // args.resize_factor,
                    frame.shape[0] // args.resize_factor
                ))
            full_frames.append(frame)
        
    if len(full_frames) == 0:
        emit_error("无法读取视频帧")
        return
    
    emit_progress(10, f"读取了 {len(full_frames)} 帧 ({fps:.1f} FPS)")
    
    # ---- 2. 处理音频 ----
    emit_progress(15, "处理音频...")
    
    temp_dir = os.path.join(SCRIPT_DIR, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # 如果不是 wav 格式，转换
    wav_path = audio_path
    if not audio_path.lower().endswith('.wav'):
        wav_path = os.path.join(temp_dir, 'temp_audio.wav')
        ffmpeg_cmd = os.environ.get('FFMPEG_PATH', 'ffmpeg')
        subprocess.call([
            ffmpeg_cmd, '-y', '-i', audio_path, '-ar', '16000', '-ac', '1', wav_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    wav = audio.load_wav(wav_path, 16000)
    mel = audio.melspectrogram(wav)
    
    if np.isnan(mel.reshape(-1)).sum() > 0:
        emit_error("音频 Mel 频谱包含 NaN 值，请检查音频文件")
        return
    
    # 分割 mel 块
    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1
    
    emit_progress(20, f"音频处理完成: {len(mel_chunks)} 个片段")
    
    # 截取帧数以匹配音频
    full_frames = full_frames[:len(mel_chunks)]
    
    # ---- 3. 人脸检测 ----
    emit_progress(25, "检测人脸...")
    
    try:
        face_det_results = face_detect(
            full_frames,
            device,
            batch_size=args.face_det_batch_size,
            pads=tuple(args.pads)
        )
    except ValueError as e:
        emit_error(str(e))
        return
    
    emit_progress(40, f"人脸检测完成")
    
    # ---- 4. 加载模型 ----
    emit_progress(45, "加载 Wav2Lip 模型...")
    model = load_model(checkpoint_path, device)
    emit_progress(50, "模型加载完成")
    
    # ---- 5. 推理 ----
    emit_progress(55, "开始口型合成...")
    
    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks, face_det_results, wav2lip_batch_size=batch_size)
    
    total_batches = int(np.ceil(float(len(mel_chunks)) / batch_size))
    
    frame_h, frame_w = full_frames[0].shape[:-1]
    temp_video = os.path.join(temp_dir, 'result_temp.avi')
    out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_w, frame_h))
    
    start_time = time.time()
    processed_frames = 0
    
    for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        
        with torch.no_grad():
            pred = model(mel_batch, img_batch)
        
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            
            # 遮罩融合：只替换下半脸（嘴巴区域），上半脸保留原始画面
            h_face = y2 - y1
            w_face = x2 - x1
            
            # 创建渐变遮罩：上方为 0（保留原图），下方为 1（使用生成结果）
            mask = np.zeros((h_face, w_face), dtype=np.float32)
            # 从 35% 高度开始渐变，到 55% 高度完全替换
            blend_start = int(h_face * 0.35)
            blend_end = int(h_face * 0.55)
            # 上方保留原图
            mask[:blend_start] = 0
            # 渐变区域
            if blend_end > blend_start:
                for row in range(blend_start, blend_end):
                    mask[row] = (row - blend_start) / (blend_end - blend_start)
            # 下方使用生成的嘴型
            mask[blend_end:] = 1.0
            
            # 水平方向也做羽化（边缘 15% 渐变）
            edge_w = max(1, int(w_face * 0.15))
            for col in range(edge_w):
                factor = col / edge_w
                mask[:, col] *= factor
                mask[:, w_face - 1 - col] *= factor
            
            # 高斯模糊遮罩使过渡更自然
            mask = cv2.GaussianBlur(mask, (15, 15), 5)
            mask = mask[:, :, np.newaxis]  # 扩展到3通道
            
            # 融合：original * (1 - mask) + generated * mask
            original_face = f[y1:y2, x1:x2].astype(np.float32)
            generated_face = p.astype(np.float32)
            blended = original_face * (1 - mask) + generated_face * mask
            
            # 可选：对生成区域轻微锐化
            # kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]]) / 2
            # blended_sharp = cv2.filter2D(blended, -1, kernel)
            # blended = original_face * (1 - mask) + blended_sharp * mask
            
            f[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
            out.write(f)
        
        processed_frames += len(frames)
        progress = 55 + int((i / total_batches) * 35)
        elapsed = time.time() - start_time
        fps_speed = processed_frames / max(elapsed, 0.001)
        emit_progress(min(progress, 90), f"合成中... {processed_frames}/{len(mel_chunks)} 帧 ({fps_speed:.1f} fps)")
    
    out.release()
    
    # ---- 6. 合并音视频 ----
    emit_progress(92, "合并音视频...")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    ffmpeg_cmd = os.environ.get('FFMPEG_PATH', 'ffmpeg')
    merge_cmd = [
        ffmpeg_cmd, '-y',
        '-i', audio_path,
        '-i', temp_video,
        '-strict', '-2',
        '-q:v', '1',
        output_path
    ]
    subprocess.call(merge_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 清理临时文件
    try:
        os.remove(temp_video)
        wav_temp = os.path.join(temp_dir, 'temp_audio.wav')
        if os.path.exists(wav_temp):
            os.remove(wav_temp)
    except:
        pass
    
    elapsed_total = time.time() - start_time
    
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        emit_progress(100, "完成!")
        emit_result({
            "output_path": output_path,
            "frames": processed_frames,
            "duration": round(processed_frames / fps, 2),
            "processing_time": round(elapsed_total, 2),
            "file_size_mb": round(file_size, 2),
            "device": device
        })
    else:
        emit_error("输出文件生成失败，请检查 ffmpeg 是否可用")


def check_environment():
    """检查环境并返回状态"""
    status = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "device": get_device(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    
    # 检查 Wav2Lip 模型文件
    model_path = os.path.join(SCRIPT_DIR, 'checkpoints', 'wav2lip_gan.pth')
    status["model_exists"] = os.path.exists(model_path)
    if status["model_exists"]:
        status["model_size_mb"] = round(os.path.getsize(model_path) / (1024 * 1024), 1)
    
    # 检查人脸检测模型（本地缓存）
    face_models = {
        's3fd': any(f.startswith('s3fd') for f in os.listdir(TORCH_HUB_CACHE)) if os.path.exists(TORCH_HUB_CACHE) else False,
        '2DFAN4': any(f.startswith('2DFAN4') for f in os.listdir(TORCH_HUB_CACHE)) if os.path.exists(TORCH_HUB_CACHE) else False,
    }
    status["face_det_models"] = face_models
    status["face_det_cached"] = all(face_models.values())
    
    # 检查依赖
    deps = {}
    for mod in ['cv2', 'librosa', 'scipy', 'numpy', 'face_detection']:
        try:
            __import__(mod)
            deps[mod] = True
        except ImportError:
            deps[mod] = False
    status["dependencies"] = deps
    
    emit_result(status)


def download_models():
    """预下载所有必要的模型"""
    emit_progress(0, "开始下载人脸检测模型...")
    try:
        fa = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            device='cpu'
        )
        del fa
        emit_progress(100, "模型下载完成!")
        # 复制到本地缓存
        system_cache = os.path.expanduser('~/.cache/torch/hub/checkpoints')
        if os.path.exists(system_cache):
            for fname in os.listdir(system_cache):
                src = os.path.join(system_cache, fname)
                dst = os.path.join(LOCAL_CACHE_DIR, fname)
                if os.path.isfile(src) and not os.path.exists(dst):
                    import shutil
                    shutil.copy2(src, dst)
                    emit_progress(100, f"缓存模型: {fname}")
        emit_result({"success": True, "message": "所有模型已下载并缓存"})
    except Exception as e:
        emit_error(f"模型下载失败: {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wav2Lip Inference API')
    subparsers = parser.add_subparsers(dest='command')
    
    # check 命令
    subparsers.add_parser('check', help='检查环境')
    
    # download 命令 - 预下载模型
    subparsers.add_parser('download', help='预下载所有模型')
    
    # run 命令
    run_parser = subparsers.add_parser('run', help='运行推理')
    run_parser.add_argument('--face', required=True, help='视频/图片路径')
    run_parser.add_argument('--audio', required=True, help='音频路径')
    run_parser.add_argument('--output', default='output.mp4', help='输出路径')
    run_parser.add_argument('--checkpoint', default='', help='模型路径')
    run_parser.add_argument('--fps', type=float, default=25.0, help='静态图片的帧率')
    run_parser.add_argument('--resize_factor', type=int, default=1, help='缩放因子')
    run_parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0])
    run_parser.add_argument('--face_det_batch_size', type=int, default=8)
    run_parser.add_argument('--wav2lip_batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    if args.command == 'check':
        check_environment()
    elif args.command == 'download':
        download_models()
    elif args.command == 'run':
        try:
            run_inference(args)
        except Exception as e:
            import traceback
            emit_error(f"{str(e)}\n{traceback.format_exc()}")
    else:
        parser.print_help()
