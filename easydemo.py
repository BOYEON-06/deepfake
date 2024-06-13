# 메인 소스 코드 
# 실행 : ~/Thin-plate-Spline-Motion-Model -> python3 easydemo.py;

import torch
import imageio
import imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from skimage import img_as_ubyte
import warnings
import face_alignment
import ssl
import certifi
import urllib.request

from demo import load_checkpoints
from demo import make_animation
from demo import find_best_frame as _find

# 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_name = 'vox'  # ['vox', 'taichi', 'ted', 'mgif']
config_path = 'config/vox-256.yaml'
checkpoint_path = 'checkpoints/vox.pth.tar'
predict_mode = 'relative'  # ['standard', 'relative', 'avd']
find_best_frame = True

# 해상도 설정
pixel = 256
if dataset_name == 'ted':
    pixel = 384

warnings.filterwarnings("ignore")

print("체크포인트 로드 중...")
inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
print("체크포인트 로드 완료")


# 전처리
source_image_path = 'assets/source.png'  # 딥페이크 대상 인물 이미지
driving_video_path = 'assets/driving.mp4' # 대상에 씌울 영상 소스

print("이미지 및 비디오 로드 중...")
source_image = imageio.imread(source_image_path)
reader = imageio.get_reader(driving_video_path)

# 비디오 리사이즈
source_image = resize(source_image, (pixel, pixel))[..., :3]

# 프레임 단위로 분류
fps = reader.get_meta_data()['fps']
driving_video = []
try:
    for im in reader:
        driving_video.append(im)
except RuntimeError:
    pass
reader.close()

driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in driving_video]

def create_animation(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 4))
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani

# 딥페이크된 영상 최종 저장 경로 
output_video_path = 'assets/result.mp4'

if predict_mode == 'relative' and find_best_frame:
    print("가장 잘 맞는 프레임 찾기...")
    i = _find(source_image, driving_video, device == 'cpu')

    print("Best frame: " + str(i))
    driving_forward = driving_video[i:]
    driving_backward = driving_video[:(i+1)][::-1]
    print("애니메이션 생성 중 (forward)...")
    predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network, avd_network, device=device, mode=predict_mode)
    print("애니메이션 생성 중 (backward)...")
    predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector, dense_motion_network, avd_network, device=device, mode=predict_mode)
    predictions = predictions_backward[::-1] + predictions_forward[1:]
else:
    print("애니메이션 생성 중...")
    predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device=device, mode=predict_mode)

# 결과 비디오 저장
print("결과 비디오 저장 중...")
imageio.mimsave(output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

print(f"애니메이션이 '{output_video_path}'로 저장되었습니다. 다음 경로에서 영상을 확인하세요.")
