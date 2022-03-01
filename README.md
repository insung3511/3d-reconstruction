# 이-것은 바로....
비전 처리 공부를 계속 해나가기 위해서 이렇게 공부를 하고 있다. 현재 Stereo Camera를 직접 제작해서 사용하기에 정확도는 보장하기가 어렵다. 기존에 사용하던 로지텍 USB 웹캠을 활용하여 이를 연결하여 활용하고 있다. 동시에 사진을 찍고 이를 다시 3D Reconstrucition 으로 변환하는 과정으로 통해 ply 파일로 출력을 한다.

## 정확성과 화질
각 PC의 사양에 따라서 이를 downsample을 얼마나 해야하나에 따라서 달라지게 된다. 현재 개발자 (나, 박인성 본인ㄴ임 ㅇㅇ )이 사용하는 macbook pro (m1) 모델에서는 큰 downsample을 하지 않고 큰 문제가 되지 않는다. 현재 가장 많이 활용하고 있는 코드는 아래 경로에 보면 있다.
```
./3d-reconstruction/for_3d/Reconstruction/disparity_image.py
```
이후에 더 많은 내용을 작성할 예정...

## Datasets
- Driving Stereo Datasets from <a href="https://drivingstereo-dataset.github.io">https://drivingstereo-dataset.github.io</a>