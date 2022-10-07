<div align="center">
  <h1>Origin-NMS</h1>
  <h4>merge overlapping detect box & save true correct box when use sahi<h4>
</div>
<div align="center">
  <h1>Overview</h1>
  <a>sahi 사용시 원본 이미지의 detect box와 슬라이싱된 이미지의 detect box를 nms처리를 하지만<br>
    오탐박스가 남거나(overlapping) 올바른 detect box까지 삭제되는 경우가 생기기 때문에 그 부분을 개선했다.</a>
</div>

# Result Screenshots - IOS
<div align="center">
  <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/Original.jpg", width="50%",style="display:block;"> <br>
  <a>Original</a>
  <div width="40%", float = "left">
    <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/NMS_yolov5m.png", width="40%"><br>
    <a>NMS(yolov5m)</a> 
  </div>
  <div width="40%",float = "right">
    <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/Origin_NMS_yolov5m.png", width="40%"><br>
    <a textstyle="bold">Origin-NMS(yolov5m)</a>
  </div>
</div>

# USE
base sahi/sahi-yolox
change sahi/predict.py & sahi/postprocess/combine.py


# Reference
<ul>
  <li>https://github.com/obss/sahi
  <li>https://github.com/Resham-Sundar/sahi-yolox
</ul>
    

# 과정(2022.9.27 ~ 2022.10.07)
1. merge detect box & NMS개선 논문 탐색
2. sahi 내부에 있는 merge알고리즘 사용하여 결과 분석, iou,ios에 대한 이해
3. nms, ios기반으로 방향성 잡음
4. nms진행시 오리지널 이미지의 box score우선하여 ios계산하는 방식 진행
5. 위의 단계에서 ios계산 후 오리지널 이미지의 box데이터 살리는 방식 진행 - origin-nms
6. 기존에는 슬라이싱 이미지의 detect box와 오리지널 이미지의 detect box데이터를 합쳐서 nms처리하지만
   먼저 슬라이싱 이미지의 detect box 데이터만 nms처리 후 소거된 슬라이싱이미지의 detect box 데이터와 오리지널 데이터를 합쳐서 origin-nms진행







 
