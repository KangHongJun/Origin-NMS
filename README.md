<div align="center">
  <h1>Origin-NMS</h1>
</div>
<div align="center">
  <h1>Overview</h1>
  <a>더 좋은 객체 탐지를 위해 이미지를 분할하여 detect를 하는 sahi를 이용했고, Sahi가 기본적으로 제공하는 detect box merge알고리즘인 NMS, NMM, GREEDYNMM은 다음과 같이 오탐박스가 남거나, 오리지널 이미지에서 detect한 올바른 박스까지 삭제하는 아쉬운 모습을 보이기 때문에 그 부분을 개선해가는 과정에 대해 작성해봤다.
</a>
</div>

<div><h1>Result Screenshots - YOLOv5m</h1></div>
  
<div align="center">
  <p float="left">
    <div align = "left", weight="40%">
      <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/Original.jpg", width="40%">
    
    </div>
    <div align = "right", weight="40%">
      <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/Original.jpg", width="40%">
      
    </div>
  </p>
  
  
  
  <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/Original.jpg", width="50%",style="display:block;"> <br>
  <h4>Original</h4>
  <div width="40%", float = "left">
    <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/NMS_yolov5m.png", style="display:inline-block;",  width="40%"><br>
    <h4>NMS(YOLOv5m)</h4> 
  </div>
  <div width="40%",float = "left">
    <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/Origin_NMS_yolov5m.png", style="display:inline-block;", width="40%"><br>
    <h4>Origin-NMS(YOLOv5m)</h4>
  </div>
</div>
   
[Result Screenshots - yolox](https://github.com/KangHongJun/Origin-NMS/tree/main/sahi_yolox)
    
# USE
setting sahi/sahi-yolox<br>
change sahi/predict.py & sahi/postprocess/combine.py

    
<details closed>
<summary>
<big><b>코드</b></big>
</summary>

- Origin-nms(IOS)

```python
def origin_nms(
    predictions: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
    len_original:int=0,#오리지널 이미지 detect 박스 매개변수 추가
):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        predictions: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        A list of filtered indexes, Shape: [ ,]
    """

    # we extract coordinates for every
    # prediction box present in P

    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]

    # we extract the confidence scores as well
 
    #슬라이스 이미지/오리지널 이미지의 detect box각각 argsort
    slice_value = len(predictions)-len_original
    scores = predictions[:slice_value, 4]
    #scores = predictions[:len_original, 4]
    original_scores = predictions[slice_value:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P
    # according to their confidence scores

    order = scores.argsort() #sort index num
    original_order = original_scores.argsort()

    #idx값 조정 - 오리지널 이미지 detect box의 score 우선하여 origin-nms를 진행하기 때문에 뒤로 붙인다.
    original_order = original_order+len(scores)
    order = torch.cat([order, original_order])

    # initialise an empty list for
    # filtered prediction boxes
    keep = []

    while len(order) > 0:
        # extract the index of the
        # prediction with highest score
        # we call this prediction S

        idx = order[-1]

        # push S in filtered predictions list
        keep.append(idx.tolist())

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection (교차) area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        if match_metric == "IOU":
            # find the union of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[idx]
            # find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # find the smaller area of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            smaller = torch.min(rem_areas, areas[idx])
            #print("IOS--areas[idx] ", areas[idx])
            # find the IoU of every prediction in P with S
            match_metric_value = inter / smaller
            #print("IOS--inter ", inter)
            #print("IOS--smaller", smaller)
        else:
            raise ValueError()

        # keep the boxes with IoU less than thresh_iou
        mask = match_metric_value < match_threshold
        order = order[mask] #boolean indexing

    #오리지널 데이터가 삭제됐다면 다시 추가
    for i in original_order.tolist():
        if i not in keep:
            keep.append(i)
    return keep
```
  
</details>


    
# 과정 <h4>(2022.9.27 ~ 2022.10.07)</h4>
1. merge detect box & NMS개선 논문 탐색
2. sahi 내부에 있는 merge알고리즘(NMS, NMM, GREEDYNMM) 사용하여 결과 분석, sahi 소스코드 분석, iou,ios에 대한 이해
3. nms, ios기반으로 방향성 잡음
4. 대체로 정확도가 높은 오리지널 이미지 detect box도 삭제되어 nms 후 오리지널 박스를 살리는 밯향으로 진행
5. 오탐박스를 줄이기 위해 nms진행시 오리지널 이미지의 box score우선하여 ios계산하는 방식 진행
6. 위의 단계에서 박스가 너무 많이 삭제되어 위의 방법을 토대로 오리지널 삭제된 이미지의 box데이터 살리는 방식 진행(origin-nms)
7. 기존에는 슬라이싱 이미지의 detect box와 오리지널 이미지의 detect box데이터를 합쳐서 nms처리하지만
   먼저 슬라이싱 이미지의 detect box 데이터만 nms처리 후 소거된 슬라이싱이미지의 detect box 데이터와 오리지널 데이터를 합쳐서 origin-nms진행함
  



<details closed>
<summary>
<big><b>진행과정 이미지</b></big>
</summary>

</details>




 
