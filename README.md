<div align="center">
  <h1>Origin-NMS</h1>
</div>
<div align="center">
  <h1>개요</h1>
  <a>더 좋은 객체 탐지를 위해 이미지를 분할하여 detect를 하는 sahi를 이용한다. <br>
    이 방법을 사용하면 겹치는 box가 생기기 때문에 Sahi에서 자체적으로 내장되어 있는 detect box merge 알고리즘인<br>
    NMS, NMM, GREEDYNMM을 사용하지만 아래 사진과 같이 겹치는 box가 여전히 남아있거나, 오리지널 이미지에서 detect한 올바른 box까지 삭제하는 아쉬운 모습을 보이기 때문에 그 부분을 개선한 알고리즘 개발과정이다. <br><h4>NMS방식을 변형하여 오리지널 이미지 box score우선하여 IOS연산 후 삭제된 오리지널 데이터를 복구하는 것을 Origin-NMS라 하겠다.</h4>
  </a>
  
  <p float="left">
    <div align = "left">
      <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/MAIN.png", width="30%">
      <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/onms_FLOWCHAART.png", width="50%",height="50%">
    </div>
  </p>

<p float="left">
    <div align = "center">
      <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/NMS.png", width="60%",height="80%"><br>
      <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/Origin_NMS.png", width="90%", height="80%" ><br>
    </div>
  </p>
</div>

<div><h2>진행과정</h2></div>
  
<div align="center">
  <p float="left">
    <div align = "center">
      <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/Original_yolo5.png", width="45%">
      <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/NMS_yolov5m.png", width="45%"><br>
      [좌 : 오리지널 이미지 detect, 우 : sahi를 이용한 이미지 detect 후 nms]
    </div>
  </p>
  
  
  
  <p float="left">
    <div align = "center">
      위 사진을 보면 내장된 NMS사용시 box가 많아 사라진 것을 볼 수 있다.<br>
      이 때 오리지널 이미지의 detect box는 신뢰도가 높기 때문에 삭제되면 안된다고 판단하여 삭제된 오리지널 box들을 복구했다.<br>
      복구하면서 겹치는 box들이 생긴것을 확인할 수 있다.
      <br>
       <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/NMS_yolov5m.png", width="45%">
       <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/SaveOriginal_yolo5.png", width="45%"><br>
      [좌 : sahi를 이용한 이미지 detct 후 NMS, 우 : 좌측 이미지에서 Original detect box 복구]
    </div>
  </p>
  
  
  
  <p float="left">
    <div align = "center">
      겹치는 박스를 삭제하는 방법으로 NMS를 진행할 때 할 때 오리지널 데이터가 중점이 되어야 한다고 판단했고, <br>
      기존 nms 방법인 전체 box의 score를 sort하고 최대 score부터 NMS를 진행하는 방식을 변형하여<br>
      오리지널 이미지의 box의 score를 우선적으로 NMS를 진행하였다.<br>
       <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/OriginalF_nms_yolo5.png", width="45%">
       <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/OriginalF_save_yolo5.png", width="45%"><br>
      [좌 : sahi를 이용한 이미지 detct 후 original box score 우선 nms, 우 : 좌측 이미지에서 Original detect box 복구]
    </div>
  </p>
     <p float="left">
    <div align = "center">
      다음으로는 기존 방식인 슬라이싱 이미지의 box와 오리지날 이미지의 box의 데이터를 합쳐서 NMS를 진행하는 방법을 변형하여<br>
      슬라이싱 이미지의 box끼리 기존 NMS를 진행하고, 나머지 box와 오리지날 이미지의 box 데이터를 합쳐서 NMS를 진행후 삭제된 오리지널 이미지의 box는 복구하는 방법으로 진행했다.<br>
       <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/NMS_yolov5m.png"><br>
     [ sahi를 이용한 이미지 detect 후 NMS ]<br><br>
       <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/Origin_NMS_yolov5m.png"><br>
     [ 개선한 방법 ]
    </div>
  </p>
  기존 NMS와 비교하면 상당히 개선된 것을 확인할 수 있다.
</div>
  
  
   
[Result Screenshots - yolox](https://github.com/KangHongJun/Origin-NMS/tree/main/sahi_yolox)
    
## USE
setting [sahi](https://github.com/obss/sahi) & [sahi-yolox](https://github.com/Resham-Sundar/sahi-yolox)   <br>
change sahi/predict.py & sahi/postprocess/combine.py

- Origin-NMS(IOS)

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
