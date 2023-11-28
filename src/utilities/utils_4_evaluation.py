def calculate_iou(box1, box2):
    box2 = list(map(int, box2))
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = box1_area + box2_area - intersection
    iou = intersection / union
    return iou

def calculate_precision_recall(predictions, ground_truth, confidence_threshold, iou_threshold):
    true_positives = 0
    false_positives = 0
    total_objects = len(ground_truth)
    for _, confidence, pred_box in predictions:
        if confidence < confidence_threshold:
            continue
        iou_scores = [calculate_iou(pred_box, gt_box) for _, _, gt_box in ground_truth]
        max_iou = max(iou_scores, default=0)
        if max_iou >= iou_threshold:
            true_positives += 1
        else:
            false_positives += 1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / total_objects if total_objects > 0 else 0
    return precision, recall

def calculate_average_precision(predictions, ground_truth, confidence_threshold=0.5, iou_threshold=0.5):
    predictions = sorted(predictions, key=lambda x: x[0], reverse=True)
    precision_list = []
    recall_list = []
    for i in range(len(predictions)):
        precision, recall = calculate_precision_recall(predictions[:i+1], ground_truth, confidence_threshold, iou_threshold)
        precision_list.append(precision)
        recall_list.append(recall)
    ap = 0
    for i in range(1, len(precision_list)):
        ap += (recall_list[i] - recall_list[i-1]) * precision_list[i]
    return ap

def calculate_average_precision_over_images(predictions_list, ground_truth_list, confidence_threshold=0.5, iou_threshold=0.5):
    total_ap = 0
    for predictions, ground_truth in zip(predictions_list, ground_truth_list):
        ap = calculate_average_precision(predictions, ground_truth, confidence_threshold, iou_threshold)
        total_ap += ap
    avg_precision = total_ap / len(predictions_list)
    return avg_precision
