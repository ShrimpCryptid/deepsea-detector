""" Code adapted from the tryolabs/norfair project
(https://github.com/tryolabs/norfair/blob/master/demos/yolov5/yolov5demo.py)
"""

import argparse
from typing import Dict, Union, List, Optional, Set
import csv

import cv2
import norfair
import numpy as np
import torch
import yolov5

max_distance_between_points: int = 30
PREVIEW_WINDOW_NAME = "image"
classifications: List[str] = ['annelida', 'arthropoda', 'cnidaria', 'echinodermata', 'fish',
                              'mollusca', 'other-invertebrates', 'porifera', 'unidentified-biology']


class YOLO:
    "Wrapper class for loading and running YOLO model"
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # load model
        self.model = yolov5.load(model_path, device=device)

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None
    ) -> torch.Tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)  # pylint: disable=not-callable
        return detections


def euclidean_distance(detection, tracked_object):
    """Returns euclidean distance between two points."""
    return np.linalg.norm(detection.points - tracked_object.estimate)


def center(points):
    """Returns the center coordinates of a series of points."""
    return [np.mean(np.array(points), axis=0)]


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.Tensor,
    track_points: str = 'bbox'  # bbox or centroid
) -> List[norfair.Detection]:
    """Converts detections_as_xywh to norfair detections.
    Label is the classification of the detection.
    """
    norfair_detections: List[norfair.Detection] = []

    if track_points == 'centroid':
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [
                    detection_as_xywh[0].item(),
                    detection_as_xywh[1].item()
                ]
            )
            scores = np.array([detection_as_xywh[4].item()])
            label = classifications[int(detection_as_xywh[5].item())]
            norfair_detections.append(
                norfair.Detection(points=centroid, scores=scores, label=label)
            )
    elif track_points == 'bbox':
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()]
                ]
            )
            label = classifications[int(detection_as_xyxy[5].item())]
            scores = np.array([detection_as_xyxy[4].item(),
                              detection_as_xyxy[4].item()])
            norfair_detections.append(
                norfair.Detection(points=bbox, scores=scores, label=label)
            )

    return norfair_detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument("files", type=str, nargs="+",
                        help="Video files to process")
    parser.add_argument("--detector_path", type=str,
                        default="yolov5m6.pt", help="YOLOv5 model path")
    parser.add_argument("--img_size", type=int, default="720",
                        help="YOLOv5 inference size (pixels)")
    parser.add_argument("--conf_thres", type=float, default="0.25",
                        help="YOLOv5 object confidence threshold")
    parser.add_argument("--iou_thresh", type=float,
                        default="0.45", help="YOLOv5 IOU threshold for NMS")
    parser.add_argument('--classes', nargs='+', type=int,
                        help='Filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument("--device", type=str, default=None,
                        help="Inference device: 'cpu' or 'cuda'")
    parser.add_argument("--track_points", type=str, default="centroid",
                        help="Track points: 'centroid' or 'bbox'")
    parser.add_argument("--period", type=int, default=1,
                        help="The period (in frames) with which the detector should be run.")
    # TODO: Handle multiple input videos without overwriting the same file
    parser.add_argument("--output_csv", type=str, default="out.csv",
                        help="Output path for the tracking data CSV file.")
    parser.add_argument("--output_video", type=str, default="out.mp4",
                        help="Output path for the MP4 video file, showing annotations.")
    parser.add_argument("--show_preview", action="store_true",
                        help="Show a preview of detections in real time. Do not use in notebooks.")

    args = parser.parse_args()

    model = YOLO(args.detector_path, device=args.device)

    for input_path in args.files:
        video = norfair.Video(input_path=input_path,
                              output_path=args.output_video)
        tracker = norfair.Tracker(
            distance_function=euclidean_distance,
            distance_threshold=max_distance_between_points,
        )
        paths_drawer = norfair.Paths(center, attenuation=0.01)

        # Data objects for storing currently detected objects.
        # 0: first frame of appearance
        # 1: last frame of appearance
        # 2: classification label index
        track_id_to_data: Dict[int, List[Union[str, int]]] = {}
        currently_tracked_ids: Set[int] = set()

        for i, frame in enumerate(video):

            if i % args.period == 0:
                yolo_detections = model(
                    frame,
                    conf_threshold=args.conf_thres,
                    iou_threshold=args.iou_thresh,
                    image_size=args.img_size,
                    classes=args.classes
                )
                norfair_detections = yolo_detections_to_norfair_detections(
                    yolo_detections, track_points=args.track_points)
                tracked_objects = tracker.update(
                    detections=norfair_detections, period=args.period)
                if args.track_points == 'centroid':
                    norfair.draw_points(frame, norfair_detections)
                elif args.track_points == 'bbox':
                    norfair.draw_boxes(frame, norfair_detections)
            else:
                tracked_objects = tracker.update()

            # Loop through tracked objects and save their label ("initialized id") and unique id.
            ids_seen_this_frame: Set[int] = set()
            for obj in tracked_objects:
                if obj.id not in track_id_to_data:
                    # First time this object has been identified, so we save it.
                    track_id_to_data[obj.id] = [i, -1, obj.label]
                    currently_tracked_ids.add(obj.id)
                ids_seen_this_frame.add(obj.id)

            # Check if any ids were no longer tracked this frame, and update their entries.
            for track_id in set(currently_tracked_ids):
                if track_id not in ids_seen_this_frame:  # ID disappeared, record last frame.
                    track_id_to_data[track_id][1] = i - 1
                    currently_tracked_ids.remove(track_id)

            norfair.draw_tracked_boxes(frame, tracked_objects, color_by_label=True,
                                       draw_labels=True, border_width=1, label_size=1.0)
            frame = paths_drawer.draw(frame, tracked_objects)
            video.write(frame)

            if args.show_preview:
                cv2.imshow(PREVIEW_WINDOW_NAME, frame)

    print(track_id_to_data)
    # Finished traversing frames, so we write out CSV tracking data.
    with open(args.output_csv, 'w', newline='') as csv_file:
        fieldnames = ['classification', 'first_frame', 'last_frame']
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        for track_data in track_id_to_data.values():
            # Write out each tracked object as its own row.
            # classification, starting frame, ending frame
            writer.writerow({
                'classification': track_data[2],
                'first_frame': track_data[0],
                'last_frame': track_data[1]
            })

    # Close the preview window
    if args.show_preview:
        cv2.destroyWindow(PREVIEW_WINDOW_NAME)
