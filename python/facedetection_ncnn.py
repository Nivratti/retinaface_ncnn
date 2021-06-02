"""
Usage: 
python facedetection_ncnn.py   "path/to/image"
"""
import sys
import cv2
import numpy as np
import ncnn
from ncnn.utils import draw_faceobjects
from retinaface import RetinaFace

class FaceDetectionRTncnn:
    def __init__(self, prob_threshold=0.8, nms_threshold=0.4, num_threads=4, use_gpu=False, model="resnet50"):
        self.net = RetinaFace(
            prob_threshold=prob_threshold, nms_threshold=nms_threshold, num_threads=num_threads, use_gpu=use_gpu, 
            model=model
        )

    def get_face_bbox(self, img_bgr, max_faces=0, return_boxformat="xywh", is_draw_faceobjects=False):
        """
        Get face box

        Args:
            img_bgr (opencv): Opencv image in bgr default format
            max_faces (int, optional): If 0 return all face counts. Defaults to 0.
            return_boxformat (str, optional): If xywh return x,y and width and height. Defaults to "xywh".
                                            else x1y1-x2y2 - x, y , x+ w, x + height
        """
        def _get_box(obj, return_boxformat="xywh"):
            if return_boxformat == "xywh":
                bbox = [
                    round(obj.rect.x), round(obj.rect.y), round(obj.rect.w), round(obj.rect.h), obj.prob # xywh format
                ]
            else:
                # x1y1-x2y2
                bbox = [
                    round(obj.rect.x), round(obj.rect.y), 
                    round(obj.rect.x + obj.rect.w), round(obj.rect.y + obj.rect.h), 
                    obj.prob # x1y1x2y2 facebox format
                ]
            return bbox

        faceobjects = self.net(img_bgr)

        # for obj in faceobjects:
        #     print(
        #         "%.5f at %.2f %.2f %.2f x %.2f"
        #         % (obj.prob, obj.rect.x, obj.rect.y, obj.rect.w, obj.rect.h)
        #     )

        if is_draw_faceobjects:
            draw_faceobjects(img_bgr, faceobjects)
            
        if max_faces <= 0 or max_faces >=2:
            face_boxes = []
            for idx, obj in enumerate(faceobjects, start=1):
                # print(f"obj.rect.x: {obj.rect.x}")
                # print(f"obj.rect.y: {obj.rect.y}")
                # print(f"obj.rect.w: {obj.rect.w}")
                # print(f"obj.rect.h: {obj.rect.h}")

                bbox = _get_box(obj, return_boxformat="xywh")
                face_boxes.append(bbox)
                if idx == max_faces:
                    break
            return face_boxes
        else:
            # return singal high confidence face
            if len(faceobjects) >= 1:
                return _get_box(faceobjects[0], return_boxformat="xywh")
            else:
                return []


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Face detection using retinaface ncnn.')
    parser.add_argument('image', type=str, help='Image path')
    args = parser.parse_args()

    face_detector = FaceDetectionRTncnn(use_gpu=True)
    
    imagepath = args.image

    img_bgr = cv2.imread(imagepath)
    if img_bgr is None:
        print("cv2.imread %s failed\n" % (imagepath))
        sys.exit(0)

    face_boxes = face_detector.get_face_bbox(
        img_bgr, max_faces=2,
        return_boxformat="xywh",
        is_draw_faceobjects=True
    )
    print(f"\nface_boxes: {face_boxes}")

if __name__ == "__main__":
    main()
