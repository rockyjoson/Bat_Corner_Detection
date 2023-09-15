from ultralytics import YOLO
import cv2

# using the best developed model
model = YOLO("bat_corner_model.pt")
img=cv2.imread(r'C:\Users\user\OneDrive\Desktop\Bat_Corner_Detection\input_image_kohli.jpg')

#class_names = ['top left', 'top right', 'bottom left', 'bottom right']

results = model(img)
#print(results)
for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 3)

cv2.imshow('Bat Detection', img)
if cv2.waitKey(0) & 0xFF == ord('q'):
  cv2.destroyAllWindows()
