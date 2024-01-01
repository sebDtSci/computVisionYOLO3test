import cv2
import numpy as np

# Charger les noms des classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Charger le modèle YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Fonction pour dessiner les boîtes englobantes
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    confi=str(round(confidences[i],2))
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0,255,0), 2)
    cv2.putText(img, label+" " +confi , (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Initialiser la caméra
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    Width, Height = frame.shape[1], frame.shape[0]
    scale = 0.00392

    # Convertir l'image en blob
    blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)

    # Obtenir la détection
    outs = net.forward(get_output_layers(net))

    # Parcourir toutes les détections pour dessiner les boîtes
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0] if isinstance(i, np.ndarray) else i  # Ajustement pour la compatibilité
        box = boxes[int(i)]
        x, y, w, h = box[0], box[1], box[2], box[3]
        draw_bounding_box(frame, class_ids[int(i)], confidences[int(i)], round(x), round(y), round(x+w), round(y+h))


    # Afficher l'image
    cv2.imshow("Object detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
