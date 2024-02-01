import cv2


bikes_classifier = cv2.CascadeClassifier('/Users/joaquimperes/Desktop/bike-detection/two_wheeler.xml')

camera = cv2.VideoCapture('/Users/joaquimperes/Desktop/bike-detection/bikes.mp4')
count = 0

while(True):
  ret, img = camera.read()
  height, whidth = img.shape[0:2]
  img[0:70, 0:whidth] = [255,0,0]
  cv2.putText(img, 'Quantidade encontrada:', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
              (0,0,0), 2)
  cv2.line(img,(0,height-200), (whidth, height-200), (0,255,255), 2)
  


  blur = cv2.blur(img, (3,3))
  gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
  bikes = bikes_classifier.detectMultiScale(gray)
  

  for (x,y,w,h) in bikes:
    bikeCenter = int(y+h/2)
    lineCenter = height-200
    if(bikeCenter<lineCenter+6 and bikeCenter>lineCenter-6):
      count = count + 1
      cv2.line(img,(0,height-200), (whidth, height-200), (0,255,255), 2)


    cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),2)
    cv2.putText(img, 'Bike', (x,y -10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0,255,0), 2)
    cv2.putText(img,str(count),(600,50),cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255,255,255), 2)
    
    
    


  cv2.imshow('teste', img)
  key = cv2.waitKey(1)
  if(key == 27):
    break
cv2.destroyAllWindows
camera.release()
