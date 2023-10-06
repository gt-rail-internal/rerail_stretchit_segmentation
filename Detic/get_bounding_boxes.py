from demo_rerail import detect

#  Get bounding boxes and classes for all objects in image
input = 'test.png'
output = 'out2.png'

bounding_boxes, classes = detect(input,output)

print(bounding_boxes, classes)