# Example on how to call the detic model within another code

from demo_rerail import detect

#  Get bounding boxes and classes for all objects in image
input = 'test.png'
output = 'out2.png'

# 'masks_dets' is the binary mask if each of the predicted objects. It has three dimensions; the first represents the no of images, 
# the second and third are the columns and rows or vice versa

boxes,confidences,classes,masks_dets = detect(input,output)

# instances = masks_dets.shape[0]
# rows = masks_dets.shape[1]
# cols = masks_dets.shape[2]

# for k in range(instances):
#     found = 0
#     for i in range(rows):
#         for j in range(cols):
#             if masks_dets[k][i][j] == 1 and found == 0:
#                 print(k,classes[k],boxes[k],confidences[k],j,i)
#                 found = 1
#                 break