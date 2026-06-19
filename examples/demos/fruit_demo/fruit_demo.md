# Fruit demo

The fruit detection demo running 3 parallel networks: yolov8lpose, yolov8sseg, and yolov8s. To run this demo, you will only need a camera and some (fake) fruit. For the setup we used a logitech brio 4k usb camera. 

This demo runs yolov8s object detection, filtered for fruit, on a downscaled 640x640 image. This detection only works when the fruits are up close due to limited resolution of the 640x640 network. To solve the issue, we run a downscaled 640x640 pose detection, cascaded with a segmentation model that is also filtered on fruit. The pose detection gives us the region of interest, which is cropped from the original 4k resolution and fed into the segmentation model. 

For a longer explanation, please view https://www.youtube.com/watch?v=JL3ULSC-f6Q&ab_channel=AxeleraAI

You may run this demo using the following command from your voyager-sdk home: `./examples/demos/fruit_demo.py usb:0 --window-size=fullscreen`

If your end_to_end fps drops when many people are in the screen, this is caused by the duplication of RoIs for the cascaded segmentation model. Each RoI generates images that the segmentation model needs to process, causing a multiplication of input images for this model. This can be changed by using the `top_k` for the input part of the segmentations entry of the [yaml file](/ax_models/reference/cascade/fruit-demo.yaml)
