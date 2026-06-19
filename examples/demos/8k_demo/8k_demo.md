# 8k demo

The 8k demo is a real-time inference (using multiple Metis AI acceleration cards), powering an 8K video stream and high-precision object detection. From detecting small objects and smart devices to processing 380 overlapping image tiles per second.

Instead of upscaling the model, this demo uses tiling to cut the image into smaller pieces and processing them separately. That way, we can process the full resolution without losing information.

For a longer explanation, please view https://www.youtube.com/watch?v=Idifa-UNQRM&ab_channel=AxeleraAI

Tiling is model agnostic, meaning that if we can run the model without tiling, we can run it with tiling. However, it is currently only tested and verified using Yolov8 object detection models.

## Using this demo

This is a heavy demo, usualy running on multiple PCIe cards. If you’re using a single card, you may need to either increase the tile size (resulting in less tiles per frame), use a smaller network like `yolov8s-coco-onnx`, or use a smaller input resolution like 4k. Be sure to tune the cameras FPS to the number of frames you’re processing:
- If the camera does not feed enough frames, our e2e fps drops because there’s not enough available frames
- If the camera feeds too many frames, the chip cannot keep up and you will see delays between the camera and the screen

To tune this demo, follow these steps
- Play with tilesize: bigger tiles result in less tiles per frame. This means higher fps as there's less processing per frame. However, you may lose a bit of detail when doing so. When we run this demo at shows, we usually use tiles of size 960 or 1280 when using multiple Metis PCIe cards.
- Run at highest fps: once you've found your desired tile size, run once at the maximum camera FPS. This will likely result in high camera-to-screen delay as the camera feeds more frames than your runtime can handle. After a few minutes, look at the end_to_end FPS.
- Set camera to 1 fps lower than the stable fps that you found in the previous step. After a few minutes of "warmup", the end_to_end fps will stabilize at the camera FPS and your latency will be small. The final latency depends on your number of tiles, your camera latency, and the processing latency of your model. The reason the delay lowers in the beginning of the run, is because the camera starts sending frames before inference starts, meaning the buffers fill up. The system needs some time to work through these frames and empty the buffers before you start seeing real time results.

For conferences, we generally use the following command to run this demo:\
`./examples/demos/8k_demo.py ${video_source} --tiled 960 --window fullscreen --show-tiles --no-show-host-fps --no-show-cpu-usage`. \
Your camera's user manual shows how to connect to the rtsp stream of your camera. For usb cameras, simply use the `usb:x` source, where `x` is your camera number. Refer to our [video_sources](../../../docs/tutorials/video-sources.md) tutorial for more information.

The `--show-tiles` option shows the tiles on the screen. While the screen may be a rectangular shape, the tiles follow the shape of the model (square for square models, 16/9 for 16/9 models, etc.). Therefore, the tiles may overlap somewhat in either horizontal or vertical (or both) direction. You can force this behavior by using the `--tile-overlap x` flag, where `x` specifies the minimum amount of overlap as a percentage.

## Turning camera FPS
If you are using a usb camera, you can do tune the fps by for example usb:0@15 to set the camera fps to 15 for a usb camera. Make sure that the camera supports your desired FPS.

In RTSP, you can find the setting in the user manual of your camera.

## Changing this demo
If you want to make changes to this demo, or use it in a different way, please read through the following.

If you are using tiles smaller than the model size, our SDK automatically picks the tile size using a `max(tilesize, modelsize)` check. This is to avoid unneccesary use of resources, as there is no reason to use tiles smaller than the model size.
