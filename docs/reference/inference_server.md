![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Inference Server

- [Inference Server](#inference-server)
  - [Example Inference Client](#example-inference-client)
  - [Creating a New Client](#creating-a-new-client)
    - [Connectting to the Server](#connectting-to-the-server)
  - [Links](#links)

Axelera's inference server enables the use of Axelera's hardware for running inference on remote
hardware. The server runs the specified network on images provided by the remote client and returns
the results back to the client. The server can switch between networks if needed.

All communication between the client and server is handled via [GRPC](https://grpc.io) and the data
is transferred via [protocol buffers](https://protobuf.dev). The APIs are kept deliberately simple
so that users can easily produce their own clients that match their own use cases.

The server communicates with the client via a socket and connects on the port specified as part of
the launch command. If no port is specified then the default of 5000 is used. To launch the server,
simply run:

```bash
./inference_server.py <network-name> --port 5000
```

Where `<network-name>` is the name of the neural network to start serving.

## Example Inference Client

The sample client takes as input a video source (or a single image), streams each frame to the
client and upon receipt of the results composes the output on the original frame and displays the
composed image.

To run the sample example client:

```bash
./inference_client.py <hostname>:<port> <network-name> <source>
```

**hostname:port** specifies the hostname or IP address of the device the server was started on and
port specifies the port that the server is listening on.

**network-name** specifies the actual neural network to run. Any network that can be built by
Axelera's Voyager™ SDK is available.

**source** specifies the input source. This can be a single image file, video file or an RTSP
stream.


## Creating a New Client

Using the tools provided by the Axelera Voyager™ SDK, creating a new client is straightforward.
There are 3 main steps involved, connecting to the server, creating the stream and running the
inference loop.

### Connectting to the Server

In order to connect to the server we use GRPC, first creating a communications channel and then a
stub for our remote procedure calls:

```python
import grpc
import axelera.app.inference_pb2_grpc as inference_pb_grpc

    ...
    # server is hostname:port
    with grpc.insecure_channel(server) as channel:
        stub = inference_pb_grpc.InferenceStub(channel)

    ...
```

Once this is done, we can create the remote stream:

```python
from axelera.app.inference_client import remote_stream

    ...
    # input is input source
    # network is the network to run on the server
    stream = remote_stream(stub, input, network)

```
This may fail and raise a *ValueError* if the network is not supported (or is incorrectly named).


Once we have a remote stream we can iterate over the frames and access the results:

```python
def inference(stream, input)
   try:
        for image, meta in stream:
            # The image is an axelera.types.Image
            # The metadata is an axelera.app.meta.AxMeta
            if image is not None:
                ... # Do something with image and/or meta

    except Exception as e:
        # Exceptions may happen if the network is invalid or the connection to the server is lost
        pass
```

Further processing can then be applied to the image and metadata. See the sample inference_client.py
for an example of rendering the results on the display.

## Links
* [inference_server.py](/inference_server.py)
* [inference_client.py](/inference_client.py)
