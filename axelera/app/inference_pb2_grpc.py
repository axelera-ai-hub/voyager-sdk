# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import axelera.app.inference_pb2 as inference__pb2


class InferenceStub(object):
    """The Inference service definition.
    Sends an request to perform inference
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Infer = channel.unary_stream(
            '/InferenceServer.Inference/Infer',
            request_serializer=inference__pb2.InferenceRequest.SerializeToString,
            response_deserializer=inference__pb2.Inferenceresult.FromString,
        )
        self.StreamInit = channel.unary_unary(
            '/InferenceServer.Inference/StreamInit',
            request_serializer=inference__pb2.InitInferenceRequest.SerializeToString,
            response_deserializer=inference__pb2.InitInferenceResponse.FromString,
        )
        self.StreamInfer = channel.stream_stream(
            '/InferenceServer.Inference/StreamInfer',
            request_serializer=inference__pb2.StreamInferenceRequest.SerializeToString,
            response_deserializer=inference__pb2.Inferenceresult.FromString,
        )


class InferenceServicer(object):
    """The Inference service definition.
    Sends an request to perform inference
    """

    def Infer(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StreamInit(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StreamInfer(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InferenceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'Infer': grpc.unary_stream_rpc_method_handler(
            servicer.Infer,
            request_deserializer=inference__pb2.InferenceRequest.FromString,
            response_serializer=inference__pb2.Inferenceresult.SerializeToString,
        ),
        'StreamInit': grpc.unary_unary_rpc_method_handler(
            servicer.StreamInit,
            request_deserializer=inference__pb2.InitInferenceRequest.FromString,
            response_serializer=inference__pb2.InitInferenceResponse.SerializeToString,
        ),
        'StreamInfer': grpc.stream_stream_rpc_method_handler(
            servicer.StreamInfer,
            request_deserializer=inference__pb2.StreamInferenceRequest.FromString,
            response_serializer=inference__pb2.Inferenceresult.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'InferenceServer.Inference', rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class Inference(object):
    """The Inference service definition.
    Sends an request to perform inference
    """

    @staticmethod
    def Infer(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_stream(
            request,
            target,
            '/InferenceServer.Inference/Infer',
            inference__pb2.InferenceRequest.SerializeToString,
            inference__pb2.Inferenceresult.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def StreamInit(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/InferenceServer.Inference/StreamInit',
            inference__pb2.InitInferenceRequest.SerializeToString,
            inference__pb2.InitInferenceResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def StreamInfer(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            '/InferenceServer.Inference/StreamInfer',
            inference__pb2.StreamInferenceRequest.SerializeToString,
            inference__pb2.Inferenceresult.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
