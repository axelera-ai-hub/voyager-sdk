# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: inference.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0finference.proto\x12\x0fInferenceServer\"\'\n\x14InitInferenceRequest\x12\x0f\n\x07network\x18\x01 \x01(\t\"\'\n\x15InitInferenceResponse\x12\x0e\n\x06status\x18\x01 \x01(\t\"?\n\x16StreamInferenceRequest\x12%\n\x05image\x18\x01 \x01(\x0b\x32\x16.InferenceServer.Image\"2\n\x10InferenceRequest\x12\x0f\n\x07network\x18\x01 \x01(\t\x12\r\n\x05input\x18\x02 \x01(\t\"<\n\nObjectMeta\x12\r\n\x05\x62oxes\x18\x01 \x01(\x0c\x12\x0e\n\x06scores\x18\x02 \x01(\x0c\x12\x0f\n\x07\x63lasses\x18\x03 \x01(\x0c\"\x18\n\x06Scores\x12\x0e\n\x06scores\x18\x01 \x03(\x0c\"\x16\n\x05\x42oxes\x12\r\n\x05\x62oxes\x18\x01 \x03(\x0c\"\x1a\n\x07\x43lasses\x12\x0f\n\x07\x63lasses\x18\x01 \x03(\x0c\"\x87\x01\n\nClassifier\x12\'\n\x06scores\x18\x01 \x03(\x0b\x32\x17.InferenceServer.Scores\x12%\n\x05\x62oxes\x18\x02 \x03(\x0b\x32\x16.InferenceServer.Boxes\x12)\n\x07\x63lasses\x18\x03 \x03(\x0b\x32\x18.InferenceServer.Classes\"G\n\x05Image\x12\r\n\x05width\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\x10\n\x08\x63hannels\x18\x03 \x01(\x05\x12\r\n\x05image\x18\x04 \x01(\x0c\"\xc3\x01\n\x0fInferenceresult\x12-\n\x03obj\x18\x01 \x01(\x0b\x32\x1b.InferenceServer.ObjectMetaH\x00\x88\x01\x01\x12\x34\n\nclassifier\x18\x02 \x01(\x0b\x32\x1b.InferenceServer.ClassifierH\x01\x88\x01\x01\x12*\n\x05image\x18\x03 \x01(\x0b\x32\x16.InferenceServer.ImageH\x02\x88\x01\x01\x42\x06\n\x04_objB\r\n\x0b_classifierB\x08\n\x06_image2\x9c\x02\n\tInference\x12P\n\x05Infer\x12!.InferenceServer.InferenceRequest\x1a .InferenceServer.Inferenceresult\"\x00\x30\x01\x12]\n\nStreamInit\x12%.InferenceServer.InitInferenceRequest\x1a&.InferenceServer.InitInferenceResponse\"\x00\x12^\n\x0bStreamInfer\x12\'.InferenceServer.StreamInferenceRequest\x1a .InferenceServer.Inferenceresult\"\x00(\x01\x30\x01\x62\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'inference_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _globals['_INITINFERENCEREQUEST']._serialized_start = 36
    _globals['_INITINFERENCEREQUEST']._serialized_end = 75
    _globals['_INITINFERENCERESPONSE']._serialized_start = 77
    _globals['_INITINFERENCERESPONSE']._serialized_end = 116
    _globals['_STREAMINFERENCEREQUEST']._serialized_start = 118
    _globals['_STREAMINFERENCEREQUEST']._serialized_end = 181
    _globals['_INFERENCEREQUEST']._serialized_start = 183
    _globals['_INFERENCEREQUEST']._serialized_end = 233
    _globals['_OBJECTMETA']._serialized_start = 235
    _globals['_OBJECTMETA']._serialized_end = 295
    _globals['_SCORES']._serialized_start = 297
    _globals['_SCORES']._serialized_end = 321
    _globals['_BOXES']._serialized_start = 323
    _globals['_BOXES']._serialized_end = 345
    _globals['_CLASSES']._serialized_start = 347
    _globals['_CLASSES']._serialized_end = 373
    _globals['_CLASSIFIER']._serialized_start = 376
    _globals['_CLASSIFIER']._serialized_end = 511
    _globals['_IMAGE']._serialized_start = 513
    _globals['_IMAGE']._serialized_end = 584
    _globals['_INFERENCERESULT']._serialized_start = 587
    _globals['_INFERENCERESULT']._serialized_end = 782
    _globals['_INFERENCE']._serialized_start = 785
    _globals['_INFERENCE']._serialized_end = 1069
# @@protoc_insertion_point(module_scope)
