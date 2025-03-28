# Copyright Axelera AI, 2024
# helper functions for building gst pipelines
import enum
import itertools
from pathlib import Path
import re
import subprocess
from typing import List

import gi
import tqdm

gi.require_version("Gst", "1.0")
from gi.repository import GLib, GObject, Gst

from .. import logging_utils

LOG = logging_utils.getLogger(__name__)
_counters = {}


def _gst_iterate(gst_iter: Gst.Iterator):
    items = []
    while 1:
        ok, value = gst_iter.next()
        if ok == Gst.IteratorResult.DONE:
            break
        items.append(value)
    return items


class InitState(enum.IntEnum):
    started = 0
    pipeline_created = enum.auto()
    connecting_elements = enum.auto()
    stream_starting = enum.auto()
    stream_ready = enum.auto()
    stream_paused = enum.auto()
    stream_playing = enum.auto()
    first_frame_received = enum.auto()


class InitProgress:
    def __init__(self):
        self._current = InitState.started
        desc = 'GStreamer Initialization'
        self._progress = tqdm.tqdm(total=len(InitState) - 1, desc=desc, unit='', leave=False)

    def __enter__(self):
        self._progress.__enter__()
        return self

    def __exit__(self, *exc):
        self._progress.__exit__(*exc)

    def set_state(self, state: InitState):
        if state > self._current:
            self._progress.set_description_str(state.name.replace('_', ' ').title())
            self._progress.update(int(state) - int(self._current))
            self._current = state

    def on_message(self, message: Gst.Message, pipeline: Gst.Pipeline, logging_dir: Path):
        '''Update initialistion state, and forward to gst_on_message.'''
        if message.type == Gst.MessageType.STATE_CHANGED:
            _, new, _ = message.parse_state_changed()
            if new == Gst.State.READY:
                self.set_state(InitState.stream_ready)
            elif new == Gst.State.PAUSED:
                self.set_state(InitState.stream_paused)
            elif new == Gst.State.PLAYING:
                self.set_state(InitState.stream_playing)
        elif message.type == Gst.MessageType.STREAM_START:
            self.set_state(InitState.stream_starting)
        return gst_on_message(message, pipeline, logging_dir)


def _on_pad_added(element, newPad, sinkPad, prop, key):
    LOG.trace("Received new pad '{}' from '{}'".format(newPad.get_name(), element.get_name()))
    if sinkPad.is_linked() or newPad.is_linked():
        LOG.trace("We are already linked. Ignoring.")
        return
    template = newPad.get_pad_template()
    if template.get_name() != key:
        LOG.trace(
            f"Pad template name {template.get_name()} does not match requested {key}. Ignoring."
        )
        return
    LOG.trace("Linking {} with {} ".format(element.get_name(), sinkPad.get_parent().get_name()))
    ret = newPad.link(sinkPad)
    if not ret == Gst.PadLinkReturn.OK:
        LOG.warning("Link failed")
        return
    _set_pad_props(newPad, prop)


def _set_pad_props(pad, props):
    for propKey in props.keys():
        if "." not in propKey:
            continue
        propConfig = propKey.split(".")
        if pad.get_name() != propConfig[0]:
            continue
        _set_element_or_pad_properties(pad, propConfig[1], props[propKey])


def _set_element_or_pad_properties(element_or_pad, propKey, propValueStr):
    LOG.trace("Setting property {}.{}={}".format(element_or_pad.get_name(), propKey, propValueStr))
    if (prop := element_or_pad.find_property(propKey)) is None:
        raise RuntimeError(f'Failed to find property {propKey} in {element_or_pad.get_name()}')
    property_type = GObject.type_name(prop.value_type)
    if property_type == 'GstCaps':
        propValue = Gst.Caps.from_string(propValueStr)
    elif property_type == 'GstFraction':
        fracs = propValueStr.split('/')
        propValue = Gst.Fraction(int(fracs[0]), int(fracs[1]))
    elif property_type == 'GstElement':
        propValue = Gst.ElementFactory.make(propValueStr)
    elif property_type == 'GstValueArray':
        vals = propValueStr.split('/')
        propValue = Gst.ValueArray(vals)
    elif property_type == 'GValueArray':
        values = propValueStr.split(',')
        value_array = GObject.ValueArray.new(0)
        for value in values:
            gvalue = GObject.Value()
            gvalue.init(GObject.TYPE_DOUBLE)  # TODO add support for other gvalue types
            gvalue.set_double(float(value))
            value_array.append(gvalue)
        propValue = value_array

    else:
        propValue = propValueStr
    element_or_pad.set_property(propKey, propValue)


def _connect(key, connectionKey, element, elements, props):
    sinkPad = None
    if "." in key:
        otherElementNameAndPad = key.split(".")
        otherElementName = otherElementNameAndPad[0]
        otherElementPadName = otherElementNameAndPad[1]
        otherElement = elements[otherElementName]
    else:
        otherElementName = key
        otherElement = elements[otherElementName]
    if otherElement is None:
        raise ValueError(f"{otherElementName} not found")

    if "%" in connectionKey:
        srcPadTemplate = element.get_pad_template(connectionKey)
        if srcPadTemplate is None:
            raise ValueError(f"Failed to find pad template {connectionKey} in {element.name}")
        sinkPad = otherElement.get_static_pad(otherElementPadName)
        _set_pad_props(sinkPad, props[otherElementName])
        if srcPadTemplate.presence == Gst.PadPresence.REQUEST:
            srcPad = element.request_pad(srcPadTemplate, None, None)
            LOG.trace(
                "Linking to request pad %s.%s with %s.%s ",
                element.get_name(),
                srcPad.get_name(),
                otherElementName,
                sinkPad.get_name(),
            )
            srcPad.link(sinkPad)
            _set_pad_props(srcPad, props[element.get_name()])
        else:
            LOG.trace("Deferring linking %s with %s ", element.get_name(), otherElementName)
            element.connect(
                "pad-added", _on_pad_added, sinkPad, props[element.get_name()], connectionKey
            )
    elif "auto" == connectionKey:
        LOG.trace("Auto linking {} with {} ".format(element.get_name(), otherElementName))
        sinkPad = otherElement
        element.link(otherElement)
    else:
        srcPad = element.get_static_pad(connectionKey)
        if "%" in otherElementPadName:
            sinkPadTemplate = otherElement.get_pad_template(otherElementPadName)
            sinkPad = otherElement.request_pad(sinkPadTemplate, None, None)
        else:
            sinkPad = otherElement.get_static_pad(otherElementPadName)
        LOG.trace(
            f"Explicit linking {element.get_name()}.{srcPad.get_name()} with {otherElementName}.{sinkPad.get_name()} "
        )
        srcPad.link(sinkPad)
        _set_pad_props(srcPad, props[element.get_name()])
        _set_pad_props(sinkPad, props[otherElementName])
    return sinkPad.get_name()


def _update_counters(fullname):
    if not bool(re.search(r"\d+$", fullname)):
        _counters[fullname] = 0
        return

    match = re.match(r"^(.*?)(\d+)$", fullname)

    name = match.group(1)
    id = int(match.group(2))
    if name not in _counters:
        _counters[name] = id
    elif id > _counters[name]:
        _counters[name] = id


def _generate_name(inst):
    if inst not in _counters:
        _counters[inst] = 0
    else:
        _counters[inst] += 1
    name = inst + str(_counters[inst])
    while name in _counters:
        _counters[inst] += 1
        name = inst + str(_counters[inst])
    return name


def gst_remove_source(agg_pad_name: str, pipeline: Gst.Pipeline):
    def property_exists(element, property_name):
        try:
            # Get the GObject class for the element
            gclass = GObject.Object.get_class(element)

            # Check if the property exists using g_object_class_find_property
            prop = gclass.find_property(property_name)
            return prop is not None
        except Exception as e:
            return False

    def traverse_pad(pad):
        if not pad:
            return
        peer_pad = pad.get_peer()
        if peer_pad:
            parent_element = peer_pad.get_parent_element()
            if not parent_element:
                return
            if peer_pad.is_linked() and Gst.PadDirection.SRC == peer_pad.get_direction():
                LOG.trace(
                    f"Disconnecting {pad.get_parent_element().get_name()} and {parent_element.get_name()}"
                )
                peer_pad.unlink(pad)

            for npad in _gst_iterate(parent_element.iterate_sink_pads()):
                traverse_pad(npad)

            LOG.info(f"Setting state of {parent_element.get_name()} to READY")
            parent_element.set_state(Gst.State.READY)
            LOG.trace(f"State of {parent_element.get_name()} to READY done")
            LOG.trace(f"Setting state of {parent_element.get_name()} to NULL")
            parent_element.set_state(Gst.State.NULL)
            LOG.trace(f"State of {parent_element.get_name()} to NULL done")

            state_change_return, current, pending = parent_element.get_state(timeout=Gst.SECOND)
            if current == Gst.State.NULL:
                LOG.trace(f"{parent_element.get_name()} is now in NULL state")
            else:
                LOG.error(f"Failed to transition to NULL {parent_element.get_name()}")
            LOG.info(f"Removing {parent_element.get_name()}")
            pipeline.remove(parent_element)
            LOG.trace(f"Removing {parent_element.get_name()} done")

    agg_name = (
        "inference-funnel"
        if pipeline.get_by_name("inference-funnel") is not None
        else "inference-task0"
    )
    agg = pipeline.get_by_name(agg_name)
    if agg is None:
        raise RuntimeError("Aggregator not found")

    for pad in _gst_iterate(agg.iterate_sink_pads()):
        if pad.get_name() == agg_pad_name:
            traverse_pad(pad)
            break

    agg.release_request_pad(pad)
    return pipeline


def add_gst_input(yamlData: dict, pipeline: Gst.Pipeline):
    elements = {}
    props = {}
    agg_pad_name = None
    is_live_src = False

    for elementEntry in yamlData:
        # create Gst.Element by plugin name
        if "name" not in elementEntry:
            elementEntry["name"] = _generate_name(elementEntry["instance"])
        if elementEntry["instance"].startswith('rtsp'):
            is_live_src = True
        props[elementEntry["name"]] = elementEntry
        instance, name = elementEntry["instance"], elementEntry["name"]
        element = Gst.ElementFactory.make(instance, name)
        if element is None:
            raise RuntimeError(f"Failed to create element of type {instance} ({name})")
        # add element to pipeline
        pipeline.add(element)

        if elementEntry["instance"] == "capsfilter":
            caps = Gst.Caps.from_string(elementEntry["caps"])
            element.set_property("caps", caps)
        for propKey in elementEntry.keys():
            if (
                propKey == "instance"
                or propKey == "name"
                or propKey == "connections"
                or "." in propKey
            ):
                continue
            _set_element_or_pad_properties(element, propKey, elementEntry[propKey])

        LOG.trace("Creating {} from {}".format(elementEntry["name"], elementEntry["instance"]))
        elements[elementEntry["name"]] = element
        if "connections" in elementEntry:
            elements[elementEntry["name"] + "connections"] = elementEntry["connections"]
        elif "sink" not in elementEntry["instance"]:
            for id, elem in enumerate(yamlData):
                if elem["name"] == elementEntry["name"]:
                    nextElement = yamlData[id + 1]
                    if "name" not in nextElement:
                        nextElement["name"] = _generate_name(nextElement["instance"])
                    else:
                        _update_counters(nextElement["name"])
                    elements[elementEntry["name"] + "connections"] = {"auto": nextElement["name"]}
                    break
        if is_live_src is True:
            element.set_state(Gst.State.PLAYING)
    agg_name = (
        "inference-funnel"
        if pipeline.get_by_name("inference-funnel") is not None
        else "inference-task0"
    )

    elements[agg_name] = pipeline.get_by_name(agg_name)
    props[agg_name] = {}
    for elementKey in elements.keys():
        if "connections" not in elementKey:
            continue
        elementName = elementKey.replace("connections", "")
        element = elements[elementName]
        for connectionKey in elements[elementKey].keys():
            if isinstance(elements[elementKey][connectionKey], list):
                for key in elements[elementKey][connectionKey]:
                    pad_name = _connect(key, connectionKey, element, elements, props)
            else:
                key = elements[elementKey][connectionKey]
                pad_name = _connect(key, connectionKey, element, elements, props)
                if key.split('.')[0] == agg_name:
                    agg_pad_name = pad_name
    return pipeline, agg_pad_name


def get_agg_pads(pipeline: Gst.Pipeline, new_inference: bool):
    pads = list()
    agg_name = (
        "inference-funnel" if not new_inference else "inference-task0"
    )  # Assuming single model pipeline
    agg = pipeline.get_by_name(agg_name)

    if agg is None:
        return pads
    it = agg.iterate_sink_pads()
    while True:
        result, pad = it.next()
        if result != Gst.IteratorResult.OK:
            break
        pads.append(pad.get_name())

    return pads


def build_gst_pipelines(
    yamlData: dict, pipeline_names: List[str] = None, progress: InitProgress = None
):
    if not Gst.is_initialized():
        Gst.init(None)

    pipelines = []
    for pipelineEntry in yamlData:
        pipeline = Gst.Pipeline()
        if progress:
            progress.set_state(InitState.pipeline_created)
        pipeline.set_name('_'.join(pipeline_names))
        elements = {}
        props = {}
        for elementsEntriesKeys in pipelineEntry.keys():
            for elementEntryId in range(len(pipelineEntry[elementsEntriesKeys])):
                elementEntry = pipelineEntry[elementsEntriesKeys][elementEntryId]
                # create Gst.Element by plugin name
                if "name" not in elementEntry:
                    elementEntry["name"] = _generate_name(elementEntry["instance"])
                    pipelineEntry[elementsEntriesKeys][elementEntryId]["name"] = elementEntry[
                        "name"
                    ]
                else:
                    _update_counters(elementEntry["name"])
                props[elementEntry["name"]] = elementEntry
                instance, name = elementEntry["instance"], elementEntry["name"]
                element = Gst.ElementFactory.make(instance, name)
                if element is None:
                    raise RuntimeError(f"Failed to create element of type {instance} ({name})")

                for propKey in elementEntry.keys():
                    if (
                        propKey == "instance"
                        or propKey == "name"
                        or propKey == "connections"
                        or "." in propKey
                    ):
                        continue
                    _set_element_or_pad_properties(element, propKey, elementEntry[propKey])

                # add element to pipeline
                pipeline.add(element)
                LOG.trace(
                    "Creating {} from {}".format(elementEntry["name"], elementEntry["instance"])
                )
                elements[elementEntry["name"]] = element
                if "connections" in elementEntry:
                    elements[elementEntry["name"] + "connections"] = elementEntry["connections"]
                elif "sink" not in elementEntry["instance"]:
                    temp = list(pipelineEntry["pipeline"])
                    for id in temp:
                        if id["name"] == elementEntry["name"]:
                            nextElement = pipelineEntry[elementsEntriesKeys][
                                temp.index(elementEntry) + 1
                            ]
                            if "name" not in nextElement:
                                nextElement["name"] = _generate_name(nextElement["instance"])
                            else:
                                _update_counters(nextElement["name"])
                            elements[elementEntry["name"] + "connections"] = {
                                "auto": nextElement["name"]
                            }
                            break

        if progress:
            progress.set_state(InitState.connecting_elements)
        for elementKey in elements.keys():
            if "connections" not in elementKey:
                continue
            elementName = elementKey.replace("connections", "")
            element = elements[elementName]
            for connectionKey in elements[elementKey].keys():
                if isinstance(elements[elementKey][connectionKey], list):
                    for key in elements[elementKey][connectionKey]:
                        _connect(key, connectionKey, element, elements, props)
                else:
                    key = elements[elementKey][connectionKey]
                    _connect(key, connectionKey, element, elements, props)

        pipelines.append(pipeline)
    return pipelines


def _dump_pipeline_graph(pipeline: Gst.Pipeline, pipeline_dot_file: Path):
    pipeline_dot_file.write_text(Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL))

    try:
        result = subprocess.run(
            [
                "dot",
                "-Tpng",
                str(pipeline_dot_file),
                "-o",
                str(pipeline_dot_file.with_suffix(".png")),
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        LOG.warning("Error occurred while converting .dot to .png")


def gst_on_message(message: Gst.Message, pipeline: Gst.Pipeline, logging_dir: Path = Path.cwd()):
    '''Callback function for watching the GST pipeline. Dump the pipeline graph to a .dot file
    and convert it to a .png file once the stream starts playing.'''
    mType = message.type
    if mType == Gst.MessageType.EOS:
        LOG.debug("End of stream")
        exit(0)
    elif mType == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        LOG.error('%s: %s\n -> %s', message.src.get_name(), err.message, debug)
        try:
            pipeline.set_state(Gst.State.NULL)
        except Exception as e:
            # we really don't want to propagate any other error here
            LOG.error("Failed to set pipeline NULL state: %s", str(e))
        exit(1)
    elif mType == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        LOG.warning('%s: %s\n (%s)', message.src.get_name(), err.message, debug)
    elif mType == Gst.MessageType.STATE_CHANGED:
        old, new, pending = message.parse_state_changed()
        if LOG.isEnabledFor(logging_utils.TRACE) and pipeline == message.src:
            pipeline_dot_file = (
                logging_dir
                / f"pipeline_graph_{Gst.Element.state_get_name(old)}_to_{Gst.Element.state_get_name(new)}.dot"
            )
            LOG.trace(
                f"State change in GST pipeline: {pipeline.get_name()}, writing graph to {pipeline_dot_file}"
            )
            _dump_pipeline_graph(pipeline, pipeline_dot_file)
    elif mType == Gst.MessageType.STREAM_START and LOG.isEnabledFor(logging_utils.TRACE):
        pipeline_dot_file = logging_dir / "pipeline_graph_STREAM_START.dot"
        LOG.trace(
            f"Playing GST pipeline: {pipeline.get_name()}, writing graph to {pipeline_dot_file}"
        )
        _dump_pipeline_graph(pipeline, pipeline_dot_file)
    return True
