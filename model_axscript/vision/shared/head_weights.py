# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Where an output head's weights come from: a file the card names, or the published one.

Segmentation and depth each ship one head per backbone size and each let a customer name their
own file instead. Everything that DIFFERS between the two is a field of `PublishedHead`, which
each of them declares once; everything that is the same is the code here. So a third head type
is one more declaration rather than a third copy of the five functions that resolve a URL, a
cache path and a digest, and a change to the rule cannot reach one head and not the others.

What a head file must CONTAIN is declared the same way, as `layout` plus `pointwise` plus a line
naming the topology. The recipe authors the device graph for one head shape per task, so a file
of another shape has to be refused rather than compiled into something it is not, and the refusal
is worth more than a list of missing keys: it says what this card's head is.

The three checks are here rather than in either loader because they answer one question, is this
file the head this card compiles, and a check that reached one head and not the other is the
defect it exists to prevent. Presence of the layout is the weakest of the three: a deeper decoder
in the same layout satisfies it, and a convolution wider than 1x1 satisfies it AND the width
check, reading back as a multiple of the real class count.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import urllib.request

#: Where mmsegmentation writes a decode head's parameters, which is the layout both published
#: heads are released in. Every key under it is a LAYER, so one this card's `layout` does not
#: name is a layer the recipe has no device graph for.
_HEAD_PREFIX = "decode_head."
#: The one key under that prefix that is bookkeeping rather than a layer: BatchNorm's update
#: counter, which Meta's own ADE20K head carries and nothing reads.
_BOOKKEEPING = "num_batches_tracked"


@dataclass(frozen=True)
class PublishedHead:
    """One task's published head: where it lives, what it hashes to, and what it must contain.

    `digests` carries the backbone sizes as its keys, so the set a card can name and the set
    whose content is pinned are one thing and cannot drift apart.
    """

    #: What this head does, in a reader's words, for a refusal that names it ("segmentation").
    what: str
    #: Where the publisher keeps it, and how the per-size directory and file are spelled.
    base: str
    stem: str  # e.g. "dinov2_{size}"
    filename: str  # e.g. "{stem}_ade20k_linear_head.pth"
    #: The leaf under the shared cache root, so two head types cannot overwrite each other.
    cache_leaf: str
    #: sha256 per backbone size, computed from what the URLs below serve. The recipe was
    #: measured against exactly this content, so a build refuses anything else rather than
    #: compiling a graph whose weights nobody has scored.
    digests: dict
    #: The state-dict keys the loader reads. A file missing any of them is not this head.
    layout: tuple
    #: The key of the head's 1x1 convolution, whose OUTPUT width is the class or bin count.
    #: Named here because that count is read by reshaping the weight against the input width,
    #: which silently absorbs a wider kernel: a 3x3 conv onto 12 classes reads back as 108.
    pointwise: str
    #: One line naming the topology `layout` describes, for the refusal a customer reads.
    topology: str

    @property
    def sizes(self) -> tuple:
        """The backbone sizes a head is published for, which is what `digests` covers."""
        return tuple(self.digests)

    def _stem(self, size: str) -> str:
        return self.stem.format(size=size)

    def url(self, size: str) -> str:
        """Where the head for `size` is published, refusing a size nothing is published for.

        Checked here and not only at the fetch, because this is also what a caller derives a
        card's DECLARED weight addresses from: formatting a plausible URL for a size that has
        no head would declare an address nothing serves.
        """
        if size not in self.digests:
            raise ValueError(
                f"no {self.what} head is published for backbone {size!r}; there is one for "
                f"{', '.join(self.sizes)}"
            )
        return f"{self.base}/{self._stem(size)}/{self.cache_name(size)}"

    def cache_name(self, size: str) -> str:
        """The cache filename for `size`, so two sizes cannot overwrite each other."""
        return self.filename.format(stem=self._stem(size))

    def cache_dir(self, cache_dir=None) -> Path:
        """Where downloaded heads live; shares the DINOv2 cache root, outside the repository."""
        if cache_dir is not None:
            return Path(cache_dir)
        from model_axscript.llm.models.dino.model import default_cache_root

        return Path(default_cache_root()) / self.cache_leaf

    def file(self, spec, *, cache_dir=None) -> Path:
        """The `.pth` `spec` names: your own file, or the published head for a backbone size."""
        directory = self.cache_dir(cache_dir)
        return head_file(
            spec,
            sizes=self.sizes,
            digests=self.digests,
            url_for=self.url,
            path_for=lambda size: directory / self.cache_name(size),
            what=self.what,
        )

    def state_dict(self, spec, *, cache_dir=None, safe_globals=None) -> dict:
        """The head's parameters, refused by name unless the file holds `layout`.

        The check runs on a published head and a customer's alike, so a file that is not this
        head is caught here rather than reaching the first matrix multiply inside the compiler
        as a shape error naming neither side. `safe_globals` is the allowlist a head whose file
        carries training metadata needs; the load stays `weights_only` either way.
        """
        import torch

        path = self.file(spec, cache_dir=cache_dir)
        if safe_globals is None:
            state = torch.load(path, map_location="cpu", weights_only=True)
        else:
            with torch.serialization.safe_globals(safe_globals()):
                state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        missing = [key for key in self.layout if key not in state]
        if missing:
            raise ValueError(
                f"{path} is not a {self.what} head this card can compile: missing {missing}. "
                f"{self.topology} Save your head as those layers, not as one you have already "
                f"collapsed."
            )
        self._refuse_extra_layers(state, path)
        self._refuse_wide_kernel(state, path)
        return state

    def _refuse_extra_layers(self, state: dict, path) -> None:
        """Refuse a file that carries LAYERS beyond the ones the recipe compiles.

        Presence of the layout is not enough. A deeper decoder in the same mmsegmentation
        layout satisfies it and everything the recipe does not name is then dropped without a
        word, so the graph runs the file's last layer against weights trained to sit behind
        several. The doc invites an mmsegmentation checkpoint by name, which is exactly the
        family whose heads share these key names, so this is the likely file rather than the
        exotic one.
        """
        extra = sorted(
            key
            for key in state
            if key.startswith(_HEAD_PREFIX)
            and key not in self.layout
            and not key.endswith(_BOOKKEEPING)
        )
        if extra:
            raise ValueError(
                f"{path} carries {self.what} head layers this card has no device graph for: "
                f"{extra}. {self.topology} Compiling it would run that head's last layer alone "
                f"and drop the rest, so it is refused instead. Train the head this card "
                f"compiles, or ask for a card that compiles yours."
            )

    def _refuse_wide_kernel(self, state: dict, path) -> None:
        """Refuse a convolution wider than 1x1, which no width or class check catches.

        The class count is the conv's output width, read by reshaping the weight against the
        backbone's. A 3x3 kernel reshapes cleanly and reports nine times the real count, so the
        card compiles a graph for classes the head does not predict and nothing before this
        said so.
        """
        weight = state[self.pointwise]
        spatial = tuple(int(d) for d in weight.shape[2:])
        if weight.ndim not in (2, 4) or any(d != 1 for d in spatial):
            kernel = "x".join(str(d) for d in spatial) if spatial else f"{weight.ndim}-D"
            raise ValueError(
                f"{path} is not a {self.what} head this card can compile: {self.pointwise} has "
                f"a {kernel} kernel and this card compiles a 1x1. {self.topology}"
            )


def head_file(head, *, sizes, digests, url_for, path_for, what) -> Path:
    """The file `head` names: the card's own, or the published head for a backbone size.

    A path that exists on disk is used as it is and nothing is fetched. Anything else names one
    of the published sizes, is downloaded once into the shared cache, and has its content
    pinned, so a card that names a size always compiles the weights the recipe was measured
    against.

    The per-size facts stay with the loader that owns them and arrive as `sizes`, `digests`,
    `url_for` and `path_for`; only the RULE for choosing between a local file and a published
    one lives here, which is the whole reason this module exists.
    """
    local = local_file(head)
    if local is not None:
        return local
    size = str(head)
    if size not in sizes:
        refuse_unknown(head, sizes, what)
    return fetch_pinned(url_for(size), path_for(size), digests[size])


def local_file(spec) -> Path | None:
    """`spec` as a head file on disk, or None when it names a published backbone size.

    The filesystem is the signal, which is what the rest of the tree already uses to tell a
    local input from a remote id: `calibration.load_calibration_images` reads a directory of
    images rather than a dataset id, and `transformers.from_pretrained` reads a checkpoint
    directory rather than a repo id.
    """
    path = Path(str(spec)).expanduser()
    return path if path.is_file() else None


def refuse_unknown(head, sizes: tuple[str, ...], what: str) -> None:
    """Refuse a head spec that is neither a file on disk nor a published backbone size.

    Without this, a mistyped path falls through to the published branch and comes back as an
    unknown BACKBONE SIZE, which sends someone looking for the wrong mistake entirely.
    """
    raise ValueError(
        f"{head!r} is neither a file on disk nor a backbone size a {what} head is published "
        f"for; Meta ships one for {', '.join(sizes)}. Give the path to your own head file, or "
        f"one of those names."
    )


def fetch_pinned(url: str, path: Path, digest: str) -> Path:
    """`path`, downloaded from `url` once, refused unless its sha256 is `digest`.

    A head fetched over the network decides what the device graph computes, and the only check
    before this was that a few state-dict keys were present, which a truncated or substituted
    file can still satisfy. The digest is verified on a cached file too, so a cache corrupted
    after the download is caught rather than reused for every build after it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"[head] downloading {url} -> {path}")
        # Fetch beside the cache entry and rename, so an interrupted download cannot leave a
        # truncated file at the path every later build reads. The pid keeps two builds racing
        # on the shared cache off each other's temporary.
        part = path.with_name(f"{path.name}.{os.getpid()}.part")
        try:
            urllib.request.urlretrieve(url, part)
            os.replace(part, path)
        finally:
            part.unlink(missing_ok=True)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError(
            f"{path} is not the file this recipe is pinned to: its sha256 is {actual}, and "
            f"{url} is pinned to {digest}. Delete the file so the build fetches it again, or "
            f"name your own head file in place of a published backbone size."
        )
    return path
