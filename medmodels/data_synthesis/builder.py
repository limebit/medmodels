"""Builder for data Synthesizers."""

from __future__ import annotations

from medmodels.data_synthesis.mtgan.builder import MTGANBuilder


class SynthesizerBuilder:
    def from_mtgan(self) -> MTGANBuilder: ...
