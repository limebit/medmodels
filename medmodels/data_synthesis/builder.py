"""Builder for data Synthesizers."""

from __future__ import annotations

from medmodels.data_synthesis.mtgan.builder import MTGANBuilder


class SynthesizerBuilder:
    def from_mtgan(self) -> MTGANBuilder:
        """Creates an MTGANBuilder instance from a SynthesizerBuilder instance.

        Returns:
            MTGANBuilder: An instance of MTGANBuilder.
        """
        return MTGANBuilder()
