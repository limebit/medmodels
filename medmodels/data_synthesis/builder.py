from __future__ import annotations

from medmodels.data_synthesis.mtgan.builder import MTGANBuilder


class SynthesizerBuilder:
    def from_mtgan(self) -> MTGANBuilder:
        """Creates a SynthesizerBuilder instance from an MTGANBuilder instance.

        Args:
            mtgan_builder (MTGANBuilder): The MTGANBuilder instance.

        Returns:
            SynthesizerBuilder: The SynthesizerBuilder instance.
        """
        return MTGANBuilder()
