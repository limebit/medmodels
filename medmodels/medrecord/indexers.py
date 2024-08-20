from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple, Union, overload

from medmodels.medrecord.querying import EdgeQuery, NodeQuery
from medmodels.medrecord.types import (
    Attributes,
    AttributesInput,
    EdgeIndex,
    EdgeIndexInputList,
    MedRecordAttribute,
    MedRecordAttributeInputList,
    MedRecordValue,
    NodeIndex,
    NodeIndexInputList,
    is_attributes,
    is_edge_index,
    is_medrecord_attribute,
    is_medrecord_value,
    is_node_index,
)

if TYPE_CHECKING:
    from medmodels import MedRecord


class NodeIndexer:
    _medrecord: MedRecord

    def __init__(self, medrecord: MedRecord) -> None:
        self._medrecord = medrecord

    @overload
    def __getitem__(
        self,
        key: Union[
            NodeIndex, Tuple[NodeIndex, Union[MedRecordAttributeInputList, slice]]
        ],
    ) -> Attributes: ...

    @overload
    def __getitem__(
        self, key: Tuple[NodeIndex, MedRecordAttribute]
    ) -> MedRecordValue: ...

    @overload
    def __getitem__(
        self,
        key: Union[
            NodeIndexInputList,
            NodeQuery,
            slice,
            Tuple[
                Union[NodeIndexInputList, NodeQuery, slice],
                Union[MedRecordAttributeInputList, slice],
            ],
        ],
    ) -> Dict[NodeIndex, Attributes]: ...

    @overload
    def __getitem__(
        self,
        key: Tuple[Union[NodeIndexInputList, NodeQuery, slice], MedRecordAttribute],
    ) -> Dict[NodeIndex, MedRecordValue]: ...

    def __getitem__(
        self,
        key: Union[
            NodeIndex,
            NodeIndexInputList,
            NodeQuery,
            slice,
            Tuple[
                Union[NodeIndex, NodeIndexInputList, NodeQuery, slice],
                Union[MedRecordAttribute, MedRecordAttributeInputList, slice],
            ],
        ],
    ) -> Union[
        MedRecordValue,
        Attributes,
        Dict[NodeIndex, Attributes],
        Dict[NodeIndex, MedRecordValue],
    ]:
        if is_node_index(key):
            return self._medrecord._medrecord.node([key])[key]

        if isinstance(key, list):
            return self._medrecord._medrecord.node(key)

        if isinstance(key, NodeQuery):
            return self._medrecord._medrecord.node(self._medrecord.select_nodes(key))

        if isinstance(key, slice):
            if key.start is not None or key.stop is not None or key.step is not None:
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.node(self._medrecord.nodes)

        index_selection, attribute_selection = key

        if is_node_index(index_selection) and is_medrecord_attribute(
            attribute_selection
        ):
            return self._medrecord._medrecord.node([index_selection])[index_selection][
                attribute_selection
            ]

        if isinstance(index_selection, list) and is_medrecord_attribute(
            attribute_selection
        ):
            attributes = self._medrecord._medrecord.node(index_selection)

            return {x: attributes[x][attribute_selection] for x in attributes.keys()}

        if isinstance(index_selection, NodeQuery) and is_medrecord_attribute(
            attribute_selection
        ):
            attributes = self._medrecord._medrecord.node(
                self._medrecord.select_nodes(index_selection)
            )

            return {x: attributes[x][attribute_selection] for x in attributes.keys()}

        if isinstance(index_selection, slice) and is_medrecord_attribute(
            attribute_selection
        ):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            attributes = self._medrecord._medrecord.node(self._medrecord.nodes)

            return {x: attributes[x][attribute_selection] for x in attributes.keys()}

        if is_node_index(index_selection) and isinstance(attribute_selection, list):
            return {
                x: self._medrecord._medrecord.node([index_selection])[index_selection][
                    x
                ]
                for x in attribute_selection
            }

        if isinstance(index_selection, list) and isinstance(attribute_selection, list):
            attributes = self._medrecord._medrecord.node(index_selection)

            return {
                x: {y: attributes[x][y] for y in attribute_selection}
                for x in attributes.keys()
            }

        if isinstance(index_selection, NodeQuery) and isinstance(
            attribute_selection, list
        ):
            attributes = self._medrecord._medrecord.node(
                self._medrecord.select_nodes(index_selection)
            )

            return {
                x: {y: attributes[x][y] for y in attribute_selection}
                for x in attributes.keys()
            }

        if isinstance(index_selection, slice) and isinstance(attribute_selection, list):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            attributes = self._medrecord._medrecord.node(self._medrecord.nodes)

            return {
                x: {y: attributes[x][y] for y in attribute_selection}
                for x in attributes.keys()
            }

        if is_node_index(index_selection) and isinstance(attribute_selection, slice):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.node([index_selection])[index_selection]

        if isinstance(index_selection, list) and isinstance(attribute_selection, slice):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.node(index_selection)

        if isinstance(index_selection, NodeQuery) and isinstance(
            attribute_selection, slice
        ):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.node(
                self._medrecord.select_nodes(index_selection)
            )

        if isinstance(index_selection, slice) and isinstance(
            attribute_selection, slice
        ):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
                or attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.node(self._medrecord.nodes)

    @overload
    def __setitem__(
        self,
        key: Union[NodeIndex, NodeIndexInputList, NodeQuery, slice],
        value: AttributesInput,
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        key: Tuple[
            Union[NodeIndex, NodeIndexInputList, NodeQuery, slice],
            Union[MedRecordAttribute, MedRecordAttributeInputList, slice],
        ],
        value: MedRecordValue,
    ) -> None: ...

    def __setitem__(
        self,
        key: Union[
            NodeIndex,
            NodeIndexInputList,
            NodeQuery,
            slice,
            Tuple[
                Union[NodeIndex, NodeIndexInputList, NodeQuery, slice],
                Union[MedRecordAttribute, MedRecordAttributeInputList, slice],
            ],
        ],
        value: Union[AttributesInput, MedRecordValue],
    ) -> None:
        if is_node_index(key):
            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_node_attributes([key], value)

        if isinstance(key, list):
            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_node_attributes(key, value)

        if isinstance(key, NodeQuery):
            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_node_attributes(
                self._medrecord.select_nodes(key), value
            )

        if isinstance(key, slice):
            if key.start is not None or key.stop is not None or key.step is not None:
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_node_attributes(
                self._medrecord.nodes, value
            )

        index_selection, attribute_selection = key

        if is_node_index(index_selection) and is_medrecord_attribute(
            attribute_selection
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            return self._medrecord._medrecord.update_node_attribute(
                [index_selection], attribute_selection, value
            )

        if isinstance(index_selection, list) and is_medrecord_attribute(
            attribute_selection
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            return self._medrecord._medrecord.update_node_attribute(
                index_selection, attribute_selection, value
            )

        if isinstance(index_selection, NodeQuery) and is_medrecord_attribute(
            attribute_selection
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            return self._medrecord._medrecord.update_node_attribute(
                self._medrecord.select_nodes(index_selection),
                attribute_selection,
                value,
            )

        if isinstance(index_selection, slice) and is_medrecord_attribute(
            attribute_selection
        ):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            return self._medrecord._medrecord.update_node_attribute(
                self._medrecord.nodes,
                attribute_selection,
                value,
            )

        if is_node_index(index_selection) and isinstance(attribute_selection, list):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            for attribute in attribute_selection:
                self._medrecord._medrecord.update_node_attribute(
                    [index_selection], attribute, value
                )

            return

        if isinstance(index_selection, list) and isinstance(attribute_selection, list):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            for attribute in attribute_selection:
                self._medrecord._medrecord.update_node_attribute(
                    index_selection, attribute, value
                )

            return

        if isinstance(index_selection, NodeQuery) and isinstance(
            attribute_selection, list
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            for attribute in attribute_selection:
                self._medrecord._medrecord.update_node_attribute(
                    self._medrecord.select_nodes(index_selection), attribute, value
                )

            return

        if isinstance(index_selection, slice) and isinstance(attribute_selection, list):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            for attribute in attribute_selection:
                self._medrecord._medrecord.update_node_attribute(
                    self._medrecord.nodes, attribute, value
                )

            return

        if is_node_index(index_selection) and isinstance(attribute_selection, slice):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            attributes = self._medrecord._medrecord.node([index_selection])[
                index_selection
            ]

            for attribute in attributes.keys():
                self._medrecord._medrecord.update_node_attribute(
                    [index_selection],
                    attribute,
                    value,
                )

            return

        if isinstance(index_selection, list) and isinstance(attribute_selection, slice):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            attributes = self._medrecord._medrecord.node(index_selection)

            for node in attributes.keys():
                for attribute in attributes[node].keys():
                    self._medrecord._medrecord.update_node_attribute(
                        [node], attribute, value
                    )

            return

        if isinstance(index_selection, NodeQuery) and isinstance(
            attribute_selection, slice
        ):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            attributes = self._medrecord._medrecord.node(
                self._medrecord.select_nodes(index_selection)
            )

            for node in attributes.keys():
                for attribute in attributes[node].keys():
                    self._medrecord._medrecord.update_node_attribute(
                        [node], attribute, value
                    )

            return

        if isinstance(index_selection, slice) and isinstance(
            attribute_selection, slice
        ):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
                or attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            attributes = self._medrecord._medrecord.node(self._medrecord.nodes)

            for node in attributes.keys():
                for attribute in attributes[node].keys():
                    self._medrecord._medrecord.update_node_attribute(
                        [node], attribute, value
                    )

            return

    def __delitem__(
        self,
        key: Tuple[
            Union[NodeIndex, NodeIndexInputList, NodeQuery, slice],
            Union[MedRecordAttribute, MedRecordAttributeInputList, slice],
        ],
    ) -> None:
        index_selection, attribute_selection = key

        if is_node_index(index_selection) and is_medrecord_attribute(
            attribute_selection
        ):
            return self._medrecord._medrecord.remove_node_attribute(
                [index_selection], attribute_selection
            )

        if isinstance(index_selection, list) and is_medrecord_attribute(
            attribute_selection
        ):
            return self._medrecord._medrecord.remove_node_attribute(
                index_selection, attribute_selection
            )

        if isinstance(index_selection, NodeQuery) and is_medrecord_attribute(
            attribute_selection
        ):
            return self._medrecord._medrecord.remove_node_attribute(
                self._medrecord.select_nodes(index_selection),
                attribute_selection,
            )

        if isinstance(index_selection, slice) and is_medrecord_attribute(
            attribute_selection
        ):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.remove_node_attribute(
                self._medrecord.nodes,
                attribute_selection,
            )

        if is_node_index(index_selection) and isinstance(attribute_selection, list):
            for attribute in attribute_selection:
                self._medrecord._medrecord.remove_node_attribute(
                    [index_selection], attribute
                )

            return

        if isinstance(index_selection, list) and isinstance(attribute_selection, list):
            for attribute in attribute_selection:
                self._medrecord._medrecord.remove_node_attribute(
                    index_selection, attribute
                )

            return

        if isinstance(index_selection, NodeQuery) and isinstance(
            attribute_selection, list
        ):
            for attribute in attribute_selection:
                self._medrecord._medrecord.remove_node_attribute(
                    self._medrecord.select_nodes(index_selection), attribute
                )

            return

        if isinstance(index_selection, slice) and isinstance(attribute_selection, list):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            for attribute in attribute_selection:
                self._medrecord._medrecord.remove_node_attribute(
                    self._medrecord.nodes, attribute
                )

            return

        if is_node_index(index_selection) and isinstance(attribute_selection, slice):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.replace_node_attributes(
                [index_selection], {}
            )

        if isinstance(index_selection, list) and isinstance(attribute_selection, slice):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.replace_node_attributes(
                index_selection, {}
            )

        if isinstance(index_selection, NodeQuery) and isinstance(
            attribute_selection, slice
        ):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.replace_node_attributes(
                self._medrecord.select_nodes(index_selection), {}
            )

        if isinstance(index_selection, slice) and isinstance(
            attribute_selection, slice
        ):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
                or attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.replace_node_attributes(
                self._medrecord.nodes, {}
            )


class EdgeIndexer:
    _medrecord: MedRecord

    def __init__(self, medrecord: MedRecord) -> None:
        self._medrecord = medrecord

    @overload
    def __getitem__(
        self,
        key: Union[
            EdgeIndex, Tuple[EdgeIndex, Union[MedRecordAttributeInputList, slice]]
        ],
    ) -> Attributes: ...

    @overload
    def __getitem__(
        self, key: Tuple[EdgeIndex, MedRecordAttribute]
    ) -> MedRecordValue: ...

    @overload
    def __getitem__(
        self,
        key: Union[
            EdgeIndexInputList,
            EdgeQuery,
            slice,
            Tuple[
                Union[EdgeIndexInputList, EdgeQuery, slice],
                Union[MedRecordAttributeInputList, slice],
            ],
        ],
    ) -> Dict[EdgeIndex, Attributes]: ...

    @overload
    def __getitem__(
        self,
        key: Tuple[Union[EdgeIndexInputList, EdgeQuery, slice], MedRecordAttribute],
    ) -> Dict[EdgeIndex, MedRecordValue]: ...

    def __getitem__(
        self,
        key: Union[
            EdgeIndex,
            EdgeIndexInputList,
            EdgeQuery,
            slice,
            Tuple[
                Union[EdgeIndex, EdgeIndexInputList, EdgeQuery, slice],
                Union[MedRecordAttribute, MedRecordAttributeInputList, slice],
            ],
        ],
    ) -> Union[
        MedRecordValue,
        Attributes,
        Dict[EdgeIndex, Attributes],
        Dict[EdgeIndex, MedRecordValue],
    ]:
        if is_edge_index(key):
            return self._medrecord._medrecord.edge([key])[key]

        if isinstance(key, list):
            return self._medrecord._medrecord.edge(key)

        if isinstance(key, EdgeQuery):
            return self._medrecord._medrecord.edge(self._medrecord.select_edges(key))

        if isinstance(key, slice):
            if key.start is not None or key.stop is not None or key.step is not None:
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.edge(self._medrecord.edges)

        index_selection, attribute_selection = key

        if is_edge_index(index_selection) and is_medrecord_attribute(
            attribute_selection
        ):
            return self._medrecord._medrecord.edge([index_selection])[index_selection][
                attribute_selection
            ]

        if isinstance(index_selection, list) and is_medrecord_attribute(
            attribute_selection
        ):
            attributes = self._medrecord._medrecord.edge(index_selection)

            return {x: attributes[x][attribute_selection] for x in attributes.keys()}

        if isinstance(index_selection, EdgeQuery) and is_medrecord_attribute(
            attribute_selection
        ):
            attributes = self._medrecord._medrecord.edge(
                self._medrecord.select_edges(index_selection)
            )

            return {x: attributes[x][attribute_selection] for x in attributes.keys()}

        if isinstance(index_selection, slice) and is_medrecord_attribute(
            attribute_selection
        ):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            attributes = self._medrecord._medrecord.edge(self._medrecord.edges)

            return {x: attributes[x][attribute_selection] for x in attributes.keys()}

        if is_edge_index(index_selection) and isinstance(attribute_selection, list):
            return {
                x: self._medrecord._medrecord.edge([index_selection])[index_selection][
                    x
                ]
                for x in attribute_selection
            }

        if isinstance(index_selection, list) and isinstance(attribute_selection, list):
            attributes = self._medrecord._medrecord.edge(index_selection)

            return {
                x: {y: attributes[x][y] for y in attribute_selection}
                for x in attributes.keys()
            }

        if isinstance(index_selection, EdgeQuery) and isinstance(
            attribute_selection, list
        ):
            attributes = self._medrecord._medrecord.edge(
                self._medrecord.select_edges(index_selection)
            )

            return {
                x: {y: attributes[x][y] for y in attribute_selection}
                for x in attributes.keys()
            }

        if isinstance(index_selection, slice) and isinstance(attribute_selection, list):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            attributes = self._medrecord._medrecord.edge(self._medrecord.edges)

            return {
                x: {y: attributes[x][y] for y in attribute_selection}
                for x in attributes.keys()
            }

        if is_edge_index(index_selection) and isinstance(attribute_selection, slice):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.edge([index_selection])[index_selection]

        if isinstance(index_selection, list) and isinstance(attribute_selection, slice):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.edge(index_selection)

        if isinstance(index_selection, EdgeQuery) and isinstance(
            attribute_selection, slice
        ):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.edge(
                self._medrecord.select_edges(index_selection)
            )

        if isinstance(index_selection, slice) and isinstance(
            attribute_selection, slice
        ):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
                or attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.edge(self._medrecord.edges)

    @overload
    def __setitem__(
        self,
        key: Union[EdgeIndex, EdgeIndexInputList, EdgeQuery, slice],
        value: AttributesInput,
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        key: Tuple[
            Union[EdgeIndex, EdgeIndexInputList, EdgeQuery, slice],
            Union[MedRecordAttribute, MedRecordAttributeInputList, slice],
        ],
        value: MedRecordValue,
    ) -> None: ...

    def __setitem__(
        self,
        key: Union[
            EdgeIndex,
            EdgeIndexInputList,
            EdgeQuery,
            slice,
            Tuple[
                Union[EdgeIndex, EdgeIndexInputList, EdgeQuery, slice],
                Union[MedRecordAttribute, MedRecordAttributeInputList, slice],
            ],
        ],
        value: Union[AttributesInput, MedRecordValue],
    ) -> None:
        if is_edge_index(key):
            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_edge_attributes([key], value)

        if isinstance(key, list):
            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_edge_attributes(key, value)

        if isinstance(key, EdgeQuery):
            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_edge_attributes(
                self._medrecord.select_edges(key), value
            )

        if isinstance(key, slice):
            if key.start is not None or key.stop is not None or key.step is not None:
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_edge_attributes(
                self._medrecord.edges, value
            )

        index_selection, attribute_selection = key

        if is_edge_index(index_selection) and is_medrecord_attribute(
            attribute_selection
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            return self._medrecord._medrecord.update_edge_attribute(
                [index_selection], attribute_selection, value
            )

        if isinstance(index_selection, list) and is_medrecord_attribute(
            attribute_selection
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            return self._medrecord._medrecord.update_edge_attribute(
                index_selection, attribute_selection, value
            )

        if isinstance(index_selection, EdgeQuery) and is_medrecord_attribute(
            attribute_selection
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            return self._medrecord._medrecord.update_edge_attribute(
                self._medrecord.select_edges(index_selection),
                attribute_selection,
                value,
            )

        if isinstance(index_selection, slice) and is_medrecord_attribute(
            attribute_selection
        ):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            return self._medrecord._medrecord.update_edge_attribute(
                self._medrecord.edges,
                attribute_selection,
                value,
            )

        if is_edge_index(index_selection) and isinstance(attribute_selection, list):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            for attribute in attribute_selection:
                self._medrecord._medrecord.update_edge_attribute(
                    [index_selection], attribute, value
                )

            return

        if isinstance(index_selection, list) and isinstance(attribute_selection, list):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            for attribute in attribute_selection:
                self._medrecord._medrecord.update_edge_attribute(
                    index_selection, attribute, value
                )

            return

        if isinstance(index_selection, EdgeQuery) and isinstance(
            attribute_selection, list
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            for attribute in attribute_selection:
                self._medrecord._medrecord.update_edge_attribute(
                    self._medrecord.select_edges(index_selection), attribute, value
                )

            return

        if isinstance(index_selection, slice) and isinstance(attribute_selection, list):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            for attribute in attribute_selection:
                self._medrecord._medrecord.update_edge_attribute(
                    self._medrecord.edges, attribute, value
                )

            return

        if is_edge_index(index_selection) and isinstance(attribute_selection, slice):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            attributes = self._medrecord._medrecord.edge([index_selection])[
                index_selection
            ]

            for attribute in attributes.keys():
                self._medrecord._medrecord.update_edge_attribute(
                    [index_selection], attribute, value
                )

            return

        if isinstance(index_selection, list) and isinstance(attribute_selection, slice):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            attributes = self._medrecord._medrecord.edge(index_selection)

            for edge in attributes.keys():
                for attribute in attributes[edge].keys():
                    self._medrecord._medrecord.update_edge_attribute(
                        [edge], attribute, value
                    )

            return

        if isinstance(index_selection, EdgeQuery) and isinstance(
            attribute_selection, slice
        ):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            attributes = self._medrecord._medrecord.edge(
                self._medrecord.select_edges(index_selection)
            )

            for edge in attributes.keys():
                for attribute in attributes[edge].keys():
                    self._medrecord._medrecord.update_edge_attribute(
                        [edge], attribute, value
                    )

            return

        if isinstance(index_selection, slice) and isinstance(
            attribute_selection, slice
        ):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
                or attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            attributes = self._medrecord._medrecord.edge(self._medrecord.edges)

            for edge in attributes.keys():
                for attribute in attributes[edge].keys():
                    self._medrecord._medrecord.update_edge_attribute(
                        [edge], attribute, value
                    )

            return

    def __delitem__(
        self,
        key: Tuple[
            Union[EdgeIndex, EdgeIndexInputList, EdgeQuery, slice],
            Union[MedRecordAttribute, MedRecordAttributeInputList, slice],
        ],
    ) -> None:
        index_selection, attribute_selection = key

        if is_edge_index(index_selection) and is_medrecord_attribute(
            attribute_selection
        ):
            return self._medrecord._medrecord.remove_edge_attribute(
                [index_selection], attribute_selection
            )

        if isinstance(index_selection, list) and is_medrecord_attribute(
            attribute_selection
        ):
            return self._medrecord._medrecord.remove_edge_attribute(
                index_selection, attribute_selection
            )

        if isinstance(index_selection, EdgeQuery) and is_medrecord_attribute(
            attribute_selection
        ):
            return self._medrecord._medrecord.remove_edge_attribute(
                self._medrecord.select_edges(index_selection),
                attribute_selection,
            )

        if isinstance(index_selection, slice) and is_medrecord_attribute(
            attribute_selection
        ):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.remove_edge_attribute(
                self._medrecord.edges,
                attribute_selection,
            )

        if is_edge_index(index_selection) and isinstance(attribute_selection, list):
            for attribute in attribute_selection:
                self._medrecord._medrecord.remove_edge_attribute(
                    [index_selection], attribute
                )

            return

        if isinstance(index_selection, list) and isinstance(attribute_selection, list):
            for attribute in attribute_selection:
                self._medrecord._medrecord.remove_edge_attribute(
                    index_selection, attribute
                )

            return

        if isinstance(index_selection, EdgeQuery) and isinstance(
            attribute_selection, list
        ):
            for attribute in attribute_selection:
                self._medrecord._medrecord.remove_edge_attribute(
                    self._medrecord.select_edges(index_selection), attribute
                )

            return

        if isinstance(index_selection, slice) and isinstance(attribute_selection, list):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            for attribute in attribute_selection:
                self._medrecord._medrecord.remove_edge_attribute(
                    self._medrecord.edges, attribute
                )

            return

        if is_edge_index(index_selection) and isinstance(attribute_selection, slice):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.replace_edge_attributes(
                [index_selection], {}
            )

        if isinstance(index_selection, list) and isinstance(attribute_selection, slice):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.replace_edge_attributes(
                index_selection, {}
            )

        if isinstance(index_selection, EdgeQuery) and isinstance(
            attribute_selection, slice
        ):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.replace_edge_attributes(
                self._medrecord.select_edges(index_selection), {}
            )

        if isinstance(index_selection, slice) and isinstance(
            attribute_selection, slice
        ):
            if (
                index_selection.start is not None
                or index_selection.stop is not None
                or index_selection.step is not None
                or attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.replace_edge_attributes(
                self._medrecord.edges, {}
            )
