from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple, Union, overload

from medmodels.medrecord.querying import EdgeOperation, NodeOperation
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
            NodeOperation,
            slice,
            Tuple[
                Union[NodeIndexInputList, NodeOperation, slice],
                Union[MedRecordAttributeInputList, slice],
            ],
        ],
    ) -> Dict[NodeIndex, Attributes]: ...

    @overload
    def __getitem__(
        self,
        key: Tuple[Union[NodeIndexInputList, NodeOperation, slice], MedRecordAttribute],
    ) -> Dict[NodeIndex, MedRecordValue]: ...

    def __getitem__(
        self,
        key: Union[
            NodeIndex,
            NodeIndexInputList,
            NodeOperation,
            slice,
            Tuple[
                Union[NodeIndex, NodeIndexInputList, NodeOperation, slice],
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

        if isinstance(key, NodeOperation):
            return self._medrecord._medrecord.node(self._medrecord.select_nodes(key))

        if isinstance(key, slice):
            if key.start is not None or key.stop is not None or key.step is not None:
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.node(self._medrecord.nodes)

        if not isinstance(key, tuple):
            raise TypeError("Invalid key type")

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

        if isinstance(index_selection, NodeOperation) and is_medrecord_attribute(
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

        if isinstance(index_selection, NodeOperation) and isinstance(
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

        if isinstance(index_selection, NodeOperation) and isinstance(
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
        key: Union[NodeIndex, NodeIndexInputList, NodeOperation, slice],
        value: AttributesInput,
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        key: Tuple[
            Union[NodeIndex, NodeIndexInputList, NodeOperation, slice],
            Union[MedRecordAttribute, MedRecordAttributeInputList, slice],
        ],
        value: MedRecordValue,
    ) -> None: ...

    def __setitem__(
        self,
        key: Union[
            NodeIndex,
            NodeIndexInputList,
            NodeOperation,
            slice,
            Tuple[
                Union[NodeIndex, NodeIndexInputList, NodeOperation, slice],
                Union[MedRecordAttribute, MedRecordAttributeInputList, slice],
            ],
        ],
        value: Union[AttributesInput, MedRecordValue],
    ) -> None:
        if is_node_index(key):
            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_node_attributes(value, [key])

        if isinstance(key, list):
            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_node_attributes(value, key)

        if isinstance(key, NodeOperation):
            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_node_attributes(
                value, self._medrecord.select_nodes(key)
            )

        if isinstance(key, slice):
            if key.start is not None or key.stop is not None or key.step is not None:
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_node_attributes(
                value, self._medrecord.nodes
            )

        if not isinstance(key, tuple):
            raise TypeError("Invalid key type")

        index_selection, attribute_selection = key

        if is_node_index(index_selection) and is_medrecord_attribute(
            attribute_selection
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            return self._medrecord._medrecord.update_node_attribute(
                attribute_selection, value, [index_selection]
            )

        if isinstance(index_selection, list) and is_medrecord_attribute(
            attribute_selection
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            return self._medrecord._medrecord.update_node_attribute(
                attribute_selection, value, index_selection
            )

        if isinstance(index_selection, NodeOperation) and is_medrecord_attribute(
            attribute_selection
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            return self._medrecord._medrecord.update_node_attribute(
                attribute_selection,
                value,
                self._medrecord.select_nodes(index_selection),
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
                attribute_selection,
                value,
                self._medrecord.nodes,
            )

        if is_node_index(index_selection) and isinstance(attribute_selection, list):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            for attribute in attribute_selection:
                self._medrecord._medrecord.update_node_attribute(
                    attribute, value, [index_selection]
                )

            return

        if isinstance(index_selection, list) and isinstance(attribute_selection, list):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            for attribute in attribute_selection:
                self._medrecord._medrecord.update_node_attribute(
                    attribute, value, index_selection
                )

            return

        if isinstance(index_selection, NodeOperation) and isinstance(
            attribute_selection, list
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            for attribute in attribute_selection:
                self._medrecord._medrecord.update_node_attribute(
                    attribute, value, self._medrecord.select_nodes(index_selection)
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
                    attribute, value, self._medrecord.nodes
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
                    attribute, value, [index_selection]
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
                        attribute, value, [node]
                    )

            return

        if isinstance(index_selection, NodeOperation) and isinstance(
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
                        attribute, value, [node]
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
                        attribute, value, [node]
                    )

            return

    def __delitem__(
        self,
        key: Tuple[
            Union[NodeIndex, NodeIndexInputList, NodeOperation, slice],
            Union[MedRecordAttribute, MedRecordAttributeInputList, slice],
        ],
    ) -> None:
        index_selection, attribute_selection = key

        if is_node_index(index_selection) and is_medrecord_attribute(
            attribute_selection
        ):
            return self._medrecord._medrecord.remove_node_attribute(
                attribute_selection, [index_selection]
            )

        if isinstance(index_selection, list) and is_medrecord_attribute(
            attribute_selection
        ):
            return self._medrecord._medrecord.remove_node_attribute(
                attribute_selection, index_selection
            )

        if isinstance(index_selection, NodeOperation) and is_medrecord_attribute(
            attribute_selection
        ):
            return self._medrecord._medrecord.remove_node_attribute(
                attribute_selection,
                self._medrecord.select_nodes(index_selection),
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
                attribute_selection,
                self._medrecord.nodes,
            )

        if is_node_index(index_selection) and isinstance(attribute_selection, list):
            for attribute in attribute_selection:
                self._medrecord._medrecord.remove_node_attribute(
                    attribute, [index_selection]
                )

            return

        if isinstance(index_selection, list) and isinstance(attribute_selection, list):
            for attribute in attribute_selection:
                self._medrecord._medrecord.remove_node_attribute(
                    attribute, index_selection
                )

            return

        if isinstance(index_selection, NodeOperation) and isinstance(
            attribute_selection, list
        ):
            for attribute in attribute_selection:
                self._medrecord._medrecord.remove_node_attribute(
                    attribute, self._medrecord.select_nodes(index_selection)
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
                    attribute, self._medrecord.nodes
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
                {}, [index_selection]
            )

        if isinstance(index_selection, list) and isinstance(attribute_selection, slice):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.replace_node_attributes(
                {}, index_selection
            )

        if isinstance(index_selection, NodeOperation) and isinstance(
            attribute_selection, slice
        ):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.replace_node_attributes(
                {}, self._medrecord.select_nodes(index_selection)
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
                {}, self._medrecord.nodes
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
            EdgeOperation,
            slice,
            Tuple[
                Union[EdgeIndexInputList, EdgeOperation, slice],
                Union[MedRecordAttributeInputList, slice],
            ],
        ],
    ) -> Dict[EdgeIndex, Attributes]: ...

    @overload
    def __getitem__(
        self,
        key: Tuple[Union[EdgeIndexInputList, EdgeOperation, slice], MedRecordAttribute],
    ) -> Dict[EdgeIndex, MedRecordValue]: ...

    def __getitem__(
        self,
        key: Union[
            EdgeIndex,
            EdgeIndexInputList,
            EdgeOperation,
            slice,
            Tuple[
                Union[EdgeIndex, EdgeIndexInputList, EdgeOperation, slice],
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

        if isinstance(key, EdgeOperation):
            return self._medrecord._medrecord.edge(self._medrecord.select_edges(key))

        if isinstance(key, slice):
            if key.start is not None or key.stop is not None or key.step is not None:
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.edge(self._medrecord.edges)

        if not isinstance(key, tuple):
            raise TypeError("Invalid key type")

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

        if isinstance(index_selection, EdgeOperation) and is_medrecord_attribute(
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

        if isinstance(index_selection, EdgeOperation) and isinstance(
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

        if isinstance(index_selection, EdgeOperation) and isinstance(
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
        key: Union[EdgeIndex, EdgeIndexInputList, EdgeOperation, slice],
        value: AttributesInput,
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        key: Tuple[
            Union[EdgeIndex, EdgeIndexInputList, EdgeOperation, slice],
            Union[MedRecordAttribute, MedRecordAttributeInputList, slice],
        ],
        value: MedRecordValue,
    ) -> None: ...

    def __setitem__(
        self,
        key: Union[
            EdgeIndex,
            EdgeIndexInputList,
            EdgeOperation,
            slice,
            Tuple[
                Union[EdgeIndex, EdgeIndexInputList, EdgeOperation, slice],
                Union[MedRecordAttribute, MedRecordAttributeInputList, slice],
            ],
        ],
        value: Union[AttributesInput, MedRecordValue],
    ) -> None:
        if is_edge_index(key):
            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_edge_attributes(value, [key])

        if isinstance(key, list):
            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_edge_attributes(value, key)

        if isinstance(key, EdgeOperation):
            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_edge_attributes(
                value, self._medrecord.select_edges(key)
            )

        if isinstance(key, slice):
            if key.start is not None or key.stop is not None or key.step is not None:
                raise ValueError("Invalid slice, only ':' is allowed")

            if not is_attributes(value):
                raise ValueError("Invalid value type. Expected Attributes")

            return self._medrecord._medrecord.replace_edge_attributes(
                value, self._medrecord.edges
            )

        if not isinstance(key, tuple):
            raise TypeError("Invalid key type")

        index_selection, attribute_selection = key

        if is_edge_index(index_selection) and is_medrecord_attribute(
            attribute_selection
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            return self._medrecord._medrecord.update_edge_attribute(
                attribute_selection, value, [index_selection]
            )

        if isinstance(index_selection, list) and is_medrecord_attribute(
            attribute_selection
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            return self._medrecord._medrecord.update_edge_attribute(
                attribute_selection, value, index_selection
            )

        if isinstance(index_selection, EdgeOperation) and is_medrecord_attribute(
            attribute_selection
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            return self._medrecord._medrecord.update_edge_attribute(
                attribute_selection,
                value,
                self._medrecord.select_edges(index_selection),
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
                attribute_selection,
                value,
                self._medrecord.edges,
            )

        if is_edge_index(index_selection) and isinstance(attribute_selection, list):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            for attribute in attribute_selection:
                self._medrecord._medrecord.update_edge_attribute(
                    attribute, value, [index_selection]
                )

            return

        if isinstance(index_selection, list) and isinstance(attribute_selection, list):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            for attribute in attribute_selection:
                self._medrecord._medrecord.update_edge_attribute(
                    attribute, value, index_selection
                )

            return

        if isinstance(index_selection, EdgeOperation) and isinstance(
            attribute_selection, list
        ):
            if not is_medrecord_value(value):
                raise ValueError("Invalid value type. Expected MedRecordValue")

            for attribute in attribute_selection:
                self._medrecord._medrecord.update_edge_attribute(
                    attribute, value, self._medrecord.select_edges(index_selection)
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
                    attribute, value, self._medrecord.edges
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
                    attribute, value, [index_selection]
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
                        attribute, value, [edge]
                    )

            return

        if isinstance(index_selection, EdgeOperation) and isinstance(
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
                        attribute, value, [edge]
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
                        attribute, value, [edge]
                    )

            return

    def __delitem__(
        self,
        key: Tuple[
            Union[EdgeIndex, EdgeIndexInputList, EdgeOperation, slice],
            Union[MedRecordAttribute, MedRecordAttributeInputList, slice],
        ],
    ) -> None:
        index_selection, attribute_selection = key

        if is_edge_index(index_selection) and is_medrecord_attribute(
            attribute_selection
        ):
            return self._medrecord._medrecord.remove_edge_attribute(
                attribute_selection, [index_selection]
            )

        if isinstance(index_selection, list) and is_medrecord_attribute(
            attribute_selection
        ):
            return self._medrecord._medrecord.remove_edge_attribute(
                attribute_selection, index_selection
            )

        if isinstance(index_selection, EdgeOperation) and is_medrecord_attribute(
            attribute_selection
        ):
            return self._medrecord._medrecord.remove_edge_attribute(
                attribute_selection,
                self._medrecord.select_edges(index_selection),
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
                attribute_selection,
                self._medrecord.edges,
            )

        if is_edge_index(index_selection) and isinstance(attribute_selection, list):
            for attribute in attribute_selection:
                self._medrecord._medrecord.remove_edge_attribute(
                    attribute, [index_selection]
                )

            return

        if isinstance(index_selection, list) and isinstance(attribute_selection, list):
            for attribute in attribute_selection:
                self._medrecord._medrecord.remove_edge_attribute(
                    attribute, index_selection
                )

            return

        if isinstance(index_selection, EdgeOperation) and isinstance(
            attribute_selection, list
        ):
            for attribute in attribute_selection:
                self._medrecord._medrecord.remove_edge_attribute(
                    attribute, self._medrecord.select_edges(index_selection)
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
                    attribute, self._medrecord.edges
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
                {}, [index_selection]
            )

        if isinstance(index_selection, list) and isinstance(attribute_selection, slice):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.replace_edge_attributes(
                {}, index_selection
            )

        if isinstance(index_selection, EdgeOperation) and isinstance(
            attribute_selection, slice
        ):
            if (
                attribute_selection.start is not None
                or attribute_selection.stop is not None
                or attribute_selection.step is not None
            ):
                raise ValueError("Invalid slice, only ':' is allowed")

            return self._medrecord._medrecord.replace_edge_attributes(
                {}, self._medrecord.select_edges(index_selection)
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
                {}, self._medrecord.edges
            )
