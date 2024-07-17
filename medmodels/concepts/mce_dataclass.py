from typing import Tuple, Optional, List, Generator
import torch
import pandas as pd
import numpy as np
import random

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import Group, MedRecordAttribute, NodeIndex


class DataClassMCE:
    def __init__(
        self,
        medrecord: MedRecord,
        patients_group: Group = "patients",
        concepts_group: Group = "medical_concepts",
        time_attribute: MedRecordAttribute = "time",
        time_interval_days: int = 7,
        time_scope_radius: int = 50,
        max_num_events: int = 50,
        min_num_events: Optional[int] = None,
        seed: int = 42,
    ):
        """Constructor of the data class

        :param data: Data to hold
        :type data: pd.DataFrame
        :param time_interval: Time interval to discretize time periods
            (e.g. time_interval = 7 -> weeks, time_interval=30 -> months)
        :type time_interval: int
        :parm time_scope_radius: Look n time intervals back and forward to search for
            related concepts. Default = 50
        :type time_scope_radius: int
        :param max_events: Limit of events to consider as context for computational
            reasons
        :type max_events: int
        :param min_total_events: Minimum number of events to consider a code
        :type min_total_events: int
        :param seed: Seed for random number generator. Defaults to 42.
        :type seed: int, optional
        """
        self.patients = self.medrecord.nodes_in_group(self.patients_group)
        self.concept_counter = {
            concept_name: len(medrecord.edges_connecting(self.patients, concept_name))
            for concept_name in medrecord.nodes_in_group(concepts_group)
        }

        self.medrecord = medrecord
        if min_num_events is not None:
            self._remove_low_frequency_codes(min_num_events)

        self.patients_group = patients_group
        self.concepts_group = concepts_group
        self.time_scope_radius = time_scope_radius
        self.time_interval_days = time_interval_days
        self.max_num_events = max_num_events

        self.concept_name_to_idx_dict = {
            concept_name: idx
            for idx, concept_name in enumerate(self.concept_counter.keys())
        }
        self.negative_sampling_table, self.expanded_negative_sampling_table = (
            self._get_negative_sampling_table()
        )
        self.earliest_date = self._find_earliest_date(time_attribute=time_attribute)

        # shuffle chains
        self.seed = seed
        random.Random(self.seed).shuffle(self.patients)

    def __repr__(self) -> str:
        """Representation function of the class

        :return: Description with number of unique patients, events and entries
        :rtype: str
        """
        return (
            "DataClassMCE with {} unique patients, {} unique events, {} entries".format(
                len(self.patients),
                len(self.medrecord.nodes_in_group(self.concepts_group)),
                sum(self.concept_counter.values()),
            )
        )

    def _find_earliest_date(self, time_attribute: MedRecordAttribute) -> pd.Timestamp:
        edges = [
            edge
            for concept_name in self.medrecord.nodes_in_group(self.concepts_group)
            for edge in self.medrecord.edges_connecting(self.patients, concept_name)
        ]
        edges_attributes = self.medrecord.edge[edges].values()

        if not all(
            time_attribute in edge_attribute for edge_attribute in edges_attributes
        ):
            raise ValueError("Time attribute not found in the edge attributes")

        edge_times = [
            pd.to_datetime(str(edge_attribute[time_attribute]))
            for edge_attribute in edges_attributes
        ]

        return min(edge_times)

    def _remove_low_frequency_codes(self, min_num_events: int) -> None:
        """Remove codes with less than min_num_events occurrences

        :param min_num_events: Minimum number of events to consider a code
        :type min_num_events: int
        """
        if max(self.concept_counter.values()) < min_num_events:
            raise (ValueError("Not enough events to apply min_num_events criterion"))

        deleted_nodes = [
            k for k, v in self.concept_counter.items() if v < min_num_events
        ]
        self.concept_counter = {
            k: v for k, v in self.concept_counter.items() if v >= min_num_events
        }
        self.medrecord.remove_node(deleted_nodes)

    def _get_negative_sampling_table(
        self,
    ) -> Tuple[List[Tuple[NodeIndex, int, int]], torch.Tensor]:
        """Get the negative sampling table and the expanded negative sampling table.

        The negative sampling table contains the name of the concept, its frequency,
        and the probability of being sampled as a negative example.
        The expanded negative sampling table repeats the ids of the tokens depending on
        these probabilities.

        :return: Negative sampling table, Expanded negative sampling table
        :rtype: (list, torch.tensor)
        """
        # Sort the frequencies, then select the most frequent concepts as vocabulary.
        freqs_sorted = sorted(
            self.concept_counter.items(), key=lambda p: (p[1], p[0]), reverse=True
        )

        # Now, we'll compute the negative sampling table.
        # The negative sampling probabilities are proportional to the frequencies
        # to the power of a constant (typically 0.75).
        negative_sampling_table = {}
        sum_frequencies = 0

        for idx, frequency in freqs_sorted:
            negative_sampling_frequency = frequency**0.75
            negative_sampling_table[idx] = negative_sampling_frequency
            sum_frequencies += negative_sampling_frequency

        # Convert the negative sampling probabilities to integers, in order to make
        # sampling faster and easier.
        # We return a list of tuples consisting of:
        # - the concept name
        # - its frequency in the training data
        # - number of positions reserved for the token in the negative sampling table
        scaler = 1e6 / sum_frequencies
        negative_sampling_table = [
            (
                concept_name,
                freq,
                int(round(negative_sampling_table[concept_name] * scaler)),
            )
            for concept_name, freq in freqs_sorted
        ]

        # Expand the negative sampling table to a list of indices.
        negative_sampling_table_expanded = []
        for i, (_, _, count) in enumerate(negative_sampling_table):
            negative_sampling_table_expanded.extend([i] * count)

        return negative_sampling_table, torch.as_tensor(
            negative_sampling_table_expanded
        )

    def idx_to_concept_name(self, *i: int) -> List[NodeIndex]:
        """Translate a list of ids to it corresponding tokens (original texts)

        :param i: ids to be translated to tokens (original texts)
        :type i: int
        :return: list of tokens
        :rtype: list
        """
        return list(map(list(self.concept_counter.keys()).__getitem__, i))

    def concept_name_to_idx(self, *ts: NodeIndex) -> List[int]:
        """Translate a list of tokens (original texts) to it corresponding ids

        :param ts: tokens (original texts) to be translated to ids
        :type ts: str
        :return: list of ids for given tokens
        :rtype: np.ndarray
        """
        return [self.concept_name_to_idx_dict.get(t, -1) for t in ts]

    def zip_event_time(self, row: pd.DataFrame) -> list:
        """Zips events and times in dataframe for apply function

        :param row: Row of a pandas dataframe
        :type row: pd.DataFrame

        :return: Zipped columns
        :rtype: list
        """
        return list(zip(row["event"], row["time_window"]))

    def loader(
        self, start: float = 0.0, end: float = 1.0, batch_size: int = 1024
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Iterator to yield data batchwise from DataClassMCE

        :param start: Position in data to start, e.g. to manage train- and
            test data [0,1]. Defaults to 0.0.
        :type start: float, optional
        :param end: Position in data to end, e.g. to manage train- and test data [0,1].
            Defaults to 1.0.
        :type end: float, optional
        :param batch_size: Number of instances per batch. Defaults to 1024.
        :type batch_size: int, optional

        :return: Batch of Training data
        :rtype: torch.tensor

        :yield: Iterator[torch.tensor]: Batch of Training data
        """

        self.token_count = 0

        target = []
        context = []
        chains = self.chains[
            int(start * len(self.chains)) : int(end * len(self.chains))
        ]

        rng = np.random.default_rng(self.seed)
        self.seed += 1
        # iterate over every chain
        for chain in chains:
            chain = np.array(chain, dtype="int")

            for i, chain_content in enumerate(chain):
                event, time = chain_content

                # Get a subchain of all events that are not
                # the focused event itset (id=i) and are within
                # the pre-defined time_scope_radius
                sub_chain = np.delete(chain, i, axis=0)
                context_events = sub_chain[
                    np.where(
                        np.logical_and(
                            (sub_chain[:, 1:] >= (time - self.time_scope_radius)),
                            (sub_chain[:, 1:] <= (time + self.time_scope_radius)),
                        )
                    )[0],
                    :,
                ]

                # reduce to max num events if necessary, if not, pad with -1
                if len(context_events) >= self.max_num_events:
                    if len(context_events) >= self.max_num_events:
                        context_events = rng.choice(
                            context_events, self.max_num_events, replace=False
                        )
                else:
                    # apply padding to context events if they are less than max_num_events
                    missing_length = self.max_num_events - len(context_events)
                    context_events = np.append(
                        context_events,
                        np.array([[-1, -1] for _ in range(missing_length)]),
                        axis=0,
                    )

                target.append([event, time])
                context.append(context_events)

                if len(target) == batch_size:
                    # now we are removing irrelevant context length
                    # (e.g. if the whole batch has no more than k context elements
                    # but the maximum allow context is J and k < J
                    # then it is inefficient to deliver useless contextspace if it is
                    # not required for this batch
                    context = np.array(context)
                    target = np.array(target)

                    # this is the maximum number of context info in this batch
                    # so shorten the context dim for this batch to this number
                    max_len = np.max(np.sum(np.where(context[:, :, :1] == -1, 0, 1), 1))

                    yield (
                        torch.as_tensor(context[:, :max_len], dtype=torch.long),
                        torch.as_tensor(target, dtype=torch.long),
                    )

                    # After coming back, reset the batch.
                    target = []
                    context = []

        if len(target) > 0:
            # Yield the final batch.
            context = np.array(context)
            target = np.array(target)

            # this is the maximum number of context info in this batch
            # so shorten the context dim for this batch to this number
            max_len = np.max(np.sum(np.where(context[:, :, :1] == -1, 0, 1), 1))

            yield (
                torch.as_tensor(context[:, :max_len], dtype=torch.long),
                torch.as_tensor(target, dtype=torch.long),
            )
