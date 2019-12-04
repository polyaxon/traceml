#!/usr/bin/python
#
# Copyright 2019 Polyaxon, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8

"""
    Polyaxon SDKs and REST API specification.

    Polyaxon SDKs and REST API specification.  # noqa: E501

    OpenAPI spec version: 1.0.0
    Contact: contact@polyaxon.com
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


import pprint
import re  # noqa: F401

import six


class V1Dag(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        "kind": "str",
        "ops": "list[V1Op]",
        "components": "list[V1Component]",
        "concurrency": "int",
        "early_stopping": "list[object]",
    }

    attribute_map = {
        "kind": "kind",
        "ops": "ops",
        "components": "components",
        "concurrency": "concurrency",
        "early_stopping": "early_stopping",
    }

    def __init__(
        self,
        kind=None,
        ops=None,
        components=None,
        concurrency=None,
        early_stopping=None,
    ):  # noqa: E501
        """V1Dag - a model defined in Swagger"""  # noqa: E501

        self._kind = None
        self._ops = None
        self._components = None
        self._concurrency = None
        self._early_stopping = None
        self.discriminator = None

        if kind is not None:
            self.kind = kind
        if ops is not None:
            self.ops = ops
        if components is not None:
            self.components = components
        if concurrency is not None:
            self.concurrency = concurrency
        if early_stopping is not None:
            self.early_stopping = early_stopping

    @property
    def kind(self):
        """Gets the kind of this V1Dag.  # noqa: E501


        :return: The kind of this V1Dag.  # noqa: E501
        :rtype: str
        """
        return self._kind

    @kind.setter
    def kind(self, kind):
        """Sets the kind of this V1Dag.


        :param kind: The kind of this V1Dag.  # noqa: E501
        :type: str
        """

        self._kind = kind

    @property
    def ops(self):
        """Gets the ops of this V1Dag.  # noqa: E501


        :return: The ops of this V1Dag.  # noqa: E501
        :rtype: list[V1Op]
        """
        return self._ops

    @ops.setter
    def ops(self, ops):
        """Sets the ops of this V1Dag.


        :param ops: The ops of this V1Dag.  # noqa: E501
        :type: list[V1Op]
        """

        self._ops = ops

    @property
    def components(self):
        """Gets the components of this V1Dag.  # noqa: E501


        :return: The components of this V1Dag.  # noqa: E501
        :rtype: list[V1Component]
        """
        return self._components

    @components.setter
    def components(self, components):
        """Sets the components of this V1Dag.


        :param components: The components of this V1Dag.  # noqa: E501
        :type: list[V1Component]
        """

        self._components = components

    @property
    def concurrency(self):
        """Gets the concurrency of this V1Dag.  # noqa: E501


        :return: The concurrency of this V1Dag.  # noqa: E501
        :rtype: int
        """
        return self._concurrency

    @concurrency.setter
    def concurrency(self, concurrency):
        """Sets the concurrency of this V1Dag.


        :param concurrency: The concurrency of this V1Dag.  # noqa: E501
        :type: int
        """

        self._concurrency = concurrency

    @property
    def early_stopping(self):
        """Gets the early_stopping of this V1Dag.  # noqa: E501


        :return: The early_stopping of this V1Dag.  # noqa: E501
        :rtype: list[object]
        """
        return self._early_stopping

    @early_stopping.setter
    def early_stopping(self, early_stopping):
        """Sets the early_stopping of this V1Dag.


        :param early_stopping: The early_stopping of this V1Dag.  # noqa: E501
        :type: list[object]
        """

        self._early_stopping = early_stopping

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(
                    map(lambda x: x.to_dict() if hasattr(x, "to_dict") else x, value)
                )
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(
                    map(
                        lambda item: (item[0], item[1].to_dict())
                        if hasattr(item[1], "to_dict")
                        else item,
                        value.items(),
                    )
                )
            else:
                result[attr] = value
        if issubclass(V1Dag, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, V1Dag):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
