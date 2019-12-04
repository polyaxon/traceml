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


class V1ComponentRef(object):
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
    swagger_types = {"name": "str", "url": "str", "path": "str", "hub": "str"}

    attribute_map = {"name": "name", "url": "url", "path": "path", "hub": "hub"}

    def __init__(self, name=None, url=None, path=None, hub=None):  # noqa: E501
        """V1ComponentRef - a model defined in Swagger"""  # noqa: E501

        self._name = None
        self._url = None
        self._path = None
        self._hub = None
        self.discriminator = None

        if name is not None:
            self.name = name
        if url is not None:
            self.url = url
        if path is not None:
            self.path = path
        if hub is not None:
            self.hub = hub

    @property
    def name(self):
        """Gets the name of this V1ComponentRef.  # noqa: E501


        :return: The name of this V1ComponentRef.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this V1ComponentRef.


        :param name: The name of this V1ComponentRef.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def url(self):
        """Gets the url of this V1ComponentRef.  # noqa: E501


        :return: The url of this V1ComponentRef.  # noqa: E501
        :rtype: str
        """
        return self._url

    @url.setter
    def url(self, url):
        """Sets the url of this V1ComponentRef.


        :param url: The url of this V1ComponentRef.  # noqa: E501
        :type: str
        """

        self._url = url

    @property
    def path(self):
        """Gets the path of this V1ComponentRef.  # noqa: E501


        :return: The path of this V1ComponentRef.  # noqa: E501
        :rtype: str
        """
        return self._path

    @path.setter
    def path(self, path):
        """Sets the path of this V1ComponentRef.


        :param path: The path of this V1ComponentRef.  # noqa: E501
        :type: str
        """

        self._path = path

    @property
    def hub(self):
        """Gets the hub of this V1ComponentRef.  # noqa: E501


        :return: The hub of this V1ComponentRef.  # noqa: E501
        :rtype: str
        """
        return self._hub

    @hub.setter
    def hub(self, hub):
        """Sets the hub of this V1ComponentRef.


        :param hub: The hub of this V1ComponentRef.  # noqa: E501
        :type: str
        """

        self._hub = hub

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
        if issubclass(V1ComponentRef, dict):
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
        if not isinstance(other, V1ComponentRef):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
