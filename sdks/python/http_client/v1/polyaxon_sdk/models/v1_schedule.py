#!/usr/bin/python
#
# Copyright 2018-2020 Polyaxon, Inc.
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

    The version of the OpenAPI document: 1.2.0-rc3
    Contact: contact@polyaxon.com
    Generated by: https://openapi-generator.tech
"""


import pprint
import re  # noqa: F401

import six

from polyaxon_sdk.configuration import Configuration


class V1Schedule(object):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    """
    Attributes:
      openapi_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    openapi_types = {
        "cron": "V1CronSchedule",
        "exact_time": "V1ExactTimeSchedule",
        "interval": "V1IntervalSchedule",
        "repeatable": "V1RepeatableSchedule",
    }

    attribute_map = {
        "cron": "cron",
        "exact_time": "exact_time",
        "interval": "interval",
        "repeatable": "repeatable",
    }

    def __init__(
        self,
        cron=None,
        exact_time=None,
        interval=None,
        repeatable=None,
        local_vars_configuration=None,
    ):  # noqa: E501
        """V1Schedule - a model defined in OpenAPI"""  # noqa: E501
        if local_vars_configuration is None:
            local_vars_configuration = Configuration()
        self.local_vars_configuration = local_vars_configuration

        self._cron = None
        self._exact_time = None
        self._interval = None
        self._repeatable = None
        self.discriminator = None

        if cron is not None:
            self.cron = cron
        if exact_time is not None:
            self.exact_time = exact_time
        if interval is not None:
            self.interval = interval
        if repeatable is not None:
            self.repeatable = repeatable

    @property
    def cron(self):
        """Gets the cron of this V1Schedule.  # noqa: E501


        :return: The cron of this V1Schedule.  # noqa: E501
        :rtype: V1CronSchedule
        """
        return self._cron

    @cron.setter
    def cron(self, cron):
        """Sets the cron of this V1Schedule.


        :param cron: The cron of this V1Schedule.  # noqa: E501
        :type: V1CronSchedule
        """

        self._cron = cron

    @property
    def exact_time(self):
        """Gets the exact_time of this V1Schedule.  # noqa: E501


        :return: The exact_time of this V1Schedule.  # noqa: E501
        :rtype: V1ExactTimeSchedule
        """
        return self._exact_time

    @exact_time.setter
    def exact_time(self, exact_time):
        """Sets the exact_time of this V1Schedule.


        :param exact_time: The exact_time of this V1Schedule.  # noqa: E501
        :type: V1ExactTimeSchedule
        """

        self._exact_time = exact_time

    @property
    def interval(self):
        """Gets the interval of this V1Schedule.  # noqa: E501


        :return: The interval of this V1Schedule.  # noqa: E501
        :rtype: V1IntervalSchedule
        """
        return self._interval

    @interval.setter
    def interval(self, interval):
        """Sets the interval of this V1Schedule.


        :param interval: The interval of this V1Schedule.  # noqa: E501
        :type: V1IntervalSchedule
        """

        self._interval = interval

    @property
    def repeatable(self):
        """Gets the repeatable of this V1Schedule.  # noqa: E501


        :return: The repeatable of this V1Schedule.  # noqa: E501
        :rtype: V1RepeatableSchedule
        """
        return self._repeatable

    @repeatable.setter
    def repeatable(self, repeatable):
        """Sets the repeatable of this V1Schedule.


        :param repeatable: The repeatable of this V1Schedule.  # noqa: E501
        :type: V1RepeatableSchedule
        """

        self._repeatable = repeatable

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.openapi_types):
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

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, V1Schedule):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, V1Schedule):
            return True

        return self.to_dict() != other.to_dict()
