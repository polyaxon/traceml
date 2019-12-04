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
from __future__ import absolute_import, division, print_function

from marshmallow import fields, validate
from polyaxon_sdk import V1OpIOCondition

from polyaxon.schemas.base import BaseConfig, BaseSchema
from polyaxon.schemas.polyflow.trigger_policies import ExpressionTriggerPolicy


class OpIOConditionSchema(BaseSchema):
    kind = fields.Str(allow_none=True)
    op = fields.Str(required=True)
    name = fields.Str(required=True)
    trigger = fields.Str(required=True)

    @staticmethod
    def schema_config():
        return OpIOConditionConfig


class OpIOConditionConfig(BaseConfig, V1OpIOCondition):
    SCHEMA = OpIOConditionSchema
    IDENTIFIER = "outputs"
    IDENTIFIER_KIND = True

    @staticmethod
    def schema_config():
        return OpIOConditionSchema


class OpInputsConditionSchema(OpIOConditionSchema):
    kind = fields.Str(allow_none=True, validate=validate.Equal("inputs"))

    @staticmethod
    def schema_config():
        return OpInputsConditionConfig


class OpInputsConditionConfig(OpIOConditionConfig):
    SCHEMA = OpInputsConditionSchema
    IDENTIFIER = "outputs"


class OpOutputsConditionSchema(OpIOConditionSchema):
    kind = fields.Str(allow_none=True, validate=validate.Equal("outputs"))

    @staticmethod
    def schema_config():
        return OpOutputsConditionConfig


class OpOutputsConditionConfig(OpIOConditionConfig):
    SCHEMA = OpOutputsConditionSchema
    IDENTIFIER = "outputs"
