#!/usr/bin/python
#
# Copyright 2018-2023 Polyaxon, Inc.
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
import uuid

from typing import Dict, List, Optional

from pydantic import StrictStr

from polyaxon.schemas.base import BaseSchemaModel
from polyaxon.schemas.fields import UUIDStr
from traceml.artifacts.kinds import V1ArtifactKind


class V1RunArtifact(BaseSchemaModel):
    _IDENTIFIER = "artifact"

    name: Optional[StrictStr]
    kind: Optional[V1ArtifactKind]
    path: Optional[StrictStr]
    state: Optional[UUIDStr]
    summary: Optional[Dict]
    meta_info: Optional[Dict]
    run: Optional[UUIDStr]
    connection: Optional[StrictStr]
    is_input: Optional[bool]

    @classmethod
    def from_model(cls, model):
        return cls(
            name=model.name,
            kind=model.kind,
            path=model.path,
            state=model.state,
            summary=model.summary,
            # connection=model.connection,  # TODO: enable
        )

    def get_state(self, namespace: uuid.UUID):
        if self.state:
            return self.state
        summary = self.summary or {}
        content = str(summary)
        if not summary.get("hash") and self.path:
            content += self.path
        return uuid.uuid5(namespace, content)


class V1RunArtifacts(BaseSchemaModel):
    _IDENTIFIER = "artifacts"

    artifacts: Optional[List[V1RunArtifact]]
