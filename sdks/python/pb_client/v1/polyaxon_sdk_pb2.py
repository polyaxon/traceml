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

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: v1/polyaxon_sdk.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.api import annotations_pb2 as google_dot_api_dot_annotations__pb2
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from protoc_gen_swagger.options import annotations_pb2 as protoc__gen__swagger_dot_options_dot_annotations__pb2
from v1 import base_pb2 as v1_dot_base__pb2
from v1 import code_ref_pb2 as v1_dot_code__ref__pb2
from v1 import run_pb2 as v1_dot_run__pb2
from v1 import project_pb2 as v1_dot_project__pb2
from v1 import version_pb2 as v1_dot_version__pb2
from v1 import auth_pb2 as v1_dot_auth__pb2
from v1 import user_pb2 as v1_dot_user__pb2
from v1 import status_pb2 as v1_dot_status__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='v1/polyaxon_sdk.proto',
  package='v1',
  syntax='proto3',
  serialized_options=_b('\222A\244\002\022b\n\014Polyaxon sdk\"J\n\014Polyaxon sdk\022$https://github.com/polyaxon/polyaxon\032\024contact@polyaxon.com2\0061.14.4*\004\001\002\003\0042\020application/json:\020application/jsonR:\n\003403\0223\n1You don\'t have permission to access the resource.R)\n\003404\022\"\n\030Resource does not exist.\022\006\n\004\232\002\001\007Z\037\n\035\n\006ApiKey\022\023\010\002\032\rAuthorization \002b\014\n\n\n\006ApiKey\022\000'),
  serialized_pb=_b('\n\x15v1/polyaxon_sdk.proto\x12\x02v1\x1a\x1cgoogle/api/annotations.proto\x1a\x1bgoogle/protobuf/empty.proto\x1a,protoc-gen-swagger/options/annotations.proto\x1a\rv1/base.proto\x1a\x11v1/code_ref.proto\x1a\x0cv1/run.proto\x1a\x10v1/project.proto\x1a\x10v1/version.proto\x1a\rv1/auth.proto\x1a\rv1/user.proto\x1a\x0fv1/status.proto2\xb0\x19\n\x06RunsV1\x12m\n\x12ListBookmarkedRuns\x12\x1a.v1.UserResouceListRequest\x1a\x14.v1.ListRunsResponse\"%\x82\xd3\xe4\x93\x02\x1f\x12\x1d/api/v1/bookmarks/{user}/runs\x12j\n\x10ListArchivedRuns\x12\x1a.v1.UserResouceListRequest\x1a\x14.v1.ListRunsResponse\"$\x82\xd3\xe4\x93\x02\x1e\x12\x1c/api/v1/archives/{user}/runs\x12m\n\x08ListRuns\x12\x1e.v1.ProjectResourceListRequest\x1a\x14.v1.ListRunsResponse\"+\x82\xd3\xe4\x93\x02%\x12#/api/v1/{owner}/{project}/runs/list\x12\\\n\tCreateRun\x12\x12.v1.RunBodyRequest\x1a\x07.v1.Run\"2\x82\xd3\xe4\x93\x02,\"%/api/v1/{owner}/{project}/runs/create:\x03run\x12[\n\x06GetRun\x12\x19.v1.EntityResourceRequest\x1a\x07.v1.Run\"-\x82\xd3\xe4\x93\x02\'\x12%/api/v1/{owner}/{project}/runs/{uuid}\x12`\n\tUpdateRun\x12\x12.v1.RunBodyRequest\x1a\x07.v1.Run\"6\x82\xd3\xe4\x93\x02\x30\x1a)/api/v1/{owner}/{project}/runs/{run.uuid}:\x03run\x12_\n\x08PatchRun\x12\x12.v1.RunBodyRequest\x1a\x07.v1.Run\"6\x82\xd3\xe4\x93\x02\x30\x32)/api/v1/{owner}/{project}/runs/{run.uuid}:\x03run\x12m\n\tDeleteRun\x12\x19.v1.EntityResourceRequest\x1a\x16.google.protobuf.Empty\"-\x82\xd3\xe4\x93\x02\'*%/api/v1/{owner}/{project}/runs/{uuid}\x12\x7f\n\nDeleteRuns\x12#.v1.ProjectResourceUuidsBodyRequest\x1a\x16.google.protobuf.Empty\"4\x82\xd3\xe4\x93\x02.*%/api/v1/{owner}/{project}/runs/delete:\x05uuids\x12p\n\x07StopRun\x12\x19.v1.EntityResourceRequest\x1a\x16.google.protobuf.Empty\"2\x82\xd3\xe4\x93\x02,\"*/api/v1/{owner}/{project}/runs/{uuid}/stop\x12{\n\x08StopRuns\x12#.v1.ProjectResourceUuidsBodyRequest\x1a\x16.google.protobuf.Empty\"2\x82\xd3\xe4\x93\x02,\"#/api/v1/{owner}/{project}/runs/stop:\x05uuids\x12\x7f\n\rInvalidateRun\x12\x19.v1.EntityResourceRequest\x1a\x16.google.protobuf.Empty\";\x82\xd3\xe4\x93\x02\x35\"0/api/v1/{owner}/{project}/runs/{uuid}/invalidate:\x01*\x12\x87\x01\n\x0eInvalidateRuns\x12#.v1.ProjectResourceUuidsBodyRequest\x1a\x16.google.protobuf.Empty\"8\x82\xd3\xe4\x93\x02\x32\")/api/v1/{owner}/{project}/runs/invalidate:\x05uuids\x12z\n\x07\x43opyRun\x12\x18.v1.EntityRunBodyRequest\x1a\x07.v1.Run\"L\x82\xd3\xe4\x93\x02\x46\"?/api/v1/{entity.owner}/{entity.project}/runs/{entity.uuid}/copy:\x03run\x12\x80\x01\n\nRestartRun\x12\x18.v1.EntityRunBodyRequest\x1a\x07.v1.Run\"O\x82\xd3\xe4\x93\x02I\"B/api/v1/{entity.owner}/{entity.project}/runs/{entity.uuid}/restart:\x03run\x12~\n\tResumeRun\x12\x18.v1.EntityRunBodyRequest\x1a\x07.v1.Run\"N\x82\xd3\xe4\x93\x02H\"A/api/v1/{entity.owner}/{entity.project}/runs/{entity.uuid}/resume:\x03run\x12v\n\nArchiveRun\x12\x19.v1.EntityResourceRequest\x1a\x16.google.protobuf.Empty\"5\x82\xd3\xe4\x93\x02/\"-/api/v1/{owner}/{project}/runs/{uuid}/archive\x12v\n\nRestoreRun\x12\x19.v1.EntityResourceRequest\x1a\x16.google.protobuf.Empty\"5\x82\xd3\xe4\x93\x02/\"-/api/v1/{owner}/{project}/runs/{uuid}/restore\x12x\n\x0b\x42ookmarkRun\x12\x19.v1.EntityResourceRequest\x1a\x16.google.protobuf.Empty\"6\x82\xd3\xe4\x93\x02\x30\"./api/v1/{owner}/{project}/runs/{uuid}/bookmark\x12|\n\rUnbookmarkRun\x12\x19.v1.EntityResourceRequest\x1a\x16.google.protobuf.Empty\"8\x82\xd3\xe4\x93\x02\x32*0/api/v1/{owner}/{project}/runs/{uuid}/unbookmark\x12\x8c\x01\n\x13StartRunTensorboard\x12\x19.v1.EntityResourceRequest\x1a\x16.google.protobuf.Empty\"B\x82\xd3\xe4\x93\x02<\"7/api/v1/{owner}/{project}/runs/{uuid}/tensorboard/start:\x01*\x12\x87\x01\n\x12StopRunTensorboard\x12\x19.v1.EntityResourceRequest\x1a\x16.google.protobuf.Empty\">\x82\xd3\xe4\x93\x02\x38*6/api/v1/{owner}/{project}/runs/{uuid}/tensorboard/stop\x12o\n\x0eGetRunStatuses\x12\x19.v1.EntityResourceRequest\x1a\n.v1.Status\"6\x82\xd3\xe4\x93\x02\x30\x12./api/v1/{owner}/{project}/runs/{uuid}/statuses\x12u\n\x0f\x43reateRunStatus\x12\x1b.v1.EntityStatusBodyRequest\x1a\n.v1.Status\"9\x82\xd3\xe4\x93\x02\x33\"./api/v1/{owner}/{project}/runs/{uuid}/statuses:\x01*\x12|\n\x0eGetRunCodeRefs\x12\x19.v1.EntityResourceRequest\x1a\x18.v1.ListCodeRefsResponse\"5\x82\xd3\xe4\x93\x02/\x12-/api/v1/{owner}/{project}/runs/{uuid}/coderef\x12\x99\x01\n\x10\x43reateRunCodeRef\x12\x16.v1.CodeRefBodyRequest\x1a\x11.v1.CodeReference\"Z\x82\xd3\xe4\x93\x02T\"B/api/v1/{entity.owner}/{entity.project}/runs/{entity.uuid}/coderef:\x0e\x63ode_reference\x12r\n\x10ImpersonateToken\x12\x19.v1.EntityResourceRequest\x1a\x08.v1.Auth\"9\x82\xd3\xe4\x93\x02\x33\"1/api/v1/{owner}/{project}/runs/{uuid}/impersonate2\x81\r\n\nProjectsV1\x12l\n\x0cListProjects\x12\x1b.v1.OwnerResouceListRequest\x1a\x18.v1.ListProjectsResponse\"%\x82\xd3\xe4\x93\x02\x1f\x12\x1d/api/v1/{owner}/projects/list\x12q\n\x10ListProjectNames\x12\x1b.v1.OwnerResouceListRequest\x1a\x18.v1.ListProjectsResponse\"&\x82\xd3\xe4\x93\x02 \x12\x1e/api/v1/{owner}/projects/names\x12y\n\x16ListBookmarkedProjects\x12\x1a.v1.UserResouceListRequest\x1a\x18.v1.ListProjectsResponse\")\x82\xd3\xe4\x93\x02#\x12!/api/v1/bookmarks/{user}/projects\x12v\n\x14ListArchivedProjects\x12\x1a.v1.UserResouceListRequest\x1a\x18.v1.ListProjectsResponse\"(\x82\xd3\xe4\x93\x02\"\x12 /api/v1/archives/{user}/projects\x12\x66\n\rCreateProject\x12\x16.v1.ProjectBodyRequest\x1a\x0b.v1.Project\"0\x82\xd3\xe4\x93\x02*\"\x1f/api/v1/{owner}/projects/create:\x07project\x12X\n\nGetProject\x12\x1a.v1.ProjectResourceRequest\x1a\x0b.v1.Project\"!\x82\xd3\xe4\x93\x02\x1b\x12\x19/api/v1/{owner}/{project}\x12\x65\n\rUpdateProject\x12\x16.v1.ProjectBodyRequest\x1a\x0b.v1.Project\"/\x82\xd3\xe4\x93\x02)\x1a\x1e/api/v1/{owner}/{project.name}:\x07project\x12\x64\n\x0cPatchProject\x12\x16.v1.ProjectBodyRequest\x1a\x0b.v1.Project\"/\x82\xd3\xe4\x93\x02)2\x1e/api/v1/{owner}/{project.name}:\x07project\x12\x66\n\rDeleteProject\x12\x1a.v1.ProjectResourceRequest\x1a\x16.google.protobuf.Empty\"!\x82\xd3\xe4\x93\x02\x1b*\x19/api/v1/{owner}/{project}\x12o\n\x0e\x41rchiveProject\x12\x1a.v1.ProjectResourceRequest\x1a\x16.google.protobuf.Empty\")\x82\xd3\xe4\x93\x02#\"!/api/v1/{owner}/{project}/archive\x12r\n\x11RestoreExperiment\x12\x1a.v1.ProjectResourceRequest\x1a\x16.google.protobuf.Empty\")\x82\xd3\xe4\x93\x02#\"!/api/v1/{owner}/{project}/restore\x12q\n\x0f\x42ookmarkProject\x12\x1a.v1.ProjectResourceRequest\x1a\x16.google.protobuf.Empty\"*\x82\xd3\xe4\x93\x02$\"\"/api/v1/{owner}/{project}/bookmark\x12u\n\x11UnbookmarkProject\x12\x1a.v1.ProjectResourceRequest\x1a\x16.google.protobuf.Empty\",\x82\xd3\xe4\x93\x02&*$/api/v1/{owner}/{project}/unbookmark\x12k\n\x0f\x45nableProjectCI\x12\x1a.v1.ProjectResourceRequest\x1a\x16.google.protobuf.Empty\"$\x82\xd3\xe4\x93\x02\x1e\"\x1c/api/v1/{owner}/{project}/ci\x12l\n\x10\x44isableProjectCI\x12\x1a.v1.ProjectResourceRequest\x1a\x16.google.protobuf.Empty\"$\x82\xd3\xe4\x93\x02\x1e*\x1c/api/v1/{owner}/{project}/ci2Q\n\x06\x41uthV1\x12G\n\x05Login\x12\x14.v1.CredsBodyRequest\x1a\x08.v1.Auth\"\x1e\x82\xd3\xe4\x93\x02\x18\"\x13/api/v1/users/token:\x01*2M\n\x07UsersV1\x12\x42\n\x07getUser\x12\x16.google.protobuf.Empty\x1a\x08.v1.User\"\x15\x82\xd3\xe4\x93\x02\x0f\x12\r/api/v1/users2\xb1\x01\n\nVersionsV1\x12M\n\x0bGetVersions\x12\x16.google.protobuf.Empty\x1a\x0c.v1.Versions\"\x18\x82\xd3\xe4\x93\x02\x12\x12\x10/api/v1/versions\x12T\n\rGetLogHandler\x12\x16.google.protobuf.Empty\x1a\x0e.v1.LogHandler\"\x1b\x82\xd3\xe4\x93\x02\x15\x12\x13/api/v1/log_handlerB\xa8\x02\x92\x41\xa4\x02\x12\x62\n\x0cPolyaxon sdk\"J\n\x0cPolyaxon sdk\x12$https://github.com/polyaxon/polyaxon\x1a\x14\x63ontact@polyaxon.com2\x06\x31.14.4*\x04\x01\x02\x03\x04\x32\x10\x61pplication/json:\x10\x61pplication/jsonR:\n\x03\x34\x30\x33\x12\x33\n1You don\'t have permission to access the resource.R)\n\x03\x34\x30\x34\x12\"\n\x18Resource does not exist.\x12\x06\n\x04\x9a\x02\x01\x07Z\x1f\n\x1d\n\x06\x41piKey\x12\x13\x08\x02\x1a\rAuthorization \x02\x62\x0c\n\n\n\x06\x41piKey\x12\x00\x62\x06proto3')
  ,
  dependencies=[google_dot_api_dot_annotations__pb2.DESCRIPTOR,google_dot_protobuf_dot_empty__pb2.DESCRIPTOR,protoc__gen__swagger_dot_options_dot_annotations__pb2.DESCRIPTOR,v1_dot_base__pb2.DESCRIPTOR,v1_dot_code__ref__pb2.DESCRIPTOR,v1_dot_run__pb2.DESCRIPTOR,v1_dot_project__pb2.DESCRIPTOR,v1_dot_version__pb2.DESCRIPTOR,v1_dot_auth__pb2.DESCRIPTOR,v1_dot_user__pb2.DESCRIPTOR,v1_dot_status__pb2.DESCRIPTOR,])



_sym_db.RegisterFileDescriptor(DESCRIPTOR)


DESCRIPTOR._options = None

_RUNSV1 = _descriptor.ServiceDescriptor(
  name='RunsV1',
  full_name='v1.RunsV1',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=266,
  serialized_end=3514,
  methods=[
  _descriptor.MethodDescriptor(
    name='ListBookmarkedRuns',
    full_name='v1.RunsV1.ListBookmarkedRuns',
    index=0,
    containing_service=None,
    input_type=v1_dot_base__pb2._USERRESOUCELISTREQUEST,
    output_type=v1_dot_run__pb2._LISTRUNSRESPONSE,
    serialized_options=_b('\202\323\344\223\002\037\022\035/api/v1/bookmarks/{user}/runs'),
  ),
  _descriptor.MethodDescriptor(
    name='ListArchivedRuns',
    full_name='v1.RunsV1.ListArchivedRuns',
    index=1,
    containing_service=None,
    input_type=v1_dot_base__pb2._USERRESOUCELISTREQUEST,
    output_type=v1_dot_run__pb2._LISTRUNSRESPONSE,
    serialized_options=_b('\202\323\344\223\002\036\022\034/api/v1/archives/{user}/runs'),
  ),
  _descriptor.MethodDescriptor(
    name='ListRuns',
    full_name='v1.RunsV1.ListRuns',
    index=2,
    containing_service=None,
    input_type=v1_dot_base__pb2._PROJECTRESOURCELISTREQUEST,
    output_type=v1_dot_run__pb2._LISTRUNSRESPONSE,
    serialized_options=_b('\202\323\344\223\002%\022#/api/v1/{owner}/{project}/runs/list'),
  ),
  _descriptor.MethodDescriptor(
    name='CreateRun',
    full_name='v1.RunsV1.CreateRun',
    index=3,
    containing_service=None,
    input_type=v1_dot_run__pb2._RUNBODYREQUEST,
    output_type=v1_dot_run__pb2._RUN,
    serialized_options=_b('\202\323\344\223\002,\"%/api/v1/{owner}/{project}/runs/create:\003run'),
  ),
  _descriptor.MethodDescriptor(
    name='GetRun',
    full_name='v1.RunsV1.GetRun',
    index=4,
    containing_service=None,
    input_type=v1_dot_base__pb2._ENTITYRESOURCEREQUEST,
    output_type=v1_dot_run__pb2._RUN,
    serialized_options=_b('\202\323\344\223\002\'\022%/api/v1/{owner}/{project}/runs/{uuid}'),
  ),
  _descriptor.MethodDescriptor(
    name='UpdateRun',
    full_name='v1.RunsV1.UpdateRun',
    index=5,
    containing_service=None,
    input_type=v1_dot_run__pb2._RUNBODYREQUEST,
    output_type=v1_dot_run__pb2._RUN,
    serialized_options=_b('\202\323\344\223\0020\032)/api/v1/{owner}/{project}/runs/{run.uuid}:\003run'),
  ),
  _descriptor.MethodDescriptor(
    name='PatchRun',
    full_name='v1.RunsV1.PatchRun',
    index=6,
    containing_service=None,
    input_type=v1_dot_run__pb2._RUNBODYREQUEST,
    output_type=v1_dot_run__pb2._RUN,
    serialized_options=_b('\202\323\344\223\00202)/api/v1/{owner}/{project}/runs/{run.uuid}:\003run'),
  ),
  _descriptor.MethodDescriptor(
    name='DeleteRun',
    full_name='v1.RunsV1.DeleteRun',
    index=7,
    containing_service=None,
    input_type=v1_dot_base__pb2._ENTITYRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\002\'*%/api/v1/{owner}/{project}/runs/{uuid}'),
  ),
  _descriptor.MethodDescriptor(
    name='DeleteRuns',
    full_name='v1.RunsV1.DeleteRuns',
    index=8,
    containing_service=None,
    input_type=v1_dot_base__pb2._PROJECTRESOURCEUUIDSBODYREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\002.*%/api/v1/{owner}/{project}/runs/delete:\005uuids'),
  ),
  _descriptor.MethodDescriptor(
    name='StopRun',
    full_name='v1.RunsV1.StopRun',
    index=9,
    containing_service=None,
    input_type=v1_dot_base__pb2._ENTITYRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\002,\"*/api/v1/{owner}/{project}/runs/{uuid}/stop'),
  ),
  _descriptor.MethodDescriptor(
    name='StopRuns',
    full_name='v1.RunsV1.StopRuns',
    index=10,
    containing_service=None,
    input_type=v1_dot_base__pb2._PROJECTRESOURCEUUIDSBODYREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\002,\"#/api/v1/{owner}/{project}/runs/stop:\005uuids'),
  ),
  _descriptor.MethodDescriptor(
    name='InvalidateRun',
    full_name='v1.RunsV1.InvalidateRun',
    index=11,
    containing_service=None,
    input_type=v1_dot_base__pb2._ENTITYRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\0025\"0/api/v1/{owner}/{project}/runs/{uuid}/invalidate:\001*'),
  ),
  _descriptor.MethodDescriptor(
    name='InvalidateRuns',
    full_name='v1.RunsV1.InvalidateRuns',
    index=12,
    containing_service=None,
    input_type=v1_dot_base__pb2._PROJECTRESOURCEUUIDSBODYREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\0022\")/api/v1/{owner}/{project}/runs/invalidate:\005uuids'),
  ),
  _descriptor.MethodDescriptor(
    name='CopyRun',
    full_name='v1.RunsV1.CopyRun',
    index=13,
    containing_service=None,
    input_type=v1_dot_run__pb2._ENTITYRUNBODYREQUEST,
    output_type=v1_dot_run__pb2._RUN,
    serialized_options=_b('\202\323\344\223\002F\"?/api/v1/{entity.owner}/{entity.project}/runs/{entity.uuid}/copy:\003run'),
  ),
  _descriptor.MethodDescriptor(
    name='RestartRun',
    full_name='v1.RunsV1.RestartRun',
    index=14,
    containing_service=None,
    input_type=v1_dot_run__pb2._ENTITYRUNBODYREQUEST,
    output_type=v1_dot_run__pb2._RUN,
    serialized_options=_b('\202\323\344\223\002I\"B/api/v1/{entity.owner}/{entity.project}/runs/{entity.uuid}/restart:\003run'),
  ),
  _descriptor.MethodDescriptor(
    name='ResumeRun',
    full_name='v1.RunsV1.ResumeRun',
    index=15,
    containing_service=None,
    input_type=v1_dot_run__pb2._ENTITYRUNBODYREQUEST,
    output_type=v1_dot_run__pb2._RUN,
    serialized_options=_b('\202\323\344\223\002H\"A/api/v1/{entity.owner}/{entity.project}/runs/{entity.uuid}/resume:\003run'),
  ),
  _descriptor.MethodDescriptor(
    name='ArchiveRun',
    full_name='v1.RunsV1.ArchiveRun',
    index=16,
    containing_service=None,
    input_type=v1_dot_base__pb2._ENTITYRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\002/\"-/api/v1/{owner}/{project}/runs/{uuid}/archive'),
  ),
  _descriptor.MethodDescriptor(
    name='RestoreRun',
    full_name='v1.RunsV1.RestoreRun',
    index=17,
    containing_service=None,
    input_type=v1_dot_base__pb2._ENTITYRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\002/\"-/api/v1/{owner}/{project}/runs/{uuid}/restore'),
  ),
  _descriptor.MethodDescriptor(
    name='BookmarkRun',
    full_name='v1.RunsV1.BookmarkRun',
    index=18,
    containing_service=None,
    input_type=v1_dot_base__pb2._ENTITYRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\0020\"./api/v1/{owner}/{project}/runs/{uuid}/bookmark'),
  ),
  _descriptor.MethodDescriptor(
    name='UnbookmarkRun',
    full_name='v1.RunsV1.UnbookmarkRun',
    index=19,
    containing_service=None,
    input_type=v1_dot_base__pb2._ENTITYRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\0022*0/api/v1/{owner}/{project}/runs/{uuid}/unbookmark'),
  ),
  _descriptor.MethodDescriptor(
    name='StartRunTensorboard',
    full_name='v1.RunsV1.StartRunTensorboard',
    index=20,
    containing_service=None,
    input_type=v1_dot_base__pb2._ENTITYRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\002<\"7/api/v1/{owner}/{project}/runs/{uuid}/tensorboard/start:\001*'),
  ),
  _descriptor.MethodDescriptor(
    name='StopRunTensorboard',
    full_name='v1.RunsV1.StopRunTensorboard',
    index=21,
    containing_service=None,
    input_type=v1_dot_base__pb2._ENTITYRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\0028*6/api/v1/{owner}/{project}/runs/{uuid}/tensorboard/stop'),
  ),
  _descriptor.MethodDescriptor(
    name='GetRunStatuses',
    full_name='v1.RunsV1.GetRunStatuses',
    index=22,
    containing_service=None,
    input_type=v1_dot_base__pb2._ENTITYRESOURCEREQUEST,
    output_type=v1_dot_status__pb2._STATUS,
    serialized_options=_b('\202\323\344\223\0020\022./api/v1/{owner}/{project}/runs/{uuid}/statuses'),
  ),
  _descriptor.MethodDescriptor(
    name='CreateRunStatus',
    full_name='v1.RunsV1.CreateRunStatus',
    index=23,
    containing_service=None,
    input_type=v1_dot_status__pb2._ENTITYSTATUSBODYREQUEST,
    output_type=v1_dot_status__pb2._STATUS,
    serialized_options=_b('\202\323\344\223\0023\"./api/v1/{owner}/{project}/runs/{uuid}/statuses:\001*'),
  ),
  _descriptor.MethodDescriptor(
    name='GetRunCodeRefs',
    full_name='v1.RunsV1.GetRunCodeRefs',
    index=24,
    containing_service=None,
    input_type=v1_dot_base__pb2._ENTITYRESOURCEREQUEST,
    output_type=v1_dot_code__ref__pb2._LISTCODEREFSRESPONSE,
    serialized_options=_b('\202\323\344\223\002/\022-/api/v1/{owner}/{project}/runs/{uuid}/coderef'),
  ),
  _descriptor.MethodDescriptor(
    name='CreateRunCodeRef',
    full_name='v1.RunsV1.CreateRunCodeRef',
    index=25,
    containing_service=None,
    input_type=v1_dot_code__ref__pb2._CODEREFBODYREQUEST,
    output_type=v1_dot_code__ref__pb2._CODEREFERENCE,
    serialized_options=_b('\202\323\344\223\002T\"B/api/v1/{entity.owner}/{entity.project}/runs/{entity.uuid}/coderef:\016code_reference'),
  ),
  _descriptor.MethodDescriptor(
    name='ImpersonateToken',
    full_name='v1.RunsV1.ImpersonateToken',
    index=26,
    containing_service=None,
    input_type=v1_dot_base__pb2._ENTITYRESOURCEREQUEST,
    output_type=v1_dot_auth__pb2._AUTH,
    serialized_options=_b('\202\323\344\223\0023\"1/api/v1/{owner}/{project}/runs/{uuid}/impersonate'),
  ),
])
_sym_db.RegisterServiceDescriptor(_RUNSV1)

DESCRIPTOR.services_by_name['RunsV1'] = _RUNSV1


_PROJECTSV1 = _descriptor.ServiceDescriptor(
  name='ProjectsV1',
  full_name='v1.ProjectsV1',
  file=DESCRIPTOR,
  index=1,
  serialized_options=None,
  serialized_start=3517,
  serialized_end=5182,
  methods=[
  _descriptor.MethodDescriptor(
    name='ListProjects',
    full_name='v1.ProjectsV1.ListProjects',
    index=0,
    containing_service=None,
    input_type=v1_dot_base__pb2._OWNERRESOUCELISTREQUEST,
    output_type=v1_dot_project__pb2._LISTPROJECTSRESPONSE,
    serialized_options=_b('\202\323\344\223\002\037\022\035/api/v1/{owner}/projects/list'),
  ),
  _descriptor.MethodDescriptor(
    name='ListProjectNames',
    full_name='v1.ProjectsV1.ListProjectNames',
    index=1,
    containing_service=None,
    input_type=v1_dot_base__pb2._OWNERRESOUCELISTREQUEST,
    output_type=v1_dot_project__pb2._LISTPROJECTSRESPONSE,
    serialized_options=_b('\202\323\344\223\002 \022\036/api/v1/{owner}/projects/names'),
  ),
  _descriptor.MethodDescriptor(
    name='ListBookmarkedProjects',
    full_name='v1.ProjectsV1.ListBookmarkedProjects',
    index=2,
    containing_service=None,
    input_type=v1_dot_base__pb2._USERRESOUCELISTREQUEST,
    output_type=v1_dot_project__pb2._LISTPROJECTSRESPONSE,
    serialized_options=_b('\202\323\344\223\002#\022!/api/v1/bookmarks/{user}/projects'),
  ),
  _descriptor.MethodDescriptor(
    name='ListArchivedProjects',
    full_name='v1.ProjectsV1.ListArchivedProjects',
    index=3,
    containing_service=None,
    input_type=v1_dot_base__pb2._USERRESOUCELISTREQUEST,
    output_type=v1_dot_project__pb2._LISTPROJECTSRESPONSE,
    serialized_options=_b('\202\323\344\223\002\"\022 /api/v1/archives/{user}/projects'),
  ),
  _descriptor.MethodDescriptor(
    name='CreateProject',
    full_name='v1.ProjectsV1.CreateProject',
    index=4,
    containing_service=None,
    input_type=v1_dot_project__pb2._PROJECTBODYREQUEST,
    output_type=v1_dot_project__pb2._PROJECT,
    serialized_options=_b('\202\323\344\223\002*\"\037/api/v1/{owner}/projects/create:\007project'),
  ),
  _descriptor.MethodDescriptor(
    name='GetProject',
    full_name='v1.ProjectsV1.GetProject',
    index=5,
    containing_service=None,
    input_type=v1_dot_base__pb2._PROJECTRESOURCEREQUEST,
    output_type=v1_dot_project__pb2._PROJECT,
    serialized_options=_b('\202\323\344\223\002\033\022\031/api/v1/{owner}/{project}'),
  ),
  _descriptor.MethodDescriptor(
    name='UpdateProject',
    full_name='v1.ProjectsV1.UpdateProject',
    index=6,
    containing_service=None,
    input_type=v1_dot_project__pb2._PROJECTBODYREQUEST,
    output_type=v1_dot_project__pb2._PROJECT,
    serialized_options=_b('\202\323\344\223\002)\032\036/api/v1/{owner}/{project.name}:\007project'),
  ),
  _descriptor.MethodDescriptor(
    name='PatchProject',
    full_name='v1.ProjectsV1.PatchProject',
    index=7,
    containing_service=None,
    input_type=v1_dot_project__pb2._PROJECTBODYREQUEST,
    output_type=v1_dot_project__pb2._PROJECT,
    serialized_options=_b('\202\323\344\223\002)2\036/api/v1/{owner}/{project.name}:\007project'),
  ),
  _descriptor.MethodDescriptor(
    name='DeleteProject',
    full_name='v1.ProjectsV1.DeleteProject',
    index=8,
    containing_service=None,
    input_type=v1_dot_base__pb2._PROJECTRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\002\033*\031/api/v1/{owner}/{project}'),
  ),
  _descriptor.MethodDescriptor(
    name='ArchiveProject',
    full_name='v1.ProjectsV1.ArchiveProject',
    index=9,
    containing_service=None,
    input_type=v1_dot_base__pb2._PROJECTRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\002#\"!/api/v1/{owner}/{project}/archive'),
  ),
  _descriptor.MethodDescriptor(
    name='RestoreExperiment',
    full_name='v1.ProjectsV1.RestoreExperiment',
    index=10,
    containing_service=None,
    input_type=v1_dot_base__pb2._PROJECTRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\002#\"!/api/v1/{owner}/{project}/restore'),
  ),
  _descriptor.MethodDescriptor(
    name='BookmarkProject',
    full_name='v1.ProjectsV1.BookmarkProject',
    index=11,
    containing_service=None,
    input_type=v1_dot_base__pb2._PROJECTRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\002$\"\"/api/v1/{owner}/{project}/bookmark'),
  ),
  _descriptor.MethodDescriptor(
    name='UnbookmarkProject',
    full_name='v1.ProjectsV1.UnbookmarkProject',
    index=12,
    containing_service=None,
    input_type=v1_dot_base__pb2._PROJECTRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\002&*$/api/v1/{owner}/{project}/unbookmark'),
  ),
  _descriptor.MethodDescriptor(
    name='EnableProjectCI',
    full_name='v1.ProjectsV1.EnableProjectCI',
    index=13,
    containing_service=None,
    input_type=v1_dot_base__pb2._PROJECTRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\002\036\"\034/api/v1/{owner}/{project}/ci'),
  ),
  _descriptor.MethodDescriptor(
    name='DisableProjectCI',
    full_name='v1.ProjectsV1.DisableProjectCI',
    index=14,
    containing_service=None,
    input_type=v1_dot_base__pb2._PROJECTRESOURCEREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=_b('\202\323\344\223\002\036*\034/api/v1/{owner}/{project}/ci'),
  ),
])
_sym_db.RegisterServiceDescriptor(_PROJECTSV1)

DESCRIPTOR.services_by_name['ProjectsV1'] = _PROJECTSV1


_AUTHV1 = _descriptor.ServiceDescriptor(
  name='AuthV1',
  full_name='v1.AuthV1',
  file=DESCRIPTOR,
  index=2,
  serialized_options=None,
  serialized_start=5184,
  serialized_end=5265,
  methods=[
  _descriptor.MethodDescriptor(
    name='Login',
    full_name='v1.AuthV1.Login',
    index=0,
    containing_service=None,
    input_type=v1_dot_auth__pb2._CREDSBODYREQUEST,
    output_type=v1_dot_auth__pb2._AUTH,
    serialized_options=_b('\202\323\344\223\002\030\"\023/api/v1/users/token:\001*'),
  ),
])
_sym_db.RegisterServiceDescriptor(_AUTHV1)

DESCRIPTOR.services_by_name['AuthV1'] = _AUTHV1


_USERSV1 = _descriptor.ServiceDescriptor(
  name='UsersV1',
  full_name='v1.UsersV1',
  file=DESCRIPTOR,
  index=3,
  serialized_options=None,
  serialized_start=5267,
  serialized_end=5344,
  methods=[
  _descriptor.MethodDescriptor(
    name='getUser',
    full_name='v1.UsersV1.getUser',
    index=0,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=v1_dot_user__pb2._USER,
    serialized_options=_b('\202\323\344\223\002\017\022\r/api/v1/users'),
  ),
])
_sym_db.RegisterServiceDescriptor(_USERSV1)

DESCRIPTOR.services_by_name['UsersV1'] = _USERSV1


_VERSIONSV1 = _descriptor.ServiceDescriptor(
  name='VersionsV1',
  full_name='v1.VersionsV1',
  file=DESCRIPTOR,
  index=4,
  serialized_options=None,
  serialized_start=5347,
  serialized_end=5524,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetVersions',
    full_name='v1.VersionsV1.GetVersions',
    index=0,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=v1_dot_version__pb2._VERSIONS,
    serialized_options=_b('\202\323\344\223\002\022\022\020/api/v1/versions'),
  ),
  _descriptor.MethodDescriptor(
    name='GetLogHandler',
    full_name='v1.VersionsV1.GetLogHandler',
    index=1,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=v1_dot_version__pb2._LOGHANDLER,
    serialized_options=_b('\202\323\344\223\002\025\022\023/api/v1/log_handler'),
  ),
])
_sym_db.RegisterServiceDescriptor(_VERSIONSV1)

DESCRIPTOR.services_by_name['VersionsV1'] = _VERSIONSV1

# @@protoc_insertion_point(module_scope)
