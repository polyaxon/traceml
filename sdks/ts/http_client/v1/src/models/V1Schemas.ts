// Copyright 2018-2020 Polyaxon, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* tslint:disable */
/* eslint-disable */
/**
 * Polyaxon SDKs and REST API specification.
 * Polyaxon SDKs and REST API specification.
 *
 * The version of the OpenAPI document: 1.2.0-rc3
 * Contact: contact@polyaxon.com
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */

import { exists, mapValues } from '../runtime';
import {
    V1ArtifactsMount,
    V1ArtifactsMountFromJSON,
    V1ArtifactsMountFromJSONTyped,
    V1ArtifactsMountToJSON,
    V1ArtifactsType,
    V1ArtifactsTypeFromJSON,
    V1ArtifactsTypeFromJSONTyped,
    V1ArtifactsTypeToJSON,
    V1AuthType,
    V1AuthTypeFromJSON,
    V1AuthTypeFromJSONTyped,
    V1AuthTypeToJSON,
    V1CompiledOperation,
    V1CompiledOperationFromJSON,
    V1CompiledOperationFromJSONTyped,
    V1CompiledOperationToJSON,
    V1ConnectionSchema,
    V1ConnectionSchemaFromJSON,
    V1ConnectionSchemaFromJSONTyped,
    V1ConnectionSchemaToJSON,
    V1ConnectionType,
    V1ConnectionTypeFromJSON,
    V1ConnectionTypeFromJSONTyped,
    V1ConnectionTypeToJSON,
    V1DockerfileType,
    V1DockerfileTypeFromJSON,
    V1DockerfileTypeFromJSONTyped,
    V1DockerfileTypeToJSON,
    V1EarlyStopping,
    V1EarlyStoppingFromJSON,
    V1EarlyStoppingFromJSONTyped,
    V1EarlyStoppingToJSON,
    V1Event,
    V1EventFromJSON,
    V1EventFromJSONTyped,
    V1EventToJSON,
    V1EventType,
    V1EventTypeFromJSON,
    V1EventTypeFromJSONTyped,
    V1EventTypeToJSON,
    V1GcsType,
    V1GcsTypeFromJSON,
    V1GcsTypeFromJSONTyped,
    V1GcsTypeToJSON,
    V1GitType,
    V1GitTypeFromJSON,
    V1GitTypeFromJSONTyped,
    V1GitTypeToJSON,
    V1HpParams,
    V1HpParamsFromJSON,
    V1HpParamsFromJSONTyped,
    V1HpParamsToJSON,
    V1K8sResourceType,
    V1K8sResourceTypeFromJSON,
    V1K8sResourceTypeFromJSONTyped,
    V1K8sResourceTypeToJSON,
    V1Matrix,
    V1MatrixFromJSON,
    V1MatrixFromJSONTyped,
    V1MatrixToJSON,
    V1MatrixKind,
    V1MatrixKindFromJSON,
    V1MatrixKindFromJSONTyped,
    V1MatrixKindToJSON,
    V1Operation,
    V1OperationFromJSON,
    V1OperationFromJSONTyped,
    V1OperationToJSON,
    V1OperationCond,
    V1OperationCondFromJSON,
    V1OperationCondFromJSONTyped,
    V1OperationCondToJSON,
    V1PolyaxonInitContainer,
    V1PolyaxonInitContainerFromJSON,
    V1PolyaxonInitContainerFromJSONTyped,
    V1PolyaxonInitContainerToJSON,
    V1PolyaxonSidecarContainer,
    V1PolyaxonSidecarContainerFromJSON,
    V1PolyaxonSidecarContainerFromJSONTyped,
    V1PolyaxonSidecarContainerToJSON,
    V1Reference,
    V1ReferenceFromJSON,
    V1ReferenceFromJSONTyped,
    V1ReferenceToJSON,
    V1RunSchema,
    V1RunSchemaFromJSON,
    V1RunSchemaFromJSONTyped,
    V1RunSchemaToJSON,
    V1S3Type,
    V1S3TypeFromJSON,
    V1S3TypeFromJSONTyped,
    V1S3TypeToJSON,
    V1Schedule,
    V1ScheduleFromJSON,
    V1ScheduleFromJSONTyped,
    V1ScheduleToJSON,
    V1UriType,
    V1UriTypeFromJSON,
    V1UriTypeFromJSONTyped,
    V1UriTypeToJSON,
    V1WasbType,
    V1WasbTypeFromJSON,
    V1WasbTypeFromJSONTyped,
    V1WasbTypeToJSON,
} from './';

/**
 * 
 * @export
 * @interface V1Schemas
 */
export interface V1Schemas {
    /**
     * 
     * @type {V1OperationCond}
     * @memberof V1Schemas
     */
    operation_cond?: V1OperationCond;
    /**
     * 
     * @type {V1EarlyStopping}
     * @memberof V1Schemas
     */
    early_stopping?: V1EarlyStopping;
    /**
     * 
     * @type {V1Matrix}
     * @memberof V1Schemas
     */
    matrix?: V1Matrix;
    /**
     * 
     * @type {V1RunSchema}
     * @memberof V1Schemas
     */
    run?: V1RunSchema;
    /**
     * 
     * @type {V1Operation}
     * @memberof V1Schemas
     */
    operation?: V1Operation;
    /**
     * 
     * @type {V1CompiledOperation}
     * @memberof V1Schemas
     */
    compiled_operation?: V1CompiledOperation;
    /**
     * 
     * @type {V1Schedule}
     * @memberof V1Schemas
     */
    schedule?: V1Schedule;
    /**
     * 
     * @type {V1ConnectionSchema}
     * @memberof V1Schemas
     */
    connection_schema?: V1ConnectionSchema;
    /**
     * 
     * @type {V1HpParams}
     * @memberof V1Schemas
     */
    hp_params?: V1HpParams;
    /**
     * 
     * @type {V1Reference}
     * @memberof V1Schemas
     */
    reference?: V1Reference;
    /**
     * 
     * @type {V1ArtifactsMount}
     * @memberof V1Schemas
     */
    artifacts_mount?: V1ArtifactsMount;
    /**
     * 
     * @type {V1PolyaxonSidecarContainer}
     * @memberof V1Schemas
     */
    polyaxon_sidecar_container?: V1PolyaxonSidecarContainer;
    /**
     * 
     * @type {V1PolyaxonInitContainer}
     * @memberof V1Schemas
     */
    polyaxon_init_container?: V1PolyaxonInitContainer;
    /**
     * 
     * @type {V1ArtifactsType}
     * @memberof V1Schemas
     */
    artifacs?: V1ArtifactsType;
    /**
     * 
     * @type {V1WasbType}
     * @memberof V1Schemas
     */
    wasb?: V1WasbType;
    /**
     * 
     * @type {V1GcsType}
     * @memberof V1Schemas
     */
    gcs?: V1GcsType;
    /**
     * 
     * @type {V1S3Type}
     * @memberof V1Schemas
     */
    s3?: V1S3Type;
    /**
     * 
     * @type {V1AuthType}
     * @memberof V1Schemas
     */
    autg?: V1AuthType;
    /**
     * 
     * @type {V1DockerfileType}
     * @memberof V1Schemas
     */
    dockerfile?: V1DockerfileType;
    /**
     * 
     * @type {V1GitType}
     * @memberof V1Schemas
     */
    git?: V1GitType;
    /**
     * 
     * @type {V1UriType}
     * @memberof V1Schemas
     */
    uri?: V1UriType;
    /**
     * 
     * @type {V1K8sResourceType}
     * @memberof V1Schemas
     */
    k8s_resource?: V1K8sResourceType;
    /**
     * 
     * @type {V1ConnectionType}
     * @memberof V1Schemas
     */
    connection?: V1ConnectionType;
    /**
     * 
     * @type {V1EventType}
     * @memberof V1Schemas
     */
    event_type?: V1EventType;
    /**
     * 
     * @type {V1Event}
     * @memberof V1Schemas
     */
    event?: V1Event;
    /**
     * 
     * @type {V1MatrixKind}
     * @memberof V1Schemas
     */
    matrix_kind?: V1MatrixKind;
}

export function V1SchemasFromJSON(json: any): V1Schemas {
    return V1SchemasFromJSONTyped(json, false);
}

export function V1SchemasFromJSONTyped(json: any, ignoreDiscriminator: boolean): V1Schemas {
    if ((json === undefined) || (json === null)) {
        return json;
    }
    return {
        
        'operation_cond': !exists(json, 'operation_cond') ? undefined : V1OperationCondFromJSON(json['operation_cond']),
        'early_stopping': !exists(json, 'early_stopping') ? undefined : V1EarlyStoppingFromJSON(json['early_stopping']),
        'matrix': !exists(json, 'matrix') ? undefined : V1MatrixFromJSON(json['matrix']),
        'run': !exists(json, 'run') ? undefined : V1RunSchemaFromJSON(json['run']),
        'operation': !exists(json, 'operation') ? undefined : V1OperationFromJSON(json['operation']),
        'compiled_operation': !exists(json, 'compiled_operation') ? undefined : V1CompiledOperationFromJSON(json['compiled_operation']),
        'schedule': !exists(json, 'schedule') ? undefined : V1ScheduleFromJSON(json['schedule']),
        'connection_schema': !exists(json, 'connection_schema') ? undefined : V1ConnectionSchemaFromJSON(json['connection_schema']),
        'hp_params': !exists(json, 'hp_params') ? undefined : V1HpParamsFromJSON(json['hp_params']),
        'reference': !exists(json, 'reference') ? undefined : V1ReferenceFromJSON(json['reference']),
        'artifacts_mount': !exists(json, 'artifacts_mount') ? undefined : V1ArtifactsMountFromJSON(json['artifacts_mount']),
        'polyaxon_sidecar_container': !exists(json, 'polyaxon_sidecar_container') ? undefined : V1PolyaxonSidecarContainerFromJSON(json['polyaxon_sidecar_container']),
        'polyaxon_init_container': !exists(json, 'polyaxon_init_container') ? undefined : V1PolyaxonInitContainerFromJSON(json['polyaxon_init_container']),
        'artifacs': !exists(json, 'artifacs') ? undefined : V1ArtifactsTypeFromJSON(json['artifacs']),
        'wasb': !exists(json, 'wasb') ? undefined : V1WasbTypeFromJSON(json['wasb']),
        'gcs': !exists(json, 'gcs') ? undefined : V1GcsTypeFromJSON(json['gcs']),
        's3': !exists(json, 's3') ? undefined : V1S3TypeFromJSON(json['s3']),
        'autg': !exists(json, 'autg') ? undefined : V1AuthTypeFromJSON(json['autg']),
        'dockerfile': !exists(json, 'dockerfile') ? undefined : V1DockerfileTypeFromJSON(json['dockerfile']),
        'git': !exists(json, 'git') ? undefined : V1GitTypeFromJSON(json['git']),
        'uri': !exists(json, 'uri') ? undefined : V1UriTypeFromJSON(json['uri']),
        'k8s_resource': !exists(json, 'k8s_resource') ? undefined : V1K8sResourceTypeFromJSON(json['k8s_resource']),
        'connection': !exists(json, 'connection') ? undefined : V1ConnectionTypeFromJSON(json['connection']),
        'event_type': !exists(json, 'event_type') ? undefined : V1EventTypeFromJSON(json['event_type']),
        'event': !exists(json, 'event') ? undefined : V1EventFromJSON(json['event']),
        'matrix_kind': !exists(json, 'matrix_kind') ? undefined : V1MatrixKindFromJSON(json['matrix_kind']),
    };
}

export function V1SchemasToJSON(value?: V1Schemas | null): any {
    if (value === undefined) {
        return undefined;
    }
    if (value === null) {
        return null;
    }
    return {
        
        'operation_cond': V1OperationCondToJSON(value.operation_cond),
        'early_stopping': V1EarlyStoppingToJSON(value.early_stopping),
        'matrix': V1MatrixToJSON(value.matrix),
        'run': V1RunSchemaToJSON(value.run),
        'operation': V1OperationToJSON(value.operation),
        'compiled_operation': V1CompiledOperationToJSON(value.compiled_operation),
        'schedule': V1ScheduleToJSON(value.schedule),
        'connection_schema': V1ConnectionSchemaToJSON(value.connection_schema),
        'hp_params': V1HpParamsToJSON(value.hp_params),
        'reference': V1ReferenceToJSON(value.reference),
        'artifacts_mount': V1ArtifactsMountToJSON(value.artifacts_mount),
        'polyaxon_sidecar_container': V1PolyaxonSidecarContainerToJSON(value.polyaxon_sidecar_container),
        'polyaxon_init_container': V1PolyaxonInitContainerToJSON(value.polyaxon_init_container),
        'artifacs': V1ArtifactsTypeToJSON(value.artifacs),
        'wasb': V1WasbTypeToJSON(value.wasb),
        'gcs': V1GcsTypeToJSON(value.gcs),
        's3': V1S3TypeToJSON(value.s3),
        'autg': V1AuthTypeToJSON(value.autg),
        'dockerfile': V1DockerfileTypeToJSON(value.dockerfile),
        'git': V1GitTypeToJSON(value.git),
        'uri': V1UriTypeToJSON(value.uri),
        'k8s_resource': V1K8sResourceTypeToJSON(value.k8s_resource),
        'connection': V1ConnectionTypeToJSON(value.connection),
        'event_type': V1EventTypeToJSON(value.event_type),
        'event': V1EventToJSON(value.event),
        'matrix_kind': V1MatrixKindToJSON(value.matrix_kind),
    };
}


