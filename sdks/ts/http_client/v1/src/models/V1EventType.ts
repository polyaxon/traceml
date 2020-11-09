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
 * The version of the OpenAPI document: 1.3.1
 * Contact: contact@polyaxon.com
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */

import { exists, mapValues } from '../runtime';
import {
    V1ArtifactKind,
    V1ArtifactKindFromJSON,
    V1ArtifactKindFromJSONTyped,
    V1ArtifactKindToJSON,
} from './';

/**
 * 
 * @export
 * @interface V1EventType
 */
export interface V1EventType {
    /**
     * 
     * @type {string}
     * @memberof V1EventType
     */
    name?: string;
    /**
     * 
     * @type {V1ArtifactKind}
     * @memberof V1EventType
     */
    kind?: V1ArtifactKind;
}

export function V1EventTypeFromJSON(json: any): V1EventType {
    return V1EventTypeFromJSONTyped(json, false);
}

export function V1EventTypeFromJSONTyped(json: any, ignoreDiscriminator: boolean): V1EventType {
    if ((json === undefined) || (json === null)) {
        return json;
    }
    return {
        
        'name': !exists(json, 'name') ? undefined : json['name'],
        'kind': !exists(json, 'kind') ? undefined : V1ArtifactKindFromJSON(json['kind']),
    };
}

export function V1EventTypeToJSON(value?: V1EventType | null): any {
    if (value === undefined) {
        return undefined;
    }
    if (value === null) {
        return null;
    }
    return {
        
        'name': value.name,
        'kind': V1ArtifactKindToJSON(value.kind),
    };
}


