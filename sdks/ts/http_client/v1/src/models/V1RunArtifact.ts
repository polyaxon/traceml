// Copyright 2018-2021 Polyaxon, Inc.
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
 * The version of the OpenAPI document: 1.9.3
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
 * @interface V1RunArtifact
 */
export interface V1RunArtifact {
    /**
     * 
     * @type {string}
     * @memberof V1RunArtifact
     */
    name?: string;
    /**
     * 
     * @type {string}
     * @memberof V1RunArtifact
     */
    state?: string;
    /**
     * 
     * @type {V1ArtifactKind}
     * @memberof V1RunArtifact
     */
    kind?: V1ArtifactKind;
    /**
     * 
     * @type {string}
     * @memberof V1RunArtifact
     */
    path?: string;
    /**
     * 
     * @type {string}
     * @memberof V1RunArtifact
     */
    connection?: string;
    /**
     * 
     * @type {object}
     * @memberof V1RunArtifact
     */
    summary?: object;
    /**
     * 
     * @type {boolean}
     * @memberof V1RunArtifact
     */
    is_input?: boolean;
}

export function V1RunArtifactFromJSON(json: any): V1RunArtifact {
    return V1RunArtifactFromJSONTyped(json, false);
}

export function V1RunArtifactFromJSONTyped(json: any, ignoreDiscriminator: boolean): V1RunArtifact {
    if ((json === undefined) || (json === null)) {
        return json;
    }
    return {
        
        'name': !exists(json, 'name') ? undefined : json['name'],
        'state': !exists(json, 'state') ? undefined : json['state'],
        'kind': !exists(json, 'kind') ? undefined : V1ArtifactKindFromJSON(json['kind']),
        'path': !exists(json, 'path') ? undefined : json['path'],
        'connection': !exists(json, 'connection') ? undefined : json['connection'],
        'summary': !exists(json, 'summary') ? undefined : json['summary'],
        'is_input': !exists(json, 'is_input') ? undefined : json['is_input'],
    };
}

export function V1RunArtifactToJSON(value?: V1RunArtifact | null): any {
    if (value === undefined) {
        return undefined;
    }
    if (value === null) {
        return null;
    }
    return {
        
        'name': value.name,
        'state': value.state,
        'kind': V1ArtifactKindToJSON(value.kind),
        'path': value.path,
        'connection': value.connection,
        'summary': value.summary,
        'is_input': value.is_input,
    };
}


