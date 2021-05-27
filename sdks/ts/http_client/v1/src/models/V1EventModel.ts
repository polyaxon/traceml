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
/**
 * 
 * @export
 * @interface V1EventModel
 */
export interface V1EventModel {
    /**
     * 
     * @type {string}
     * @memberof V1EventModel
     */
    framework?: string;
    /**
     * 
     * @type {string}
     * @memberof V1EventModel
     */
    path?: string;
    /**
     * 
     * @type {object}
     * @memberof V1EventModel
     */
    spec?: object;
}

export function V1EventModelFromJSON(json: any): V1EventModel {
    return V1EventModelFromJSONTyped(json, false);
}

export function V1EventModelFromJSONTyped(json: any, ignoreDiscriminator: boolean): V1EventModel {
    if ((json === undefined) || (json === null)) {
        return json;
    }
    return {
        
        'framework': !exists(json, 'framework') ? undefined : json['framework'],
        'path': !exists(json, 'path') ? undefined : json['path'],
        'spec': !exists(json, 'spec') ? undefined : json['spec'],
    };
}

export function V1EventModelToJSON(value?: V1EventModel | null): any {
    if (value === undefined) {
        return undefined;
    }
    if (value === null) {
        return null;
    }
    return {
        
        'framework': value.framework,
        'path': value.path,
        'spec': value.spec,
    };
}


